package mobile

import (
	"encoding/json"
	"fmt"
	"runtime"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// Version is the GopherLLM library version.
func Version() string { return gopherllm.Version }

// MetalAvailable reports whether this build can run the Metal GPU kernels
// the "metal" load option enables.
func MetalAvailable() bool { return gopherllm.MetalAvailable() }

// RuntimeInfoJSON describes the build and device for diagnostics screens:
// {"version","goos","goarch","cpus","metal_available","metal_status"}.
func RuntimeInfoJSON() string {
	status := "available"
	if !gopherllm.MetalAvailable() {
		status = gopherllm.MetalError()
	}
	return marshal(struct {
		Version        string `json:"version"`
		GOOS           string `json:"goos"`
		GOARCH         string `json:"goarch"`
		CPUs           int    `json:"cpus"`
		MetalAvailable bool   `json:"metal_available"`
		MetalStatus    string `json:"metal_status"`
	}{gopherllm.Version, runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), gopherllm.MetalAvailable(), status})
}

// modelSummary is what a GGUF header says about a model; InspectModel
// returns it and InfoJSON starts from it.
type modelSummary struct {
	Name         string `json:"name"`
	Architecture string `json:"architecture"`
	// Supported reports whether this build can run the architecture.
	Supported  bool  `json:"supported"`
	Parameters int64 `json:"parameters"`
	// TensorBytes is the weights' size, a lower bound for the memory they
	// occupy once loaded.
	TensorBytes   int64   `json:"tensor_bytes"`
	BitsPerWeight float64 `json:"bits_per_weight"`
	// Quantization is the tensor type holding most of the weight bytes.
	Quantization  string `json:"quantization"`
	ContextLength int    `json:"context_length"`
	Layers        int    `json:"layers"`
	VocabSize     int    `json:"vocab_size"`
	ChatTemplate  string `json:"chat_template"`
	// KVCacheBytesPerToken is the attention cache's growth per context token
	// (f32 storage, the default); multiply by prompt plus reply length to
	// budget memory for a conversation.
	KVCacheBytesPerToken int64 `json:"kv_cache_bytes_per_token"`
}

func summarize(a *gopherllm.Analysis) modelSummary {
	s := modelSummary{
		Name:          a.Name,
		Architecture:  a.Architecture,
		Supported:     a.Supported,
		Parameters:    a.Params,
		TensorBytes:   a.FileBytes,
		BitsPerWeight: a.BitsPerWeight,
		ContextLength: a.ContextLength,
		Layers:        a.Layers,
		VocabSize:     a.VocabSize,
		ChatTemplate:  a.TemplateKind,
	}
	if len(a.DTypes) > 0 {
		s.Quantization = a.DTypes[0].Type.String()
	}
	if a.ContextLength > 0 {
		s.KVCacheBytesPerToken = a.KVCacheBytesAtFullContext / int64(a.ContextLength)
	}
	return s
}

// InspectModel reads only the GGUF header at path (no weights, so it is fast
// for any file size) and returns a modelSummary as JSON: name, architecture,
// supported, parameters, tensor_bytes, bits_per_weight, quantization,
// context_length, layers, vocab_size, chat_template and
// kv_cache_bytes_per_token.
func InspectModel(path string) (_ string, err error) {
	defer recoverError("failed to inspect GGUF", &err)
	mapping, err := gopherllm.OpenMmap(path)
	if err != nil {
		return "", fmt.Errorf("failed to open GGUF: %w", err)
	}
	defer mapping.Close()
	gguf, err := gopherllm.ParseGGUFQuiet(mapping.Bytes())
	if err != nil {
		return "", fmt.Errorf("failed to parse GGUF: %w", err)
	}
	tok, _ := gopherllm.TokenizerFromMetadata(gguf.Metadata) // only refines chat_template
	return marshal(summarize(gopherllm.AnalyzeGGUF(gguf, tok))), nil
}

func loadedModelInfo(m *gopherllm.Model) string {
	c, info := m.Config(), m.Info()
	s := summarize(gopherllm.AnalyzeGGUF(m.GGUF(), m.Tokenizer()))
	s.Architecture, s.ContextLength, s.VocabSize = c.Arch, c.MaxSeqLen, c.VocabSize
	return marshal(struct {
		modelSummary
		FileSizeBytes  int   `json:"file_size_bytes"`
		Mapped         bool  `json:"mapped"`
		OutOfCore      bool  `json:"out_of_core"`
		LoadTimeMS     int64 `json:"load_time_ms"`
		MetalAvailable bool  `json:"metal_available"`
	}{s, info.FileSizeBytes, m.IsMapped(), info.OutOfCore, info.LoadTime.Milliseconds(), gopherllm.MetalAvailable()})
}

// resultJSON is the JSON form of a finished generation that Chat returns and
// StreamSink.OnComplete receives. Its first three fields are the historical
// OnComplete payload; the rest were added later and are safe to ignore.
func resultJSON(r gopherllm.GenerationResult) string {
	st := r.Stats
	var tokensPerSecond float64
	if st.DecodeTime > 0 {
		tokensPerSecond = float64(st.GeneratedTokens) / st.DecodeTime.Seconds()
	}
	return marshal(struct {
		Text            string                       `json:"text"`
		FinishReason    string                       `json:"finish_reason"`
		GeneratedTokens int                          `json:"generated_tokens"`
		ReasoningText   string                       `json:"reasoning_text,omitempty"`
		PromptTokens    int                          `json:"prompt_tokens"`
		TTFTMS          float64                      `json:"ttft_ms"`
		PrefillMS       float64                      `json:"prefill_ms"`
		DecodeMS        float64                      `json:"decode_ms"`
		TotalMS         float64                      `json:"total_ms"`
		TokensPerSecond float64                      `json:"tokens_per_second"`
		ContextWindow   *gopherllm.ContextWindowInfo `json:"context_window,omitempty"`
	}{r.Text, r.FinishReason, st.GeneratedTokens, r.ReasoningText, st.PromptTokens,
		ms(st.TTFT), ms(st.PrefillTime), ms(st.DecodeTime), ms(st.TotalTime), tokensPerSecond, r.ContextWindow})
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// marshal encodes values that are always representable as JSON.
func marshal(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}
