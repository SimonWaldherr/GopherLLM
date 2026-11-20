package engine

import (
	"os"
	"slices"
	"testing"
)

// TestLittleBitPublishedCheckpoint is opt-in because the GGUF is about 3 GB.
// Run each backend in a separate process to capture its peak RSS with time.
func TestLittleBitPublishedCheckpoint(t *testing.T) {
	path := os.Getenv("GOPHERLLM_LITTLEBIT_TEST_GGUF")
	if path == "" {
		t.Skip("set GOPHERLLM_LITTLEBIT_TEST_GGUF to the converted Qwen3-8B LittleBit-2 checkpoint")
	}
	useMetal := os.Getenv("GOPHERLLM_LITTLEBIT_TEST_METAL") == "1"
	if useMetal && !MetalAvailable() {
		t.Fatal("Metal requested but unavailable")
	}
	r, info, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: useMetal})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	a := AnalyzeGGUF(r.GGUF(), r.Tokenizer())
	if r.Architecture() != "qwen3" || r.config.NLayers != 36 || a.LittleBitVariant != 2 ||
		a.LittleBitOriginalParams != 6945767424 || a.LittleBitBytes != 478312704 {
		t.Fatalf("expected the public Qwen3-8B checkpoint, got %s, %d layers, variant %d, %d projections weights, %d factor bytes",
			r.Architecture(), r.config.NLayers, a.LittleBitVariant, a.LittleBitOriginalParams, a.LittleBitBytes)
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens, opts.SystemPrompt = 16, ""
	opts.Sampler.Temperature, opts.Sampler.TopK, opts.Sampler.RepeatPenalty = 0, 1, 1
	const prompt = "Explain why the sky is blue, then name the physical effect in one short sentence."
	want := []uint32{840, 20772, 3170, 279, 12884, 374, 6303, 11, 1221, 829, 279, 6961, 2456, 304, 825, 2805}
	for run := range 3 {
		r.clearPrefixCache()
		got, err := r.Generate(prompt, opts)
		if err != nil {
			t.Fatal(err)
		}
		s := got.Stats
		if s.GeneratedTokens == 0 || cap(r.workspaceBuf.GeneratedTokens) < s.GeneratedTokens {
			t.Fatalf("missing generated tokens: %+v", got)
		}
		ids := r.workspaceBuf.GeneratedTokens[:s.GeneratedTokens]
		if !slices.Equal(ids, want) {
			t.Fatalf("run %d: generated %v, want %v", run+1, ids, want)
		}
		gpu, scratch := r.LittleBitMetalMemory()
		if useMetal && gpu < uint64(a.LittleBitBytes) {
			t.Fatalf("Metal retains only %d bytes for %d stored factor bytes", gpu, a.LittleBitBytes)
		}
		t.Logf("run=%d prompt=%d generated=%d prefill=%.2f tok/s decode=%.2f tok/s ids=%v text=%q reasoning=%q",
			run+1, s.PromptTokens, s.GeneratedTokens, float64(s.PromptTokens)/s.PrefillTime.Seconds(),
			float64(s.GeneratedTokens)/s.DecodeTime.Seconds(), ids, got.Text, got.ReasoningText)
		t.Logf("GGUF=%d tensor=%d factor=%d BPW=%0.9f regular=%d embedding_raw=%d head_raw=%d head_f32=%d KV=%d gpu_factors=%d gpu_scratch=%d",
			info.FileSizeBytes, a.FileBytes, a.LittleBitBytes, 8*float64(a.LittleBitBytes)/float64(a.LittleBitOriginalParams),
			a.FileBytes-a.LittleBitBytes, len(r.standard.TokenEmbd.Raw), len(r.standard.Output.Raw), len(r.standard.Output.F32)*4,
			kvCacheStorageBytes(r.workspaceCache), gpu, scratch)
	}
}
