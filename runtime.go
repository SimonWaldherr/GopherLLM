package gopherllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

var ErrGenerationCanceled = errors.New("generation canceled")

type EmbeddingResult struct {
	Embedding  []float32
	TokenCount int
}

type ChatRole int

const (
	ChatRoleSystem ChatRole = iota
	ChatRoleUser
	ChatRoleAssistant
	// ChatRoleTool carries the result of a previously requested tool call back
	// to the model. ToolCallID must match the id the assistant's ToolCalls
	// entry used.
	ChatRoleTool
)

type ChatMessage struct {
	Role    ChatRole
	Content string
	// ToolCalls is set on an assistant message that is replaying a prior turn
	// in which the model requested one or more tool calls.
	ToolCalls []ToolCall
	// ToolCallID and Name identify which prior tool call a ChatRoleTool
	// message is answering.
	ToolCallID string
	Name       string
}

func UserMessage(content string) ChatMessage {
	return ChatMessage{Role: ChatRoleUser, Content: content}
}
func AssistantMessage(content string) ChatMessage {
	return ChatMessage{Role: ChatRoleAssistant, Content: content}
}

// ToolResultMessage renders the output of tool call callID (named name) back
// into the conversation for the model to see on its next turn.
func ToolResultMessage(callID, name, content string) ChatMessage {
	return ChatMessage{Role: ChatRoleTool, Content: content, ToolCallID: callID, Name: name}
}

type GenerationOptions struct {
	MaxTokens     int
	Sampler       SamplerConfig
	Seed          uint64
	SystemPrompt  string
	StopSequences []string
	// Tools lists the functions the model may call. When non-empty, it is
	// rendered into the prompt using the active chat template's tool-calling
	// convention (native for Mistral, a generic <tool_call> JSON convention
	// otherwise).
	Tools []ToolDefinition
	// ToolChoice controls whether tools are offered at all. "none" suppresses
	// tool rendering for this request; any other value (including the default
	// "auto") offers all of Tools.
	ToolChoice string
	// ctx, when set (by the Model API's context-first methods), cancels
	// generation between prefill chunks and between decoded tokens. Stored on
	// the options value rather than passed positionally so the many existing
	// Generate* entry points keep their signatures; the request-options
	// pattern (like http.Request) is the accepted exception to "don't store a
	// Context in a struct".
	ctx context.Context
}

// generationContext returns the request context, defaulting to Background.
func (o GenerationOptions) generationContext() context.Context {
	if o.ctx != nil {
		return o.ctx
	}
	return context.Background()
}

func DefaultGenerationOptions() GenerationOptions {
	return GenerationOptions{MaxTokens: 256, Sampler: DefaultSamplerConfig(), SystemPrompt: "You are a helpful assistant."}
}

// activeTools returns the tools that should actually be offered to the model
// for this request, honoring ToolChoice: "none".
func (o GenerationOptions) activeTools() []ToolDefinition {
	if o.ToolChoice == "none" {
		return nil
	}
	return o.Tools
}

func (o GenerationOptions) Validate() error {
	if o.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be greater than 0")
	}
	if !finite32(o.Sampler.Temperature) || o.Sampler.Temperature < 0 {
		return fmt.Errorf("temperature must be a finite number >= 0")
	}
	if !finite32(o.Sampler.TopP) || o.Sampler.TopP <= 0 || o.Sampler.TopP > 1 {
		return fmt.Errorf("top_p must be in the range (0, 1]")
	}
	if o.Sampler.TopK < 0 {
		return fmt.Errorf("top_k must be greater than or equal to 0")
	}
	if !finite32(o.Sampler.MinP) || o.Sampler.MinP < 0 || o.Sampler.MinP > 1 {
		return fmt.Errorf("min_p must be in the range [0, 1]")
	}
	if !finite32(o.Sampler.RepeatPenalty) || o.Sampler.RepeatPenalty <= 0 {
		return fmt.Errorf("repeat_penalty must be a finite number > 0")
	}
	return nil
}

func finite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

type GenerationStats struct {
	PromptTokens    int
	GeneratedTokens int
	TTFT            time.Duration
	PrefillTime     time.Duration
	DecodeTime      time.Duration
	TotalTime       time.Duration
}

type GenerationResult struct {
	Text string
	// ReasoningText holds any chain-of-thought the model emitted separately
	// from its answer (e.g. DeepSeek-R1/QwQ <think> blocks, or gpt-oss's
	// analysis channel), stripped out of Text.
	ReasoningText string
	// ToolCalls holds structured function calls extracted from the model's
	// raw output, stripped out of Text. Empty unless GenerationOptions.Tools
	// was non-empty for this request.
	ToolCalls []ToolCall
	// FinishReason is "stop" (natural end or stop-sequence match), "length"
	// (max_tokens or context exhausted), or "tool_calls" (ToolCalls is
	// non-empty).
	FinishReason string
	Stats        GenerationStats
}

type LoadInfo struct {
	FileSizeBytes int
	LoadTime      time.Duration
}

type LoadOptions struct {
	PrepareQuantized bool
	UseMetal         bool
	LogWriter        io.Writer
}

type loadedKind int

const (
	loadedStandard loadedKind = iota
	loadedGptOss
	loadedGemma4
)

// Runner is a fully loaded model ready to generate: parsed GGUF header,
// tokenizer, config, and weights (one of the three kind-specific sets).
// Generations and embeddings are serialized by genLock — a Runner is safe to
// share across goroutines (the HTTP server does), but runs one request at a
// time. Close releases the memory-mapped weight file; quantized weights
// borrow from it, so no method may be called after Close.
type Runner struct {
	gguf       *GGUFFile
	arch       string
	tok        *Tokenizer
	config     Config
	kind       loadedKind
	standard   ModelWeights
	gptOss     GptOssWeights
	gemma4     Gemma4Weights
	genLock    sync.Mutex
	mappedFile *MmapFile
}

// ArchitectureSupported reports whether the loader accepts this
// general.architecture value. Notes on specific families:
//
//   - qwen3 (incl. the DeepSeek-R1-0528 Qwen3 distills): the qwen2 graph plus
//     per-head QK-norm, which loads via the optional attn_q_norm/attn_k_norm
//     tensors and applies exactly as for Gemma 3/4.
//   - deepseek2 (MLA attention) is NOT supported; DeepSeek-R1 distills ship
//     as qwen2/qwen3/llama and work through those graphs.
//   - Devstral and Mistral-Small GGUFs usually declare llama or mistral3;
//     their [INST]/Tekken behavior is picked up from tokenizer metadata, not
//     the arch string.
func ArchitectureSupported(arch string) bool {
	switch arch {
	case "llama", "llama2", "llama3", "mistral", "mistral3", "ministral", "mixtral",
		"qwen2", "qwen3", "gpt-oss", "gemma", "gemma2", "gemma4":
		return true
	default:
		return false
	}
}

// RunnerFromGGUFBytes loads a model from an in-memory GGUF, copying quantized
// tensors into owned memory. It is silent; use Open with WithLogWriter for
// load-progress diagnostics.
func RunnerFromGGUFBytes(data []byte) (*Runner, error) {
	return RunnerFromGGUFBytesWithOptions(data, LoadOptions{})
}

func RunnerFromGGUFBytesWithOptions(data []byte, options LoadOptions) (*Runner, error) {
	return runnerFromGGUFBytes(data, false, options)
}

func runnerFromGGUFBytes(data []byte, borrowQuantized bool, options LoadOptions) (*Runner, error) {
	logw := options.LogWriter
	if logw == nil {
		logw = io.Discard
	}
	gguf, err := ParseGGUF(data)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(logw, "GGUF v%d - %d tensors, %d metadata entries\n", gguf.Version, len(gguf.Tensors), len(gguf.Metadata))
	arch, ok := gguf.GetString("general.architecture")
	if !ok || arch == "" {
		arch = "llama"
	}
	if !ArchitectureSupported(arch) {
		return nil, fmt.Errorf("unsupported architecture: %s", arch)
	}
	tok, err := TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		return nil, err
	}
	r := &Runner{gguf: gguf, arch: arch, tok: tok}
	switch arch {
	case "gpt-oss":
		config, weights, err := LoadGptOssModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw)
		if err != nil {
			return nil, err
		}
		r.config, r.gptOss, r.kind = config, weights, loadedGptOss
	case "gemma", "gemma2", "gemma4":
		// The dense Gemma graph is implemented (sqrt(dim) embedding scaling,
		// GELU FFN, QK-norm, post-attention/post-FFN norms, attention/final
		// logit softcapping, per-layer sliding-window map, <start_of_turn>
		// template) but has not yet been validated against real Gemma
		// weights, and the Gemma 4-specific mechanisms (p-RoPE frequency
		// factors, per-layer RoPE bases, cross-layer KV sharing, per-layer
		// embeddings, the 26B MoE) are still missing. See
		// docs/INFERENCE_NOTES.md.
		fmt.Fprintf(logw, "Warning: %s support is experimental (dense graph implemented, unvalidated against real weights; Gemma 4 p-RoPE/PLE/MoE missing — see docs/INFERENCE_NOTES.md)\n", arch)
		config, weights, err := LoadGemma4Model(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw)
		if err != nil {
			return nil, err
		}
		r.config, r.gemma4, r.kind = config, weights, loadedGemma4
	default:
		config, weights, err := LoadModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw)
		if err != nil {
			return nil, err
		}
		r.config, r.standard, r.kind = config, weights, loadedStandard
	}
	return r, nil
}

// RunnerFromPath memory-maps a GGUF file and loads it with zero-copy borrowed
// quantized weights. Silent; prefer Open, which adds context support,
// configurable logging, and a higher-level Model wrapper.
func RunnerFromPath(path string) (*Runner, LoadInfo, error) {
	return RunnerFromPathWithOptions(path, LoadOptions{})
}

func RunnerFromPathWithOptions(path string, options LoadOptions) (*Runner, LoadInfo, error) {
	t0 := time.Now()
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, LoadInfo{}, fmt.Errorf("failed to open model: %w", err)
	}
	r, err := runnerFromGGUFBytes(mmap.Bytes(), true, options)
	if err != nil {
		_ = mmap.Close()
		return nil, LoadInfo{}, err
	}
	r.mappedFile = mmap
	return r, LoadInfo{FileSizeBytes: mmap.Len(), LoadTime: time.Since(t0)}, nil
}

func (r *Runner) Architecture() string      { return r.arch }
func (r *Runner) Tokenizer() *Tokenizer     { return r.tok }
func (r *Runner) GGUF() *GGUFFile           { return r.gguf }
func (r *Runner) Config() Config            { return r.config }
func (r *Runner) ModelName() (string, bool) { return r.gguf.GetString("general.name") }

func (r *Runner) Close() error {
	if r == nil {
		return nil
	}
	r.releaseMetalWeights()
	if r.mappedFile == nil {
		return nil
	}
	err := r.mappedFile.Close()
	r.mappedFile = nil
	return err
}

func (r *Runner) releaseMetalWeights() {
	if r == nil {
		return
	}
	switch r.kind {
	case loadedGptOss:
		releaseModelMetalWeights(&r.gptOss.Standard)
	case loadedGemma4:
		releaseModelMetalWeights(&r.gemma4.Standard)
	default:
		releaseModelMetalWeights(&r.standard)
	}
}

func (r *Runner) Generate(prompt string, options GenerationOptions) (GenerationResult, error) {
	return r.GenerateChat([]ChatMessage{UserMessage(prompt)}, options)
}

func (r *Runner) GenerateChat(messages []ChatMessage, options GenerationOptions) (GenerationResult, error) {
	return r.GenerateChatStream(messages, options, func(string) {})
}

func (r *Runner) GenerateStream(prompt string, options GenerationOptions, onToken func(string)) (GenerationResult, error) {
	return r.GenerateChatStream([]ChatMessage{UserMessage(prompt)}, options, onToken)
}

func (r *Runner) GenerateChatStream(messages []ChatMessage, options GenerationOptions, onToken func(string)) (GenerationResult, error) {
	return r.GenerateChatStreamUntil(messages, options, func(text string) bool {
		onToken(text)
		return true
	})
}

// GenerateChatStreamUntil is the generation entry point everything else wraps
// (Generate, GenerateChat, GenerateStream, ... are thin adapters over it):
// render messages through the model's chat template, prefill the prompt
// (batched when the architecture allows), then decode token by token until
// EOS, a stop sequence, max_tokens, or the context limit. onToken receives
// valid-UTF-8 text increments (bytes are buffered across token boundaries
// until they complete a rune, and the tail is held back while it could still
// be a stop-sequence prefix); returning false cancels generation, yielding
// the partial result with ErrGenerationCanceled. The final result carries
// content with reasoning and tool calls already extracted (classifyOutput)
// and a FinishReason of "stop", "length", or "tool_calls".
func (r *Runner) GenerateChatStreamUntil(messages []ChatMessage, options GenerationOptions, onToken func(string) bool) (GenerationResult, error) {
	r.genLock.Lock()
	defer r.genLock.Unlock()
	if err := options.Validate(); err != nil {
		return GenerationResult{}, err
	}
	if len(messages) == 0 {
		return GenerationResult{}, fmt.Errorf("no prompt provided")
	}
	totalStart := time.Now()
	tokens := r.renderMessages(messages, options.SystemPrompt, options.activeTools())
	if len(tokens) == 0 {
		return GenerationResult{}, fmt.Errorf("prompt rendered to zero tokens")
	}
	// The KV cache is sized to r.config.MaxSeqLen; without this check a prompt
	// at or beyond that length (easily reached once a request injects a large
	// tool listing) would silently overflow it deeper in the forward pass
	// instead of failing here with a clear error.
	if r.config.MaxSeqLen > 0 && len(tokens) >= r.config.MaxSeqLen {
		return GenerationResult{}, fmt.Errorf("prompt (%d tokens) leaves no room to generate within the model's context length (%d)", len(tokens), r.config.MaxSeqLen)
	}
	cacheLen := min(r.config.MaxSeqLen, len(tokens)+options.MaxTokens+1)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, cacheLen)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	seed := options.Seed
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	rng := NewRng(seed)
	ctx := options.generationContext()
	if err := ctx.Err(); err != nil {
		return GenerationResult{}, err
	}
	prefillStart := time.Now()
	logits := []float32{}
	if r.canBatchPrefill() {
		if err := r.prefillBatched(ctx, cache, buf, tokens, &logits); err != nil {
			return GenerationResult{}, err
		}
	} else {
		for pos, tok := range tokens {
			if err := ctx.Err(); err != nil {
				return GenerationResult{}, err
			}
			if pos == len(tokens)-1 {
				r.forwardTokenInto(cache, buf, tok, pos, &logits)
			} else {
				r.forwardPrefillToken(cache, buf, tok, pos)
			}
		}
	}
	prefillTime := time.Since(prefillStart)
	decodeStart := time.Now()
	var ttft time.Duration
	output := strings.Builder{}
	maxStopLen := 0
	for _, stop := range options.StopSequences {
		maxStopLen = max(maxStopLen, len(stop))
	}
	streamBuf := []byte{}
	flushStream := func(final bool) bool {
		if len(streamBuf) == 0 {
			return true
		}
		n := validUTF8PrefixLen(streamBuf)
		if n == 0 {
			if final {
				streamBuf = streamBuf[:0]
			}
			return true
		}
		if !final && maxStopLen > 1 {
			hold := maxStopLen - 1
			if n <= hold {
				return true
			}
			n -= hold
		}
		text := string(streamBuf[:n])
		if !onToken(text) {
			return false
		}
		copy(streamBuf, streamBuf[n:])
		streamBuf = streamBuf[:len(streamBuf)-n]
		if final && !utf8.Valid(streamBuf) {
			streamBuf = streamBuf[:0]
		}
		return true
	}
	generated := []uint32{}
	recent := append([]uint32(nil), tokens...)
	pos := len(tokens)
	finishReason := "length"
	greedyFastPath := r.canGreedyOutputFastPath(options)
	haveNextToken := false
	var nextToken uint32
	buildResult := func() GenerationResult {
		stats := GenerationStats{PromptTokens: len(tokens), GeneratedTokens: len(generated), TTFT: ttft, PrefillTime: prefillTime, DecodeTime: time.Since(decodeStart), TotalTime: time.Since(totalStart)}
		content, reasoning, calls := r.classifyOutput(output.String(), options.activeTools(), rng)
		reason := finishReason
		if len(calls) > 0 {
			reason = "tool_calls"
		}
		return GenerationResult{Text: content, ReasoningText: reasoning, ToolCalls: calls, FinishReason: reason, Stats: stats}
	}
decode:
	for range options.MaxTokens {
		if err := ctx.Err(); err != nil {
			return buildResult(), err
		}
		token := nextToken
		if haveNextToken {
			haveNextToken = false
		} else {
			token = SampleWithScratch(logits, options.Sampler, rng, recent, &buf.SamplerCandidates)
		}
		if r.isStopToken(token) {
			finishReason = "stop"
			break
		}
		if ttft == 0 {
			ttft = time.Since(totalStart)
		}
		text := r.tok.DecodeToken(token)
		output.WriteString(text)
		current := output.String()
		if maxStopLen > 0 {
			windowStart := max(0, len(current)-maxStopLen-len(text))
			window := current[windowStart:]
			for _, stop := range options.StopSequences {
				if idx := strings.Index(window, stop); idx >= 0 {
					current = current[:windowStart+idx]
					output.Reset()
					output.WriteString(current)
					streamBuf = streamBuf[:0]
					finishReason = "stop"
					break decode
				}
			}
		}
		streamBuf = append(streamBuf, text...)
		if !flushStream(false) {
			return buildResult(), ErrGenerationCanceled
		}
		generated = append(generated, token)
		recent = append(recent, token)
		if len(recent) > 64 {
			recent = recent[len(recent)-64:]
		}
		if len(generated) >= options.MaxTokens || pos >= cacheLen {
			break
		}
		if greedyFastPath {
			var ok bool
			nextToken, ok = r.forwardGreedyToken(cache, buf, token, pos, &logits)
			haveNextToken = ok
		} else {
			r.forwardTokenInto(cache, buf, token, pos, &logits)
		}
		pos++
	}
	if !flushStream(true) {
		return buildResult(), ErrGenerationCanceled
	}
	return buildResult(), nil
}

func validUTF8PrefixLen(b []byte) int {
	if utf8.Valid(b) {
		return len(b)
	}
	for n := len(b) - 1; n >= 0 && len(b)-n <= utf8.UTFMax; n-- {
		if utf8.Valid(b[:n]) {
			return n
		}
	}
	return 0
}

func (r *Runner) cacheDims() (int, int, int, int, int) {
	if r.kind == loadedGemma4 {
		maxHD, maxKV, maxVal := r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
		for _, l := range r.gemma4.Layers {
			maxHD = max(maxHD, l.HeadDim)
			maxKV = max(maxKV, l.NKVHeads)
			maxVal = max(maxVal, l.ValueDim)
		}
		return maxKV * maxHD, maxKV * maxVal, maxHD, maxKV, maxVal
	}
	return r.config.NKVHeads * r.config.HeadDim, r.config.KVDim, r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
}

func (r *Runner) forwardTokenInto(cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	switch r.kind {
	case loadedGptOss:
		ForwardGptOssInto(r.config, r.gptOss, cache, buf, token, pos, logits)
	case loadedGemma4:
		ForwardGemma4Into(r.config, r.gemma4, cache, buf, token, pos, logits)
	default:
		ForwardInto(r.config, r.standard, cache, buf, token, pos, logits)
	}
}

func (r *Runner) canGreedyOutputFastPath(options GenerationOptions) bool {
	s := options.Sampler
	return os.Getenv("GOPHERLLM_NO_GREEDY_ARGMAX") != "1" &&
		s.RepeatPenalty == 1 &&
		(s.Temperature < 1e-6 || s.TopK == 1)
}

func (r *Runner) forwardGreedyToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) (uint32, bool) {
	switch r.kind {
	case loadedGptOss:
		ForwardBodyInto(r.config, r.gptOss.Standard, cache, buf, token, pos)
		if next, ok := ArgmaxOutputToken(r.config, r.gptOss.Standard, buf); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.gptOss.Standard, buf, logits)
	case loadedGemma4:
		ForwardBodyInto(r.config, r.gemma4.Standard, cache, buf, token, pos)
		if next, ok := ArgmaxOutputToken(r.config, r.gemma4.Standard, buf); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.gemma4.Standard, buf, logits)
	default:
		ForwardBodyInto(r.config, r.standard, cache, buf, token, pos)
		if next, ok := ArgmaxOutputToken(r.config, r.standard, buf); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.standard, buf, logits)
	}
	return 0, false
}

func (r *Runner) forwardHiddenToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	switch r.kind {
	case loadedGptOss:
		return ForwardHiddenGptOss(r.config, r.gptOss, cache, buf, token, pos)
	case loadedGemma4:
		return ForwardHiddenGemma4(r.config, r.gemma4, cache, buf, token, pos)
	default:
		return ForwardHidden(r.config, r.standard, cache, buf, token, pos)
	}
}

// canBatchPrefill reports whether the model uses the standard non-fused
// transformer path that ForwardBatchInto supports.
func (r *Runner) canBatchPrefill() bool {
	if r.kind != loadedStandard || len(r.standard.Layers) == 0 {
		return false
	}
	if os.Getenv("GOPHERLLM_NO_BATCH_PREFILL") != "" {
		return false
	}
	// The batched graph implements the plain llama-style block, including
	// fused QKV and fused gate/up tensors. Gemma-family mechanics (GELU,
	// QK-norm, post-norms) fall back to the per-token path, which supports
	// everything.
	if r.config.UseGELU {
		return false
	}
	for _, l := range r.standard.Layers {
		if l.AttnQNorm != nil || l.AttnKNorm != nil || l.PostAttnNorm != nil || l.PostFFNNorm != nil {
			return false
		}
	}
	return true
}

// prefillBatched processes the prompt in chunks, streaming each weight once per
// chunk instead of once per token.
//
// chunk was swept empirically (BenchmarkChunkThroughputP*, not checked in):
// per-token throughput improves from P=8 up to a peak around P=32, then
// degrades again through P=48/64/128/256 as the chunk's activation buffers
// (X, XN, Q, K, V, ...; each sized dim*chunk floats) outgrow cache and start
// costing more in misses per dequantized weight row than the extra dequant
// amortization buys back. 32 was consistently ~15-20% faster than the
// previous default of 64 across repeated runs.
func (r *Runner) prefillBatched(ctx context.Context, cache *KVCache, buf *DecodeBuffer, tokens []uint32, logits *[]float32) error {
	chunk := prefillChunkSize(r.config)
	n := len(tokens)
	for start := 0; start < n; start += chunk {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+chunk, n)
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens[start:end], start, end == n, logits)
	}
	return nil
}

// prefillChunkSize keeps a conservative default for larger models while using
// larger chunks on small dense models such as Ministral-3-3B. Bottleneck:
// prompt prefill is chunk-size sensitive because larger chunks amortize
// dequantization but grow activation working sets. Change: small models default
// to 128 after real Ministral measurement; GOPHERLLM_PREFILL_CHUNK can override
// it for A/B testing. Expected effect: lower TTFT on 3B-class models. Risk:
// too-large chunks can regress cache locality on bigger models. Rollback: set
// GOPHERLLM_PREFILL_CHUNK=32.
func prefillChunkSize(config Config) int {
	const def = 32
	raw := strings.TrimSpace(os.Getenv("GOPHERLLM_PREFILL_CHUNK"))
	if raw == "" {
		if config.Dim > 0 && config.Dim <= 3072 && config.HiddenDim <= 12288 {
			return 128
		}
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return def
	}
	if n > 256 {
		return 256
	}
	return n
}

func (r *Runner) forwardPrefillToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	switch r.kind {
	case loadedGptOss:
		ForwardPrefill(r.config, r.gptOss.Standard, cache, buf, token, pos)
	case loadedGemma4:
		ForwardPrefill(r.config, r.gemma4.Standard, cache, buf, token, pos)
	default:
		ForwardPrefill(r.config, r.standard, cache, buf, token, pos)
	}
}

// Embed produces a text embedding by mean-pooling the final-layer hidden
// states over all input tokens and L2-normalizing the result (so dot product
// equals cosine similarity). Dimension is the model's hidden size — note this
// uses the generation model's hidden states, not a dedicated embedding head.
func (r *Runner) Embed(text string) (EmbeddingResult, error) {
	r.genLock.Lock()
	defer r.genLock.Unlock()
	tokens := r.tok.Encode(text)
	if len(tokens) == 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: input tokenised to zero tokens")
	}
	if r.config.MaxSeqLen > 0 && len(tokens) > r.config.MaxSeqLen {
		return EmbeddingResult{}, fmt.Errorf("embed: input (%d tokens) exceeds the model's context length (%d)", len(tokens), r.config.MaxSeqLen)
	}
	cacheLen := min(r.config.MaxSeqLen, len(tokens)+1)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, cacheLen)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	sum := make([]float32, r.config.Dim)
	for pos, tok := range tokens {
		h := r.forwardHiddenToken(cache, buf, tok, pos)
		for i, v := range h {
			sum[i] += v
		}
	}
	meanPoolInPlace(sum, len(tokens))
	l2NormalizeInPlace(sum)
	return EmbeddingResult{Embedding: sum, TokenCount: len(tokens)}, nil
}

func (r *Runner) isStopToken(token uint32) bool {
	if r.arch == "gpt-oss" {
		return token == r.tok.EOSID || token == 200002 || token == 200007
	}
	if gemmaFamily(r.arch) {
		// Gemma instruct models end assistant turns with <end_of_turn>, not
		// the <eos> the GGUF declares as EOS.
		if id, ok := r.tok.SpecialID("<end_of_turn>"); ok && token == id {
			return true
		}
	}
	return token == r.tok.EOSID
}

// renderMessages renders the conversation (and, if any, the tool listing) into
// tokens using the active chat template. Mistral gets its native
// [AVAILABLE_TOOLS]/[TOOL_CALLS]/[TOOL_RESULTS] convention; every other
// template (and gpt-oss, for which tool calling is not yet implemented) uses
// the generic <tool_call> JSON convention, applied uniformly by flattening
// tool listings and tool-call history into ordinary system/user/assistant
// text before delegating to the per-family renderer below.
func (r *Runner) renderMessages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) []uint32 {
	if r.arch == "gpt-oss" {
		return r.renderGptOssMessages(messages, systemPrompt)
	}
	if r.chatTemplateKind() == "mistral-inst" {
		if tokens, ok := r.renderMistralInstMessages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	generic, genericSystem := injectGenericTools(messages, systemPrompt, tools)
	switch r.chatTemplateKind() {
	case "gemma-chat":
		if tokens, ok := r.renderGemmaMessages(generic, genericSystem); ok {
			return tokens
		}
	case "header-chat":
		if tokens, ok := r.renderHeaderChatMessages(generic, genericSystem); ok {
			return tokens
		}
	case "chatml":
		if tokens, ok := r.renderChatMLMessages(generic, genericSystem); ok {
			return tokens
		}
	case "phi-chat":
		if tokens, ok := r.renderPhiMessages(generic, genericSystem); ok {
			return tokens
		}
	case "deepseek-r1-qwen":
		if tokens, ok := r.renderDeepSeekR1QwenMessages(generic, genericSystem); ok {
			return tokens
		}
	case "granite-chat":
		if tokens, ok := r.renderGraniteMessages(generic, genericSystem); ok {
			return tokens
		}
	}
	return r.renderPlainMessages(generic, genericSystem)
}

// injectGenericTools flattens tool listings and tool-call/tool-result history
// into ordinary text so any chat-template renderer that only understands
// system/user/assistant turns can carry tool use anyway. A tool listing is
// merged into an existing explicit system message's content when present,
// otherwise appended to systemPrompt so the caller's default system prompt
// (e.g. "You are a helpful assistant.") is preserved rather than replaced.
// When there is no tool activity at all, messages/systemPrompt are returned
// unchanged (no allocation) so the common no-tools path is a no-op.
func injectGenericTools(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]ChatMessage, string) {
	hasActivity := len(tools) > 0
	if !hasActivity {
		for _, m := range messages {
			if m.Role == ChatRoleTool || (m.Role == ChatRoleAssistant && len(m.ToolCalls) > 0) {
				hasActivity = true
				break
			}
		}
	}
	if !hasActivity {
		return messages, systemPrompt
	}

	hasExplicitSystem := len(messages) > 0 && messages[0].Role == ChatRoleSystem
	out := make([]ChatMessage, len(messages))
	for i, m := range messages {
		switch {
		case i == 0 && hasExplicitSystem && len(tools) > 0:
			m.Content = appendSection(m.Content, genericToolListText(tools))
		case m.Role == ChatRoleAssistant && len(m.ToolCalls) > 0:
			m.Content = renderGenericAssistantToolCalls(m.Content, m.ToolCalls)
		case m.Role == ChatRoleTool:
			m.Role = ChatRoleUser
			m.Content = renderGenericToolResult(m.Name, m.Content)
		}
		out[i] = m
	}
	if !hasExplicitSystem && len(tools) > 0 {
		systemPrompt = appendSection(systemPrompt, genericToolListText(tools))
	}
	return out, systemPrompt
}

func appendSection(base, section string) string {
	base = strings.TrimRight(base, "\n")
	if base == "" {
		return section
	}
	return base + "\n\n" + section
}

// genericToolListText renders an OpenAI-shaped tool list into the Hermes/Qwen
// style calling convention: a <tool_call> JSON block per invocation.
func genericToolListText(tools []ToolDefinition) string {
	b, err := json.Marshal(tools)
	if err != nil {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("You have access to the following tools. To call one, respond with only a block of exactly this form (multiple blocks if you need multiple calls in the same turn):\n")
	sb.WriteString("<tool_call>\n{\"name\": \"<tool name>\", \"arguments\": <arguments object>}\n</tool_call>\n\n")
	sb.WriteString("Available tools:\n")
	sb.Write(b)
	return sb.String()
}

func renderGenericAssistantToolCalls(content string, calls []ToolCall) string {
	var sb strings.Builder
	if trimmed := strings.TrimSpace(content); trimmed != "" {
		sb.WriteString(trimmed)
		sb.WriteString("\n")
	}
	for i, call := range calls {
		if i > 0 {
			sb.WriteString("\n")
		}
		args := call.Function.Arguments
		if strings.TrimSpace(args) == "" {
			args = "{}"
		}
		nameJSON, _ := json.Marshal(call.Function.Name)
		fmt.Fprintf(&sb, "<tool_call>\n{\"name\": %s, \"arguments\": %s}\n</tool_call>", nameJSON, args)
	}
	return sb.String()
}

func renderGenericToolResult(name, content string) string {
	if name != "" {
		return fmt.Sprintf("<tool_response name=%q>\n%s\n</tool_response>", name, content)
	}
	return fmt.Sprintf("<tool_response>\n%s\n</tool_response>", content)
}

func (r *Runner) renderPlainMessages(messages []ChatMessage, systemPrompt string) []uint32 {
	var b strings.Builder
	if strings.TrimSpace(systemPrompt) != "" {
		b.WriteString("System: ")
		b.WriteString(strings.TrimSpace(systemPrompt))
		b.WriteString("\n\n")
	}
	for _, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			b.WriteString("System: ")
		case ChatRoleAssistant:
			b.WriteString("Assistant: ")
		default:
			b.WriteString("User: ")
		}
		b.WriteString(strings.TrimSpace(m.Content))
		b.WriteString("\n\n")
	}
	b.WriteString("Assistant:")
	return r.tok.Encode(b.String())
}

func (r *Runner) renderGptOssMessages(messages []ChatMessage, systemPrompt string) []uint32 {
	start := specialOr(r.tok, "<|start|>", 200006)
	channel := specialOr(r.tok, "<|channel|>", 200005)
	message := specialOr(r.tok, "<|message|>", 200008)
	end := specialOr(r.tok, "<|end|>", 200007)
	user := specialOrEncoded(r.tok, "user")
	assistant := specialOrEncoded(r.tok, "assistant")
	system := specialOrEncoded(r.tok, "system")
	finalTok := specialOrEncoded(r.tok, "final")
	tokens := []uint32{}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		tokens = append(tokens, start, system, message)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(systemPrompt)...)
		tokens = append(tokens, end)
	}
	for _, m := range messages {
		role := user
		if m.Role == ChatRoleSystem {
			role = system
		} else if m.Role == ChatRoleAssistant {
			role = assistant
		}
		tokens = append(tokens, start, role, message)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
		tokens = append(tokens, end)
	}
	tokens = append(tokens, start, assistant, channel, finalTok, message)
	return tokens
}

// renderMistralInstMessages renders the Mistral/Ministral instruct format:
//
//	<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][AVAILABLE_TOOLS]{tools}[/AVAILABLE_TOOLS][INST]{user}[/INST]{assistant}</s>...
//
// [INST]/[/INST] (and, on newer Tekken vocabularies, [SYSTEM_PROMPT]/
// [/SYSTEM_PROMPT] and the tool-calling markers) are emitted as control
// tokens. When the vocabulary lacks the dedicated system-prompt tokens we
// fall back to the older Mistral 2410 behavior of folding the system prompt
// into the final user turn. Format verified directly against the
// tokenizer.chat_template of a real Ministral-3-3B-Instruct-2512 GGUF.
func (r *Runner) renderMistralInstMessages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]uint32, bool) {
	instTok, ok1 := r.tok.SpecialID("[INST]")
	instEndTok, ok2 := r.tok.SpecialID("[/INST]")
	if !(ok1 && ok2) {
		return nil, false
	}
	sysStart, sysEnd, hasSysTokens := r.systemPromptTokens()
	callTok := r.mistralMarker("[TOOL_CALLS]")
	argsTok := r.mistralMarker("[ARGS]")
	resultsStart := r.mistralMarker("[TOOL_RESULTS]")
	resultsEnd := r.mistralMarker("[/TOOL_RESULTS]")

	system := strings.TrimSpace(systemPrompt)
	loop := make([]ChatMessage, 0, len(messages))
	for _, m := range messages {
		if m.Role == ChatRoleSystem {
			if s := strings.TrimSpace(m.Content); s != "" {
				system = s
			}
			continue
		}
		loop = append(loop, m)
	}
	lastUser := -1
	for i, m := range loop {
		if m.Role == ChatRoleUser {
			lastUser = i
		}
	}

	tokens := []uint32{}
	if r.tok.AddBOS {
		tokens = append(tokens, r.tok.BOSID)
	}
	if system != "" && hasSysTokens {
		tokens = append(tokens, sysStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(system)...)
		tokens = append(tokens, sysEnd)
	}
	if len(tools) > 0 {
		if toolsJSON, err := json.Marshal(tools); err == nil {
			tokens = append(tokens, r.mistralMarker("[AVAILABLE_TOOLS]")...)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(string(toolsJSON))...)
			tokens = append(tokens, r.mistralMarker("[/AVAILABLE_TOOLS]")...)
		}
	}
	for i, m := range loop {
		switch m.Role {
		case ChatRoleAssistant:
			if content := strings.TrimSpace(m.Content); content != "" {
				tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
			}
			for _, call := range m.ToolCalls {
				args := call.Function.Arguments
				if args == "" {
					args = "{}"
				}
				tokens = append(tokens, callTok...)
				tokens = append(tokens, r.tok.EncodeWithoutBOS(call.Function.Name)...)
				tokens = append(tokens, argsTok...)
				tokens = append(tokens, r.tok.EncodeWithoutBOS(args)...)
			}
			// A trailing assistant message is a prefill continuation: leave
			// the turn open (no EOS) so generation extends it — the standard
			// way to seed a reply prefix with Mistral/Devstral models.
			if i == len(loop)-1 && len(m.ToolCalls) == 0 {
				break
			}
			tokens = append(tokens, r.tok.EOSID)
		case ChatRoleTool:
			tokens = append(tokens, resultsStart...)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
			tokens = append(tokens, resultsEnd...)
		default:
			content := strings.TrimSpace(m.Content)
			if i == lastUser && system != "" && !hasSysTokens {
				content = system + "\n\n" + content
			}
			tokens = append(tokens, instTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
			tokens = append(tokens, instEndTok)
		}
	}
	return tokens, true
}

// mistralMarker returns literal as a single control token when the vocabulary
// defines one (true for every marker on real Mistral/Ministral Tekken
// tokenizers, verified directly against a Ministral-3-3B-Instruct-2512 GGUF),
// falling back to plain-text encoding so rendering degrades rather than fails
// outright on a hypothetical vocabulary that lacks it.
func (r *Runner) mistralMarker(literal string) []uint32 {
	if id, ok := r.tok.SpecialID(literal); ok {
		return []uint32{id}
	}
	return r.tok.EncodeWithoutBOS(literal)
}

func (r *Runner) systemPromptTokens() (start, end uint32, ok bool) {
	s, ok1 := r.tok.SpecialID("[SYSTEM_PROMPT]")
	e, ok2 := r.tok.SpecialID("[/SYSTEM_PROMPT]")
	return s, e, ok1 && ok2
}

// renderGemmaMessages renders the Gemma turn format:
//
//	<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{reply}<end_of_turn>\n...
//
// Gemma generations before 4 have no system role; the system prompt is folded
// into the first user turn (the convention Google's reference templates use).
// The assistant role is spelled "model".
func (r *Runner) renderGemmaMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	startTurn, ok1 := r.tok.SpecialID("<start_of_turn>")
	endTurn, ok2 := r.tok.SpecialID("<end_of_turn>")
	if !(ok1 && ok2) {
		return nil, false
	}
	system := strings.TrimSpace(systemPrompt)
	loop := make([]ChatMessage, 0, len(messages))
	for _, m := range messages {
		if m.Role == ChatRoleSystem {
			if s := strings.TrimSpace(m.Content); s != "" {
				system = s
			}
			continue
		}
		loop = append(loop, m)
	}
	tokens := []uint32{}
	if r.tok.AddBOS {
		tokens = append(tokens, r.tok.BOSID)
	}
	firstUser := true
	for _, m := range loop {
		role := "user"
		if m.Role == ChatRoleAssistant {
			role = "model"
		}
		content := strings.TrimSpace(m.Content)
		if m.Role == ChatRoleUser && firstUser && system != "" {
			content = system + "\n\n" + content
			firstUser = false
		} else if m.Role == ChatRoleUser {
			firstUser = false
		}
		tokens = append(tokens, startTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n"+content)...)
		tokens = append(tokens, endTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	tokens = append(tokens, startTurn)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("model\n")...)
	return tokens, true
}

func (r *Runner) renderHeaderChatMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	bot, ok1 := r.tok.SpecialID("<|begin_of_text|>")
	startHeader, ok2 := r.tok.SpecialID("<|start_header_id|>")
	endHeader, ok3 := r.tok.SpecialID("<|end_header_id|>")
	eot, ok4 := r.tok.SpecialID("<|eot_id|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := []uint32{bot}
	pushHeader := func(role string) {
		tokens = append(tokens, startHeader)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, endHeader)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n\n")...)
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		pushHeader("system")
		tokens = append(tokens, r.tok.EncodeWithoutBOS(systemPrompt)...)
		tokens = append(tokens, eot)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		pushHeader(role)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
		tokens = append(tokens, eot)
	}
	pushHeader("assistant")
	return tokens, true
}

func (r *Runner) renderChatMLMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imEnd, ok2 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, imStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n"+strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, imEnd)
			tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
		}
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt, true)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		appendTurn(role, m.Content, true)
	}
	tokens = append(tokens, imStart)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant\n")...)
	return tokens, true
}

func (r *Runner) renderPhiMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	systemTok, ok1 := r.tok.SpecialID("<|system|>")
	userTok, ok2 := r.tok.SpecialID("<|user|>")
	assistantTok, ok3 := r.tok.SpecialID("<|assistant|>")
	endTok, ok4 := r.tok.SpecialID("<|end|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role ChatRole, content string) {
		switch role {
		case ChatRoleSystem:
			tokens = append(tokens, systemTok)
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
		default:
			tokens = append(tokens, userTok)
		}
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n"+strings.TrimSpace(content))...)
		tokens = append(tokens, endTok)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn(ChatRoleSystem, systemPrompt)
	}
	for _, m := range messages {
		appendTurn(m.Role, m.Content)
	}
	tokens = append(tokens, assistantTok)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	return tokens, true
}

func (r *Runner) renderDeepSeekR1QwenMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	userTok, ok1 := r.tok.SpecialID("<｜User｜>")
	assistantTok, ok2 := r.tok.SpecialID("<｜Assistant｜>")
	endTok, ok3 := r.tok.SpecialID("<｜end▁of▁sentence｜>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{r.tok.BOSID}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(systemPrompt))...)
	}
	for _, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
			tokens = append(tokens, endTok)
		default:
			tokens = append(tokens, userTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
		}
	}
	tokens = append(tokens, assistantTok)
	return tokens, true
}

func (r *Runner) renderGraniteMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	startRole, ok1 := r.tok.SpecialID("<|start_of_role|>")
	endRole, ok2 := r.tok.SpecialID("<|end_of_role|>")
	endText, ok3 := r.tok.SpecialID("<|end_of_text|>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, startRole)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, endRole)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, endText)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(" ")...)
		}
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt, true)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		appendTurn(role, m.Content, true)
	}
	appendTurn("assistant", "", false)
	return tokens, true
}

func (r *Runner) chatTemplateKind() string {
	if v, ok := r.gguf.Metadata["tokenizer.chat_template"]; ok {
		if s, ok := v.AsString(); ok {
			switch {
			case strings.Contains(s, "[INST]") && strings.Contains(s, "[/INST]"):
				return "mistral-inst"
			case strings.Contains(s, "<|start_header_id|>") && strings.Contains(s, "<|eot_id|>"):
				return "header-chat"
			case strings.Contains(s, "<|im_start|>") && strings.Contains(s, "<|im_end|>"):
				return "chatml"
			case strings.Contains(s, "<|user|>") && strings.Contains(s, "<|assistant|>") && strings.Contains(s, "<|end|>"):
				return "phi-chat"
			case strings.Contains(s, "<｜User｜>") && strings.Contains(s, "<｜Assistant｜>"):
				return "deepseek-r1-qwen"
			case strings.Contains(s, "<|start_of_role|>") && strings.Contains(s, "<|end_of_role|>"):
				return "granite-chat"
			case strings.Contains(s, "<start_of_turn>") && strings.Contains(s, "<end_of_turn>"):
				return "gemma-chat"
			}
		}
	}
	if _, ok := r.tok.SpecialID("[INST]"); ok {
		if _, ok := r.tok.SpecialID("[/INST]"); ok {
			return "mistral-inst"
		}
	}
	if _, ok := r.tok.SpecialID("<|im_start|>"); ok {
		if _, ok := r.tok.SpecialID("<|im_end|>"); ok {
			return "chatml"
		}
	}
	if _, ok := r.tok.SpecialID("<|user|>"); ok {
		if _, ok := r.tok.SpecialID("<|assistant|>"); ok {
			if _, ok := r.tok.SpecialID("<|end|>"); ok {
				return "phi-chat"
			}
		}
	}
	if _, ok := r.tok.SpecialID("<｜User｜>"); ok {
		if _, ok := r.tok.SpecialID("<｜Assistant｜>"); ok {
			return "deepseek-r1-qwen"
		}
	}
	if _, ok := r.tok.SpecialID("<|start_of_role|>"); ok {
		if _, ok := r.tok.SpecialID("<|end_of_role|>"); ok {
			return "granite-chat"
		}
	}
	if _, ok := r.tok.SpecialID("<start_of_turn>"); ok {
		if _, ok := r.tok.SpecialID("<end_of_turn>"); ok {
			return "gemma-chat"
		}
	}
	return ""
}

func specialOr(t *Tokenizer, token string, fallback uint32) uint32 {
	if id, ok := t.SpecialID(token); ok {
		return id
	}
	return fallback
}

func specialOrEncoded(t *Tokenizer, token string) uint32 {
	if id, ok := t.SpecialID(token); ok {
		return id
	}
	ids := t.EncodeWithoutBOS(token)
	if len(ids) > 0 {
		return ids[0]
	}
	return 0
}

func meanPoolInPlace(values []float32, count int) {
	if count == 0 {
		return
	}
	scale := float32(1) / float32(count)
	for i := range values {
		values[i] *= scale
	}
}

func l2NormalizeInPlace(values []float32) {
	var ss float32
	for _, v := range values {
		ss += v * v
	}
	norm := float32(math.Sqrt(float64(ss)))
	if norm > 1e-8 {
		for i := range values {
			values[i] /= norm
		}
	}
}

func CosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("cosine_similarity: dimension mismatch (%d vs %d)", len(a), len(b))
	}
	if len(a) == 0 {
		return 0, fmt.Errorf("cosine_similarity: empty vectors")
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(normA)) * math.Sqrt(float64(normB)))
	if denom <= 1e-12 {
		return 0, fmt.Errorf("cosine_similarity: zero-norm vector encountered")
	}
	return dot / denom, nil
}
