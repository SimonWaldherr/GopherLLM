package gopherllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

var ErrGenerationCanceled = errors.New("generation canceled")

// repeatPenaltyWindow bounds the prompt and output history considered by the
// sampler's repeat penalty. Keeping the same window from the first sampled
// token avoids a long-prompt allocation and an O(prompt length) first-token
// penalty pass.
const repeatPenaltyWindow = 64

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
	// ContextWindowMode controls whether an oversized chat history fails as it
	// historically did (full, the zero-value behavior), is reduced to the
	// newest complete turns before rendering (recent), or lexically condensed
	// before applying that same complete-turn selection (autoCompress).
	ContextWindowMode ContextWindowMode
	// Tools lists the functions the model may call. When non-empty, it is
	// rendered into the prompt using the active chat template's tool-calling
	// convention (native for Mistral, a generic <tool_call> JSON convention
	// otherwise).
	Tools []ToolDefinition
	// ToolChoice controls which of Tools are offered. "none" suppresses tool
	// rendering entirely; a value of the form "function:<name>" (as produced
	// by an OpenAI-style tool_choice object naming one function) narrows
	// offering to just that tool; any other value (including the default
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

// WithContext returns a copy of the options bound to ctx, so generation is
// cancelled between prefill chunks and decoded tokens when ctx is done. The
// context field itself stays unexported (it must not be JSON-decoded from a
// request body); this is how out-of-package servers attach a request context.
func (o GenerationOptions) WithContext(ctx context.Context) GenerationOptions {
	o.ctx = ctx
	return o
}

func DefaultGenerationOptions() GenerationOptions {
	return GenerationOptions{MaxTokens: 256, Sampler: DefaultSamplerConfig(), SystemPrompt: "You are a helpful assistant."}
}

// ActiveTools returns the tools that should actually be offered to the model
// for this request, honoring ToolChoice: "none" (suppress) and
// "function:<name>" (narrow to the single named tool, degrading back to all
// of Tools if no such name exists).
func (o GenerationOptions) ActiveTools() []ToolDefinition {
	if o.ToolChoice == "none" {
		return nil
	}
	if name, ok := strings.CutPrefix(o.ToolChoice, "function:"); ok {
		for _, t := range o.Tools {
			if t.Function.Name == name {
				return []ToolDefinition{t}
			}
		}
	}
	return o.Tools
}

func (o GenerationOptions) Validate() error {
	if o.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be greater than 0")
	}
	if !o.ContextWindowMode.valid() {
		return fmt.Errorf("context_window_mode must be full, recent, or autoCompress")
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

// PromptCacheInfo reports how much of a rendered prompt was already present
// in the Runner's bounded KV prefix cache. It is deliberately about exact
// token positions, not message count: templates, tool calls, and edited turns
// can all change the rendered sequence without changing an obvious UI label.
type PromptCacheInfo struct {
	Mode         string `json:"mode"`
	Hit          bool   `json:"hit"`
	ReusedTokens int    `json:"reused_tokens"`
	PromptTokens int    `json:"prompt_tokens"`
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
	// ContextWindow is populated for recent-context requests with the exact
	// rendered prompt selected for this particular generation call. Agentic
	// callers receive the final loop iteration's value.
	ContextWindow *ContextWindowInfo
	// PromptCache is populated for generation calls after the prompt has been
	// rendered. It describes the actual KV-prefix reuse for this call.
	PromptCache *PromptCacheInfo
}

type LoadInfo struct {
	FileSizeBytes int
	LoadTime      time.Duration
	// OutOfCore reports whether this model was loaded through the CPU-only
	// demand-paged path. It does not promise a fixed RSS cap: the OS controls
	// mmap page residency and may retain or evict file-backed pages as needed.
	OutOfCore bool
}

type LoadOptions struct {
	PrepareQuantized bool
	UseMetal         bool
	// OutOfCore keeps scalar and quantized matrices as views of a real mmap,
	// disables GPU/prepared copies, and avoids prewarming sparse expert banks.
	// It requires a single-file GGUF opened from a filesystem path.
	OutOfCore bool
	// Prefault selects the mmap warm-up policy. The zero value preserves the
	// historical full warm-up; OutOfCore changes its effective policy to core.
	Prefault  MmapPrefaultMode
	LogWriter io.Writer
}

type loadedKind int

const (
	loadedStandard loadedKind = iota
	loadedGptOss
	loadedGemma4
	loadedNemotronH
	loadedMamba2
	loadedBERT
	loadedQwen35
)

// prefixCacheState owns no additional KV memory: it points at the retained
// generation workspace, and tokens are exactly the positions resident in it.
// Runner.genLock protects the whole structure.
type prefixCacheState struct {
	cache  *KVCache
	tokens []uint32
}

// Runner is a fully loaded model ready to generate: parsed GGUF header,
// tokenizer, config, and weights (one of the three kind-specific sets).
// Generations and embeddings are serialized by genLock — a Runner is safe to
// share across goroutines (the HTTP server does), but runs one request at a
// time. Close releases the memory-mapped weight file; quantized weights
// borrow from it, so no method may be called after Close.
type Runner struct {
	gguf           *GGUFFile
	arch           string
	tok            *Tokenizer
	config         Config
	kind           loadedKind
	standard       ModelWeights
	gptOss         GptOssWeights
	gemma4         Gemma4Weights
	nemotronH      NemotronHWeights
	mamba2         Mamba2Weights
	bert           BERTWeights
	qwen35         Qwen35Weights
	genLock        sync.Mutex
	workspaceCache *KVCache
	workspaceBuf   *DecodeBuffer
	bertScratch    bertEmbeddingScratch
	prefixCache    prefixCacheState
	mappedFile     *MmapFile
	outOfCore      bool
}

// ArchitectureSupported reports whether the loader accepts this
// general.architecture value. Notes on specific families:
//
//   - qwen3 (incl. the DeepSeek-R1-0528 Qwen3 distills): the qwen2 graph plus
//     per-head QK-norm, which loads via the optional attn_q_norm/attn_k_norm
//     tensors and applies exactly as for Gemma 3/4.
//   - Mixtral/Llama-MoE, qwen2moe, qwen3moe, gpt-oss, and DeepSeek/Kimi use
//     sparse experts when their router tensors are present. deepseek2 uses a
//     dedicated MLA attention path and its sigmoid/noaux shared-expert router.
//   - qwen35/qwen35moe use the experimental native Gated-DeltaNet hybrid
//     loader. It has focused graph and local-GGUF smoke coverage, while
//     cross-runtime logit parity, vision, and MTP speculation remain out of
//     scope. qwen35moe shares the sparse-expert tensor path; a trailing
//     Qwen MTP draft layer is skipped for ordinary generation.
//   - Phi-3, dense Granite, EXAONE, and InternLM2 use the same pre-norm, RoPE, GQA and
//     SwiGLU graph as the standard loader. Their architecture-specific scales
//     are read from GGUF metadata by ConfigFromGGUF.
//   - Devstral and Mistral-Small GGUFs usually declare llama or mistral3;
//     their [INST]/Tekken behavior is picked up from tokenizer metadata, not
//     the arch string.
func ArchitectureSupported(arch string) bool {
	switch arch {
	case "llama", "llama2", "llama3", "mistral", "mistral3", "ministral", "mixtral",
		"qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen35", "qwen35moe", "deepseek2", "kimi_k2", "phi3", "granite", "exaone", "internlm2", "stablelm", "gpt-oss", "gemma", "gemma2", "gemma3", "gemma4", "nemotron_h", "nemotron_h_moe", "mamba2", "bert", "nomic-bert":
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
	if options.OutOfCore {
		return nil, fmt.Errorf("out-of-core loading requires RunnerFromPathWithOptions: byte-backed models already reside in memory")
	}
	return runnerFromGGUFBytes(data, false, options)
}

func runnerFromGGUFBytes(data []byte, borrowQuantized bool, options LoadOptions) (*Runner, error) {
	if err := validateLoadOptions(options); err != nil {
		return nil, err
	}
	if options.OutOfCore && !borrowQuantized {
		return nil, fmt.Errorf("out-of-core loading requires a memory-mapped model file")
	}
	gguf, err := ParseGGUF(data)
	if err != nil {
		return nil, err
	}
	return runnerFromParsedGGUF(data, gguf, borrowQuantized, options)
}

// runnerFromParsedGGUF builds a Runner from an already-parsed GGUFFile plus
// the byte slice its Tensors' Offsets are relative to (via DataOffset). It is
// split out from runnerFromGGUFBytes so the split-file loader (gguf_split.go)
// can hand in a synthetic GGUFFile assembled from multiple shard files
// without needing a real single-file byte stream to re-parse.
func runnerFromParsedGGUF(data []byte, gguf *GGUFFile, borrowQuantized bool, options LoadOptions) (*Runner, error) {
	if err := validateLoadOptions(options); err != nil {
		return nil, err
	}
	logw := options.LogWriter
	if logw == nil {
		logw = io.Discard
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
	r := &Runner{gguf: gguf, arch: arch, tok: tok, outOfCore: options.OutOfCore}
	if options.OutOfCore {
		fmt.Fprintln(logw, "Out-of-core: CPU mmap mode; sparse experts remain demand-paged (Metal and prepared quantization disabled)")
	}
	switch arch {
	case "bert", "nomic-bert":
		config, weights, err := LoadBERTModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.bert, r.kind = config, weights, loadedBERT
	case "nemotron_h", "nemotron_h_moe":
		config, weights, err := LoadNemotronHModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.nemotronH, r.kind = config, weights, loadedNemotronH
	case "mamba2":
		config, weights, err := LoadMamba2Model(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.mamba2, r.kind = config, weights, loadedMamba2
	case "qwen35", "qwen35moe":
		// Text-only decode is implemented and covered by focused graph tests and
		// local GGUF smoke tests. Keep the remaining unsupported capabilities
		// explicit instead of claiming general Qwen3.6 feature parity.
		fmt.Fprintln(logw, "Warning: qwen35 support is experimental: text-only decode is implemented; vision, MTP speculative decoding, and cross-runtime logit-parity validation are pending (see qwen35.go)")
		config, weights, err := LoadQwen35Model(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.qwen35, r.kind = config, weights, loadedQwen35
	case "gpt-oss":
		config, weights, err := LoadGptOssModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		r.config, r.gptOss, r.kind = config, weights, loadedGptOss
	case "gemma", "gemma2", "gemma3", "gemma4":
		config, weights, err := LoadGemma4Model(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
		if err != nil {
			return nil, err
		}
		if arch == "gemma4" && weights.Native {
			features := []string{"proportional RoPE", "K-as-V", "per-layer output scales", "native turn template"}
			if len(weights.MoE) > 0 {
				features = append(features, "shared-dense/MoE FFN")
			}
			if weights.PerLayer != nil {
				features = append(features, "per-layer embeddings")
			}
			if nativeGemma4KVCacheLayerCount(weights) < len(weights.Layers) {
				features = append(features, "shared KV cache")
			}
			fmt.Fprintf(logw, "Gemma 4: native SWA/global decoder enabled (%s)\n", strings.Join(features, ", "))
		} else {
			fmt.Fprintf(logw, "Warning: %s support is experimental; dense decoder support has not had cross-runtime logit-parity validation\n", arch)
		}
		r.config, r.gemma4, r.kind = config, weights, loadedGemma4
	default:
		config, weights, err := LoadModel(data, gguf, borrowQuantized, options.PrepareQuantized, options.UseMetal, logw, options.OutOfCore)
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
	if err := validateLoadOptions(options); err != nil {
		return nil, LoadInfo{}, err
	}
	t0 := time.Now()
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, LoadInfo{}, fmt.Errorf("failed to open model: %w", err)
	}
	if options.OutOfCore && !mmap.mmap {
		_ = mmap.Close()
		return nil, LoadInfo{}, fmt.Errorf("out-of-core loading requires an OS memory map; this file fell back to an in-memory read")
	}
	if header, herr := ParseGGUFQuiet(mmap.Bytes()); herr == nil {
		if _, count, ok := splitInfo(header); ok && count > 1 {
			if options.OutOfCore {
				_ = mmap.Close()
				return nil, LoadInfo{}, fmt.Errorf("out-of-core loading does not support split GGUFs: shards are currently merged into memory; combine the shards first")
			}
			r, mergedBytes, err := loadSplitRunner(path, header, mmap, options)
			if err != nil {
				return nil, LoadInfo{}, err
			}
			return r, LoadInfo{FileSizeBytes: int(mergedBytes), LoadTime: time.Since(t0), OutOfCore: false}, nil
		}
		if mmap.mmap {
			prefaultMappedModel(mmap.Bytes(), header, options)
		}
	} else if mmap.mmap {
		// A malformed model will fail in the real parser below. Preserve the
		// legacy full warm-up behavior when no trustworthy tensor map exists.
		if effectivePrefaultMode(options) == MmapPrefaultAll {
			prefaultPages(mmap.Bytes())
		}
	}
	// Quantized weights borrow sub-slices of the file buffer instead of
	// copying (multi-gigabyte models load without a second copy). The one
	// case that must copy: Metal builds where mmap fell back to os.ReadFile —
	// only an actual OS mapping may be retained by Metal with bytesNoCopy, a
	// C object must never keep a pointer into Go-managed heap memory. Without
	// Metal, borrowing from the heap buffer is plain Go slice aliasing and
	// always safe.
	r, err := runnerFromGGUFBytes(mmap.Bytes(), mmap.mmap || !options.UseMetal, options)
	if err != nil {
		_ = mmap.Close()
		return nil, LoadInfo{}, err
	}
	r.mappedFile = mmap
	return r, LoadInfo{FileSizeBytes: mmap.Len(), LoadTime: time.Since(t0), OutOfCore: options.OutOfCore}, nil
}

func (r *Runner) Architecture() string      { return r.arch }
func (r *Runner) Tokenizer() *Tokenizer     { return r.tok }
func (r *Runner) GGUF() *GGUFFile           { return r.gguf }
func (r *Runner) Config() Config            { return r.config }
func (r *Runner) ModelName() (string, bool) { return r.gguf.GetString("general.name") }
func (r *Runner) OutOfCore() bool           { return r != nil && r.outOfCore }

func (r *Runner) Close() error {
	if r == nil {
		return nil
	}
	r.genLock.Lock()
	defer r.genLock.Unlock()
	r.releaseMetalWeights()
	r.workspaceCache = nil
	r.workspaceBuf = nil
	r.bertScratch = bertEmbeddingScratch{}
	r.prefixCache = prefixCacheState{}
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
	case loadedBERT:
		releaseBERTMetalWeights(&r.bert)
	case loadedGptOss:
		releaseModelMetalWeights(&r.gptOss.Standard)
	case loadedGemma4:
		releaseGemma4MetalWeights(&r.gemma4)
	case loadedNemotronH:
		releaseNemotronHMetalWeights(&r.nemotronH)
	case loadedMamba2:
		releaseMamba2MetalWeights(&r.mamba2)
	case loadedQwen35:
		releaseQwen35MetalWeights(&r.qwen35)
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
	if r.kind == loadedBERT {
		return GenerationResult{}, fmt.Errorf("%s is an embedding model and cannot generate chat completions", r.arch)
	}
	if err := options.Validate(); err != nil {
		return GenerationResult{}, err
	}
	if len(messages) == 0 {
		return GenerationResult{}, fmt.Errorf("no prompt provided")
	}
	totalStart := time.Now()
	var contextWindow *ContextWindowInfo
	// Do this at the shared generation entry point rather than only in the
	// HTTP handler. Agentic requests append assistant tool calls and tool
	// results between iterations; each subsequent model call must be budgeted
	// against those effective messages and active tools as well.
	if normalizedContextWindowMode(options.ContextWindowMode) != ContextWindowFull {
		prepared, info, err := r.PrepareChatContext(messages, options)
		if err != nil {
			return GenerationResult{}, err
		}
		messages = prepared
		contextWindow = &info
	}
	tokens := r.renderMessages(messages, options.SystemPrompt, options.ActiveTools())
	if len(tokens) == 0 {
		return GenerationResult{}, fmt.Errorf("prompt rendered to zero tokens")
	}
	if r.config.MaxSeqLen <= 0 {
		return GenerationResult{}, fmt.Errorf("model has an invalid context length (%d)", r.config.MaxSeqLen)
	}
	// The KV cache is sized to r.config.MaxSeqLen; without this check a prompt
	// at or beyond that length (easily reached once a request injects a large
	// tool listing) would silently overflow it deeper in the forward pass
	// instead of failing here with a clear error.
	if len(tokens) >= r.config.MaxSeqLen {
		return GenerationResult{}, fmt.Errorf("prompt (%d tokens) leaves no room to generate within the model's context length (%d)", len(tokens), r.config.MaxSeqLen)
	}
	cacheLen := generationCacheLen(r.config.MaxSeqLen, len(tokens), options.MaxTokens)
	cache, buf := r.generationWorkspace(cacheLen)
	cacheInfo := PromptCacheInfo{Mode: "disabled", PromptTokens: len(tokens)}
	reusedTokens := 0
	cacheEligible := r.prefixCacheSupported(cache)
	if cacheEligible {
		cacheInfo.Mode = "prefix"
		reusedTokens = r.prefixReuse(cache, tokens)
		cacheInfo.Hit = reusedTokens > 0
		cacheInfo.ReusedTokens = reusedTokens
	}
	seed := options.Seed
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	rng := NewRng(seed)
	ctx := options.generationContext()
	if err := ctx.Err(); err != nil {
		return GenerationResult{}, err
	}
	// Once this request writes into the shared workspace, prior metadata is no
	// longer safe. The deferred replacement records only resident KV rows.
	if cacheEligible {
		r.clearPrefixCache()
	}
	prefillOffset := reusedTokens
	if prefillOffset == len(tokens) {
		prefillOffset = max(0, len(tokens)-1)
		cacheInfo.Hit = prefillOffset > 0
		cacheInfo.ReusedTokens = prefillOffset
	}
	prefillBegan := time.Now()
	logits := []float32{}
	residentTokens := []uint32(nil)
	defer func() {
		if !cacheEligible {
			return
		}
		if len(residentTokens) == 0 {
			r.clearPrefixCache()
			return
		}
		state := prefixCacheState{cache: cache, tokens: append([]uint32(nil), residentTokens...)}
		r.prefixCache = state
	}()
	if prefillOffset < len(tokens) {
		if r.canBatchPrefill() {
			if err := r.prefillBatchedAt(ctx, cache, buf, tokens[prefillOffset:], prefillOffset, &logits); err != nil {
				return GenerationResult{}, err
			}
		} else {
			for pos := prefillOffset; pos < len(tokens); pos++ {
				if err := ctx.Err(); err != nil {
					return GenerationResult{}, err
				}
				if pos == len(tokens)-1 {
					r.forwardTokenInto(cache, buf, tokens[pos], pos, &logits)
				} else {
					r.forwardPrefillToken(cache, buf, tokens[pos], pos)
				}
			}
		}
	}
	residentTokens = append(residentTokens, tokens...)
	prefillTime := time.Since(prefillBegan)
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
	recent := recentTokenWindow(tokens)
	pos := len(tokens)
	finishReason := "length"
	greedyFastPath := r.canGreedyOutputFastPath(options)
	haveNextToken := false
	var nextToken uint32
	buildResult := func() GenerationResult {
		stats := GenerationStats{PromptTokens: len(tokens), GeneratedTokens: len(generated), TTFT: ttft, PrefillTime: prefillTime, DecodeTime: time.Since(decodeStart), TotalTime: time.Since(totalStart)}
		content, reasoning, calls := r.classifyOutput(output.String(), options.ActiveTools(), rng)
		reason := finishReason
		if len(calls) > 0 {
			reason = "tool_calls"
		}
		return GenerationResult{Text: content, ReasoningText: reasoning, ToolCalls: calls, FinishReason: reason, Stats: stats, ContextWindow: contextWindow, PromptCache: &cacheInfo}
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
		if len(recent) > repeatPenaltyWindow {
			recent = recent[len(recent)-repeatPenaltyWindow:]
		}
		if len(generated) >= options.MaxTokens || pos >= cacheLen {
			break
		}
		if greedyFastPath {
			var ok bool
			nextToken, ok = r.forwardGreedyToken(cache, buf, token, pos, &logits)
			haveNextToken = ok
			if cacheEligible {
				residentTokens = append(residentTokens, token)
			}
		} else {
			r.forwardTokenInto(cache, buf, token, pos, &logits)
			if cacheEligible {
				residentTokens = append(residentTokens, token)
			}
		}
		pos++
	}
	if !flushStream(true) {
		return buildResult(), ErrGenerationCanceled
	}
	return buildResult(), nil
}

// generationCacheLen returns the cache length for a prompt with a positive
// model context. It performs the context cap before adding MaxTokens so an
// untrusted large max_tokens value cannot overflow an int and turn into a
// negative cache allocation.
func generationCacheLen(maxSeqLen, promptTokens, maxTokens int) int {
	remaining := maxSeqLen - promptTokens
	if maxTokens >= remaining-1 {
		return maxSeqLen
	}
	return promptTokens + maxTokens + 1
}

func recentTokenWindow(tokens []uint32) []uint32 {
	start := max(0, len(tokens)-repeatPenaltyWindow)
	return append([]uint32(nil), tokens[start:]...)
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
	if r.kind == loadedMamba2 {
		// Pure Mamba2 has no attention graph. Keep an empty KVCache shell so
		// generation's common workspace lifecycle can own its recurrent state.
		return 0, 0, 0, 0, 0
	}
	if r.kind == loadedNemotronH {
		// Soofi uses a shared attention shape on its six attention blocks;
		// Mamba/MoE blocks have no K/V entries but retain their layer index.
		return r.config.NKVHeads * r.config.HeadDim, r.config.KVDim, r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
	}
	if r.kind == loadedGemma4 {
		if r.gemma4.Native {
			maxKDim, maxVDim, maxHD, maxKV, maxVal := 0, 0, 0, 0, 0
			for _, l := range r.gemma4.Layers {
				maxKDim = max(maxKDim, l.NKVHeads*l.HeadDim)
				maxVDim = max(maxVDim, l.NKVHeads*l.ValueDim)
				maxHD = max(maxHD, l.HeadDim)
				maxKV = max(maxKV, l.NKVHeads)
				maxVal = max(maxVal, l.ValueDim)
			}
			return maxKDim, maxVDim, maxHD, maxKV, maxVal
		}
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

// kvCacheLayerCount is normally the model's decoder depth. Qwen3.5/3.6 is a
// hybrid graph, though: only its periodic full-attention layers ever access
// K/V rows. DeltaNet layers keep their separate recurrent state, so reserving
// K/V for every decoder layer wastes three quarters of the cache for the
// standard every-fourth-layer schedule.
func (r *Runner) kvCacheLayerCount() int {
	if r.kind == loadedQwen35 {
		return qwen35AttentionLayerCount(r.qwen35)
	}
	if r.kind == loadedGemma4 && r.gemma4.Native {
		return nativeGemma4KVCacheLayerCount(r.gemma4)
	}
	return r.config.NLayers
}

const maxReusableKVCacheBytes int64 = 512 << 20

func kvCacheBytes(layers, kDim, vDim, cacheLen int, f16 bool) int64 {
	elemBytes := int64(4)
	if f16 {
		elemBytes = 2
	}
	return int64(layers) * int64(kDim+vDim) * int64(cacheLen) * elemBytes
}

func grownKVCacheLen(current, required, limit int, config Config) int {
	if current <= 0 {
		return required
	}
	grown := current * 2
	if grown < current { // integer overflow: the required size is safer.
		grown = required
	}
	grown = max(grown, current+prefillChunkSize(config))
	target := max(required, grown)
	if limit > 0 {
		target = min(target, limit)
	}
	return target
}

// copyKVPrefix transfers complete token rows between shape-compatible caches.
// It intentionally excludes Nemotron-H's recurrent state; prefix reuse for
// that hybrid architecture stays disabled until its state can be copied too.
func copyKVPrefix(dst, src *KVCache, positions int) int {
	if dst == nil || src == nil || positions <= 0 || dst.F16 != src.F16 ||
		dst.PerPosKDim != src.PerPosKDim || dst.PerPosVDim != src.PerPosVDim ||
		dst.layerCount() != src.layerCount() {
		return 0
	}
	positions = min(positions, min(dst.MaxLen, src.MaxLen))
	if positions <= 0 {
		return 0
	}
	kLen := positions * dst.PerPosKDim
	vLen := positions * dst.PerPosVDim
	for layer := 0; layer < dst.layerCount(); layer++ {
		if dst.F16 {
			copy(dst.K16[layer][:kLen], src.K16[layer][:kLen])
			copy(dst.V16[layer][:vLen], src.V16[layer][:vLen])
			continue
		}
		copy(dst.K[layer][:kLen], src.K[layer][:kLen])
		copy(dst.V[layer][:vLen], src.V[layer][:vLen])
	}
	return positions
}

func sharedTokenPrefix(a, b []uint32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func (r *Runner) clearPrefixCache() {
	r.prefixCache = prefixCacheState{}
}

func (r *Runner) prefixCacheSupported(cache *KVCache) bool {
	// Recurrent Mamba/DeltaNet state must be replayed from the beginning.
	// Reusing only the attention K/V rows (or none at all for pure Mamba2)
	// would be wrong.
	return r.kind != loadedNemotronH && r.kind != loadedMamba2 && r.kind != loadedQwen35 && cache != nil && cache == r.workspaceCache
}

// prefixReuse returns the exact number of resident KV positions that can be
// skipped. An identical prompt re-runs only its final token because sampling
// mutates logits in place; retaining raw logits would otherwise make a retry
// subtly diverge from a cold run.
func (r *Runner) prefixReuse(cache *KVCache, tokens []uint32) int {
	state := r.prefixCache
	if state.cache != cache || len(state.tokens) == 0 || cache == nil {
		return 0
	}
	matched := min(sharedTokenPrefix(tokens, state.tokens), cache.MaxLen)
	if matched == 0 {
		return 0
	}
	if matched == len(tokens) {
		return max(0, matched-1)
	}
	return matched
}

// generationWorkspace reuses request scratch behind genLock. The retained
// cache also serves as a single bounded, token-verified prefix cache for the
// most recent conversation. When it grows, copy known K/V rows geometrically
// rather than re-prefilling prior turns. Retaining at most 512 MiB avoids
// turning one unusually large context into a permanent memory commitment.
func (r *Runner) generationWorkspace(cacheLen int) (*KVCache, *DecodeBuffer) {
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	layers := r.kvCacheLayerCount()
	old := r.workspaceCache
	shapeCompatible := old != nil && old.layerCount() == layers && old.F16 == useF16KVCache &&
		old.PerPosKDim == kDim && old.PerPosVDim == vDim
	cache := old
	compatible := shapeCompatible && old.MaxLen >= cacheLen
	if !compatible {
		targetLen := cacheLen
		if shapeCompatible {
			targetLen = grownKVCacheLen(old.MaxLen, cacheLen, r.config.MaxSeqLen, r.config)
		}
		// Geometric headroom is useful only while it stays within the same
		// retention budget. Never make a one-off request allocate a larger
		// temporary cache merely because the next growth step crossed 512 MiB.
		if targetLen > cacheLen && kvCacheBytes(layers, kDim, vDim, targetLen, useF16KVCache) > maxReusableKVCacheBytes {
			targetLen = cacheLen
		}
		cache = newKVCacheAuto(layers, kDim, vDim, targetLen)
		bytes := kvCacheBytes(layers, kDim, vDim, targetLen, cache.F16)
		if bytes <= maxReusableKVCacheBytes {
			r.workspaceCache = cache
			if shapeCompatible && r.prefixCache.cache == old && r.kind != loadedNemotronH && r.kind != loadedMamba2 && r.kind != loadedQwen35 {
				if copied := copyKVPrefix(cache, old, len(r.prefixCache.tokens)); copied == len(r.prefixCache.tokens) {
					r.prefixCache.cache = cache
				} else {
					r.clearPrefixCache()
				}
			} else if r.prefixCache.cache == old {
				r.clearPrefixCache()
			}
		} else if !shapeCompatible && r.prefixCache.cache == old {
			r.clearPrefixCache()
		}
	}
	if r.kind == loadedNemotronH || r.kind == loadedMamba2 {
		if !cache.Nemotron.compatible(r.config) {
			cache.Nemotron = newNemotronHCache(r.config)
		}
		// Unlike attention K/V, recurrent state is read before a position can
		// overwrite it, so a reused workspace must start every request at zero.
		cache.Nemotron.reset()
	}
	if r.kind == loadedQwen35 {
		recurrentLayers := qwen35RecurrentLayerCount(r.qwen35)
		if !cache.Qwen35.compatible(r.config, recurrentLayers) {
			cache.Qwen35 = newQwen35Cache(r.config, recurrentLayers)
		}
		cache.Qwen35.reset()
	}
	if r.workspaceBuf == nil {
		r.workspaceBuf = NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	}
	return cache, r.workspaceBuf
}

func (r *Runner) forwardTokenInto(cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	switch r.kind {
	case loadedNemotronH:
		ForwardNemotronHInto(r.config, r.nemotronH, cache, buf, token, pos, logits)
	case loadedMamba2:
		ForwardMamba2Into(r.config, r.mamba2, cache, buf, token, pos, logits)
	case loadedQwen35:
		ForwardQwen35Into(r.config, r.qwen35, cache, buf, token, pos, logits)
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
	case loadedNemotronH:
		ForwardNemotronHBodyInto(r.config, r.nemotronH, cache, buf, token, pos)
		if next, ok := argmaxOutputTokenInto(r.config, ModelWeights{Output: r.nemotronH.Output}, buf, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, ModelWeights{Output: r.nemotronH.Output}, buf, logits)
	case loadedMamba2:
		ForwardMamba2BodyInto(r.config, r.mamba2, cache, buf, token, pos)
		if next, ok := argmaxOutputTokenInto(r.config, ModelWeights{Output: r.mamba2.Output}, buf, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, ModelWeights{Output: r.mamba2.Output}, buf, logits)
	case loadedQwen35:
		ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
		if next, ok := argmaxOutputTokenInto(r.config, ModelWeights{Output: r.qwen35.Output}, buf, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, ModelWeights{Output: r.qwen35.Output}, buf, logits)
	case loadedGptOss:
		ForwardBodyInto(r.config, r.gptOss.Standard, cache, buf, token, pos)
		if next, ok := argmaxOutputTokenInto(r.config, r.gptOss.Standard, buf, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.gptOss.Standard, buf, logits)
	case loadedGemma4:
		if r.gemma4.Native {
			forwardNativeGemma4BodyInto(r.config, r.gemma4, cache, buf, token, pos)
			nativeOutput := ModelWeights{Output: r.gemma4.Output}
			if next, ok := argmaxOutputTokenInto(r.config, nativeOutput, buf, logits); ok {
				return next, true
			}
			projectNativeGemma4Logits(r.config, r.gemma4, buf, logits)
			break
		}
		ForwardBodyInto(r.config, r.gemma4.Standard, cache, buf, token, pos)
		if next, ok := argmaxOutputTokenInto(r.config, r.gemma4.Standard, buf, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.gemma4.Standard, buf, logits)
	default:
		ForwardBodyInto(r.config, r.standard, cache, buf, token, pos)
		if next, ok := argmaxOutputTokenInto(r.config, r.standard, buf, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.standard, buf, logits)
	}
	return 0, false
}

func (r *Runner) forwardHiddenToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	switch r.kind {
	case loadedNemotronH:
		ForwardNemotronHBodyInto(r.config, r.nemotronH, cache, buf, token, pos)
	case loadedMamba2:
		ForwardMamba2BodyInto(r.config, r.mamba2, cache, buf, token, pos)
	case loadedQwen35:
		ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
	case loadedGptOss:
		ForwardBodyInto(r.config, r.gptOss.Standard, cache, buf, token, pos)
	case loadedGemma4:
		if r.gemma4.Native {
			forwardNativeGemma4BodyInto(r.config, r.gemma4, cache, buf, token, pos)
		} else {
			ForwardBodyInto(r.config, r.gemma4.Standard, cache, buf, token, pos)
		}
	default:
		ForwardBodyInto(r.config, r.standard, cache, buf, token, pos)
	}
	return buf.XN
}

// canBatchPrefill reports whether the model uses the standard non-fused
// transformer path that ForwardBatchInto supports.
func (r *Runner) canBatchPrefill() bool {
	if r.outOfCore || r.kind != loadedStandard || r.config.UsesMLA || len(r.standard.Layers) == 0 {
		return false
	}
	if os.Getenv("GOPHERLLM_NO_BATCH_PREFILL") != "" {
		return false
	}
	// The batched graph implements both RMSNorm and LayerNorm, including
	// StableLM's parallel attention/FFN residual. Gemma-family mechanisms
	// such as QK/post norms still fall back to the per-token path.
	for _, l := range r.standard.Layers {
		if l.MoE != nil || l.AttnQNorm != nil || l.AttnKNorm != nil || l.PostAttnNorm != nil || l.PostFFNNorm != nil {
			return false
		}
	}
	return true
}

// prefillBatched processes the prompt in chunks, streaming each weight once per
// chunk instead of once per token.
//
// Generic synthetic sweeps peak near 32 as activation buffers outgrow cache,
// while the dense 3B path benefits from more dequantization amortization and
// uses the model-aware default below. Deployment-specific A/B runs can override
// either choice through GOPHERLLM_PREFILL_CHUNK.
func (r *Runner) prefillBatched(ctx context.Context, cache *KVCache, buf *DecodeBuffer, tokens []uint32, logits *[]float32) error {
	return r.prefillBatchedAt(ctx, cache, buf, tokens, 0, logits)
}

// prefillBatchedAt is prefillBatched with an absolute KV-cache offset. The
// offset is what lets an append-only chat process only the new rendered-token
// suffix while attention still reads the cached prefix rows.
func (r *Runner) prefillBatchedAt(ctx context.Context, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, logits *[]float32) error {
	chunk := prefillChunkSize(r.config)
	n := len(tokens)
	for start := 0; start < n; start += chunk {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+chunk, n)
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens[start:end], startPos+start, end == n, logits)
	}
	return nil
}

// prefillChunkOverride, when positive, wins over both the env var and the
// model-aware default. The autotuner sets it after measuring the real batched
// forward on the real hardware, which beats any static heuristic.
var prefillChunkOverride atomic.Int64

// SetPrefillChunk pins the prompt-prefill chunk size; 0 restores the default
// (env var, else the model-aware heuristic).
func SetPrefillChunk(n int) {
	if n < 0 {
		n = 0
	}
	prefillChunkOverride.Store(int64(n))
}

// prefillChunkSize resolves the chunk size actually used: an explicit override
// (set by the autotuner) wins, otherwise the env var, otherwise the
// model-aware default.
func prefillChunkSize(config Config) int {
	if n := int(prefillChunkOverride.Load()); n > 0 {
		return n
	}
	return prefillChunkDefault(config)
}

// prefillChunkOverrideValue reports the raw override, 0 when unset. Callers that
// save and restore the setting must use this rather than prefillChunkSize:
// re-applying a resolved value would pin it as an explicit override and thereby
// suppress the env var and the model-aware default.
func prefillChunkOverrideValue() int { return int(prefillChunkOverride.Load()) }

// prefillChunkDefault keeps a conservative default for larger models while using
// larger chunks on small dense models such as Ministral-3-3B. Bottleneck:
// prompt prefill is chunk-size sensitive because larger chunks amortize
// dequantization but grow activation working sets. Change: small models default
// to 128 after real Ministral measurement; GOPHERLLM_PREFILL_CHUNK can override
// it for A/B testing. Expected effect: lower TTFT on 3B-class models. Risk:
// too-large chunks can regress cache locality on bigger models. Rollback: set
// GOPHERLLM_PREFILL_CHUNK=32. --auto measures this instead of guessing it.
func prefillChunkDefault(config Config) int {
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
	case loadedNemotronH:
		ForwardNemotronHBodyInto(r.config, r.nemotronH, cache, buf, token, pos)
	case loadedMamba2:
		ForwardMamba2BodyInto(r.config, r.mamba2, cache, buf, token, pos)
	case loadedQwen35:
		ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
	case loadedGptOss:
		ForwardPrefill(r.config, r.gptOss.Standard, cache, buf, token, pos)
	case loadedGemma4:
		if r.gemma4.Native {
			forwardNativeGemma4BodyInto(r.config, r.gemma4, cache, buf, token, pos)
		} else {
			ForwardPrefill(r.config, r.gemma4.Standard, cache, buf, token, pos)
		}
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
	if r.kind == loadedBERT {
		return r.embedBERT(text)
	}
	// Embeddings reuse the same scratch KV workspace but have unrelated token
	// positions, so they must never overwrite a live chat-prefix cache.
	r.clearPrefixCache()
	tokens := r.tok.Encode(text)
	if len(tokens) == 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: input tokenised to zero tokens")
	}
	if r.config.MaxSeqLen > 0 && len(tokens) > r.config.MaxSeqLen {
		return EmbeddingResult{}, fmt.Errorf("embed: input (%d tokens) exceeds the model's context length (%d)", len(tokens), r.config.MaxSeqLen)
	}
	cacheLen := min(r.config.MaxSeqLen, len(tokens)+1)
	cache, buf := r.generationWorkspace(cacheLen)
	sum := make([]float32, r.config.Dim)
	if r.canBatchPrefill() {
		// Mean pooling needs every final hidden state, not just the last
		// prompt logit. The batched path streams each projection weight once
		// per chunk and accumulates those states before its scratch is reused.
		chunk := prefillChunkSize(r.config)
		for start := 0; start < len(tokens); start += chunk {
			end := min(start+chunk, len(tokens))
			forwardBatchPoolInto(r.config, r.standard, cache, buf, tokens[start:end], start, sum)
		}
	} else {
		for pos, tok := range tokens {
			h := r.forwardHiddenToken(cache, buf, tok, pos)
			for i, v := range h {
				sum[i] += v
			}
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
	if r.chatTemplateKind() == "kimi-chat" {
		// Kimi K2 declares [EOS] as its tokenizer EOS, but its instruct
		// generation configuration ends an assistant turn with <|im_end|>.
		if id, ok := r.tok.SpecialID("<|im_end|>"); ok && token == id {
			return true
		}
	}
	if qwen35Family(r.arch) {
		// Qwen3.5/3.6 opens assistant turns with ChatML and terminates them
		// with <|im_end|>. Its tokenizer EOS is not consistently configured to
		// that turn marker across converted GGUFs.
		if id, ok := r.tok.SpecialID("<|im_end|>"); ok && token == id {
			return true
		}
	}
	if gemmaFamily(r.arch) {
		if r.arch == "gemma4" {
			// Gemma 4's native turn delimiter replaces the older
			// <end_of_turn> marker used by Gemma 1--3.
			if id, ok := r.tok.SpecialID("<turn|>"); ok && token == id {
				return true
			}
		}
		// Gemma instruct models end assistant turns with <end_of_turn>, not
		// the <eos> the GGUF declares as EOS.
		if id, ok := r.tok.SpecialID("<end_of_turn>"); ok && token == id {
			return true
		}
	}
	return token == r.tok.EOSID
}

// renderMessages renders the conversation (and, if any, the tool listing) into
// tokens using the active chat template. Mistral, Llama 3.1, and Kimi get
// their native tool conventions; every other template (and gpt-oss, for
// which tool calling is not yet implemented) uses the generic <tool_call>
// JSON convention, applied uniformly by flattening tool listings and
// tool-call history into ordinary system/user/assistant text before
// delegating to the per-family renderer below.
func (r *Runner) renderMessages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) []uint32 {
	if r.arch == "gpt-oss" {
		return r.renderGptOssMessages(messages, systemPrompt)
	}
	if qwen35Family(r.arch) {
		if tokens, ok := r.renderQwen35Messages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	if r.chatTemplateKind() == "kimi-chat" {
		if tokens, ok := r.renderKimiMessages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	if r.arch == "nemotron_h_moe" {
		generic, genericSystem := injectGenericTools(messages, systemPrompt, tools)
		if tokens, ok := r.renderSoofiIsarMessages(generic, genericSystem); ok {
			return tokens
		}
	}
	if r.chatTemplateKind() == "mistral-inst" {
		if tokens, ok := r.renderMistralInstMessages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	if r.chatTemplateKind() == "llama31-chat" {
		if tokens, ok := r.renderLlama31Messages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	generic, genericSystem := injectGenericTools(messages, systemPrompt, tools)
	switch r.chatTemplateKind() {
	case "gemma4-chat":
		if tokens, ok := r.renderGemma4Messages(generic, genericSystem); ok {
			return tokens
		}
	case "gemma-chat":
		if tokens, ok := r.renderGemmaMessages(generic, genericSystem); ok {
			return tokens
		}
	case "header-chat":
		if tokens, ok := r.renderHeaderChatMessages(generic, genericSystem); ok {
			return tokens
		}
	case "llama31-chat":
		// A historical turn which the native Llama 3.1 template cannot replay
		// (for example, multiple calls in one assistant message) still needs a
		// lossless fallback rather than silently dropping tool state.
		if tokens, ok := r.renderHeaderChatMessages(generic, genericSystem); ok {
			return tokens
		}
	case "chatml":
		if tokens, ok := r.renderChatMLMessages(generic, genericSystem); ok {
			return tokens
		}
	case "phi4-chat":
		if tokens, ok := r.renderPhi4Messages(generic, genericSystem); ok {
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

// renderGemma4Messages implements the text-only part of Gemma 4's native
// template.  Tool-call serialisation is intentionally delegated to the
// generic formatter above; the important part for ordinary chat is preserving
// Gemma 4's turn and disabled-thinking channel markers:
//
//	<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
func (r *Runner) renderGemma4Messages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	startTurn, ok1 := r.tok.SpecialID("<|turn>")
	endTurn, ok2 := r.tok.SpecialID("<turn|>")
	channelStart, ok3 := r.tok.SpecialID("<|channel>")
	channelEnd, ok4 := r.tok.SpecialID("<channel|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := make([]uint32, 0, 64)
	if r.tok.AddBOS {
		tokens = append(tokens, r.tok.BOSID)
	}
	appendTurn := func(role, content string) {
		tokens = append(tokens, startTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n")...)
		if content = strings.TrimSpace(content); content != "" {
			tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
		}
		tokens = append(tokens, endTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	hasSystem := false
	for _, message := range messages {
		hasSystem = hasSystem || message.Role == ChatRoleSystem
	}
	if system := strings.TrimSpace(systemPrompt); system != "" && !hasSystem {
		appendTurn("system", system)
	}
	for _, message := range messages {
		role := "user"
		switch message.Role {
		case ChatRoleSystem:
			role = "system"
		case ChatRoleAssistant:
			role = "model"
		}
		appendTurn(role, message.Content)
	}
	tokens = append(tokens, startTurn)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("model\n")...)
	// Some Gemma 4 checkpoints (e.g. 12B/26B) open and immediately close the
	// thought channel here when reasoning is disabled, so generation lands on
	// a fresh channel rather than the old Gemma <start_of_turn> protocol.
	// Others (E2B) never emit this in their own add_generation_prompt branch
	// at all; injecting it there is out-of-distribution and derails
	// generation into gibberish. Checked per-checkpoint against its own
	// chat_template rather than assumed for every gemma4-chat model.
	if r.gemma4ClosesThoughtChannelAtGenerate() {
		tokens = append(tokens, channelStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("thought\n")...)
		tokens = append(tokens, channelEnd)
	}
	return tokens, true
}

// gemma4ClosesThoughtChannelAtGenerate reports whether this checkpoint's own
// chat_template appends '<|channel>thought\n<channel|>' right after
// '<|turn>model\n' in its add_generation_prompt branch. That literal only
// appears there for models whose default-thinking behaviour needs an
// explicit closed channel; the same substring also shows up (via string
// concatenation, not this literal) when serialising a past assistant turn's
// reasoning back into history, so a plain substring check on that other
// occurrence would be a false positive - hence the exact literal match.
func (r *Runner) gemma4ClosesThoughtChannelAtGenerate() bool {
	if r.gguf == nil {
		return false
	}
	v, ok := r.gguf.Metadata["tokenizer.chat_template"]
	if !ok {
		return false
	}
	s, ok := v.AsString()
	if !ok {
		return false
	}
	return strings.Contains(s, `<|channel>thought\n<channel|>`)
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

const (
	llama31KnowledgeCutoff = "December 2023"
	llama31DefaultDate     = "26 Jul 2024"
)

// renderLlama31Messages mirrors Meta's Llama-3.1-Instruct chat template for
// custom function tools. In particular, the template always starts with its
// dated system envelope, puts custom tool definitions into the first user
// turn, serializes an assistant call as {"name", "parameters"}, and feeds a
// tool result back using the ipython header. Built-in tools use Llama's
// separate <|python_tag|> protocol and intentionally remain outside the
// OpenAI-compatible ToolDefinition API.
func (r *Runner) renderLlama31Messages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]uint32, bool) {
	bot, ok1 := r.tok.SpecialID("<|begin_of_text|>")
	startHeader, ok2 := r.tok.SpecialID("<|start_header_id|>")
	endHeader, ok3 := r.tok.SpecialID("<|end_header_id|>")
	eot, ok4 := r.tok.SpecialID("<|eot_id|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}

	loopMessages := messages
	system := strings.TrimSpace(systemPrompt)
	// The native template consumes a leading system message into the mandatory
	// initial system envelope. This matches the ordinary Chat API convention
	// while leaving any later system message as a normal, replayable turn.
	if len(loopMessages) > 0 && loopMessages[0].Role == ChatRoleSystem {
		system = strings.TrimSpace(loopMessages[0].Content)
		loopMessages = loopMessages[1:]
	}

	// Meta's template with tools_in_user_message=true requires the first
	// remaining message to be a user message. Returning false here gives the
	// caller's generic fallback a chance to preserve an unusual history.
	toolUser := -1
	if len(tools) > 0 {
		if len(loopMessages) == 0 || loopMessages[0].Role != ChatRoleUser {
			return nil, false
		}
		toolUser = 0
	}

	tokens := make([]uint32, 0, 64)
	appendText := func(text string) {
		tokens = append(tokens, r.tok.EncodeWithoutBOS(text)...)
	}
	pushHeader := func(role string) {
		tokens = append(tokens, startHeader)
		appendText(role)
		tokens = append(tokens, endHeader)
		appendText("\n\n")
	}
	appendTurn := func(role, content string) {
		pushHeader(role)
		appendText(strings.TrimSpace(content))
		tokens = append(tokens, eot)
	}

	tokens = append(tokens, bot)
	pushHeader("system")
	if len(tools) > 0 {
		// This is emitted for custom tools too; it tells the model that the
		// result payload will be provided as an ipython turn.
		appendText("Environment: ipython\n")
	}
	appendText("Cutting Knowledge Date: " + llama31KnowledgeCutoff + "\n")
	appendText("Today Date: " + r.llama31TemplateDate() + "\n\n")
	appendText(system)
	tokens = append(tokens, eot)

	for i, message := range loopMessages {
		if i == toolUser {
			pushHeader("user")
			// Keep the otherwise slightly surprising lack of a newline between
			// the format sentence and "Do not use variables." This is how the
			// local Meta-Llama-3.1 GGUF's Jinja whitespace controls render it.
			appendText("Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n")
			appendText("Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n")
			for _, tool := range tools {
				payload, err := json.MarshalIndent(tool, "", "    ")
				if err != nil {
					return nil, false
				}
				appendText(string(payload))
				appendText("\n\n")
			}
			appendText(strings.TrimSpace(message.Content))
			tokens = append(tokens, eot)
			continue
		}

		switch message.Role {
		case ChatRoleTool:
			// Tool output is intentionally not trimmed. The native template
			// treats it as an ipython value rather than prose, so whitespace can
			// carry semantic meaning for a caller.
			pushHeader("ipython")
			appendText(message.Content)
			tokens = append(tokens, eot)
		case ChatRoleAssistant:
			if len(message.ToolCalls) == 0 {
				appendTurn("assistant", message.Content)
				continue
			}
			// Llama 3.1's custom-function branch explicitly supports one tool
			// call per assistant turn. Do not invent a multi-call encoding: use
			// the generic lossless fallback instead for such historical turns.
			if len(message.ToolCalls) != 1 {
				return nil, false
			}
			call := message.ToolCalls[0]
			args := strings.TrimSpace(call.Function.Arguments)
			if call.Function.Name == "" || len(args) == 0 || args[0] != '{' || !json.Valid([]byte(args)) {
				return nil, false
			}
			// Keep the literal separators from Meta's template rather than
			// using a generic compact wrapper: {"name": "…", "parameters": …}.
			// The arguments themselves are already JSON as required by the
			// OpenAI-compatible ToolCall surface.
			name, err := json.Marshal(call.Function.Name)
			if err != nil {
				return nil, false
			}
			pushHeader("assistant")
			appendText(`{"name": ` + string(name) + `, "parameters": ` + args + `}`)
			tokens = append(tokens, eot)
		case ChatRoleSystem:
			appendTurn("system", message.Content)
		default:
			appendTurn("user", message.Content)
		}
	}
	pushHeader("assistant")
	return tokens, true
}

// llama31TemplateDate reads the Jinja date_string assignment when present so
// closely related Llama 3.1 GGUF exports keep their own declared date. The
// inspected Meta-Llama-3.1-8B-Instruct template falls back to 26 Jul 2024.
func (r *Runner) llama31TemplateDate() string {
	if r == nil || r.gguf == nil {
		return llama31DefaultDate
	}
	v, ok := r.gguf.Metadata["tokenizer.chat_template"]
	if !ok {
		return llama31DefaultDate
	}
	template, ok := v.AsString()
	if !ok {
		return llama31DefaultDate
	}
	const assignment = "date_string"
	for start := 0; start < len(template); {
		i := strings.Index(template[start:], assignment)
		if i < 0 {
			break
		}
		i += start
		rest := strings.TrimSpace(template[i+len(assignment):])
		start = i + len(assignment)
		if !strings.HasPrefix(rest, "=") {
			continue
		}
		rest = strings.TrimSpace(rest[1:])
		if len(rest) < 2 || (rest[0] != '"' && rest[0] != 39) {
			continue
		}
		quote := rest[0]
		if end := strings.IndexByte(rest[1:], quote); end >= 0 {
			if date := rest[1 : end+1]; date != "" {
				return date
			}
		}
	}
	return llama31DefaultDate
}

const kimiDefaultSystemPrompt = "You are Kimi, an AI assistant created by Moonshot AI."

// renderKimiMessages mirrors Moonshot's Kimi-K2-Instruct chat_template.jinja:
//
//	<|im_system|>system<|im_middle|>{system}<|im_end|>
//	<|im_user|>user<|im_middle|>{user}<|im_end|>
//	<|im_assistant|>assistant<|im_middle|>
//
// Kimi's role markers are distinct from ChatML's <|im_start|> convention.
// Tool declarations, tool-call history, and tool results are rendered natively
// rather than through the generic <tool_call> fallback, since K2 was trained
// on its own control-token protocol.
func (r *Runner) renderKimiMessages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]uint32, bool) {
	systemTok, ok1 := r.tok.SpecialID("<|im_system|>")
	userTok, ok2 := r.tok.SpecialID("<|im_user|>")
	assistantTok, ok3 := r.tok.SpecialID("<|im_assistant|>")
	middleTok, ok4 := r.tok.SpecialID("<|im_middle|>")
	endTok, ok5 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2 && ok3 && ok4 && ok5) {
		return nil, false
	}

	tokens := make([]uint32, 0, 32)
	appendTurn := func(roleTok uint32, roleName, content string) {
		tokens = append(tokens, roleTok)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(roleName)...)
		tokens = append(tokens, middleTok)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
		tokens = append(tokens, endTok)
	}

	if len(tools) > 0 {
		payload, err := marshalKimiToolDefinitions(tools)
		if err != nil {
			return nil, false
		}
		appendTurn(systemTok, "tool_declare", string(payload))
	}

	hasSystem := false
	for _, message := range messages {
		if message.Role == ChatRoleSystem {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		system := strings.TrimSpace(systemPrompt)
		if system == "" {
			system = kimiDefaultSystemPrompt
		}
		appendTurn(systemTok, "system", system)
	}

	for _, message := range messages {
		roleName := message.Name
		if roleName == "" {
			switch message.Role {
			case ChatRoleSystem:
				roleName = "system"
			case ChatRoleAssistant:
				roleName = "assistant"
			case ChatRoleTool:
				roleName = "tool"
			default:
				roleName = "user"
			}
		}

		switch message.Role {
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(roleName)...)
			tokens = append(tokens, middleTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(message.Content)...)
			if len(message.ToolCalls) > 0 {
				callSectionStart, ok1 := r.tok.SpecialID("<|tool_calls_section_begin|>")
				callStart, ok2 := r.tok.SpecialID("<|tool_call_begin|>")
				argumentStart, ok3 := r.tok.SpecialID("<|tool_call_argument_begin|>")
				callEnd, ok4 := r.tok.SpecialID("<|tool_call_end|>")
				callSectionEnd, ok5 := r.tok.SpecialID("<|tool_calls_section_end|>")
				if !(ok1 && ok2 && ok3 && ok4 && ok5) {
					return nil, false
				}
				tokens = append(tokens, callSectionStart)
				for index, call := range message.ToolCalls {
					callID := call.ID
					if _, _, ok := parseKimiToolCallID(callID); !ok {
						callID = kimiToolCallID(call.Function.Name, index)
					}
					arguments := call.Function.Arguments
					if strings.TrimSpace(arguments) == "" {
						arguments = "{}"
					}
					tokens = append(tokens, callStart)
					tokens = append(tokens, r.tok.EncodeWithoutBOS(callID)...)
					tokens = append(tokens, argumentStart)
					tokens = append(tokens, r.tok.EncodeWithoutBOS(arguments)...)
					tokens = append(tokens, callEnd)
				}
				tokens = append(tokens, callSectionEnd)
			}
			tokens = append(tokens, endTok)
		case ChatRoleTool:
			appendTurn(systemTok, roleName, "## Return of "+message.ToolCallID+" "+message.Content)
		case ChatRoleSystem:
			appendTurn(systemTok, roleName, message.Content)
		default:
			appendTurn(userTok, roleName, message.Content)
		}
	}

	// add_generation_prompt=True from the official template.
	tokens = append(tokens, assistantTok)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant")...)
	tokens = append(tokens, middleTok)
	return tokens, true
}

// marshalKimiToolDefinitions reproduces Kimi's `deep_sort_dict` plus compact
// Jinja `tojson(separators=(',', ':'))` output. encoding/json sorts map keys,
// including nested decoded parameter schemas, so the prompt is deterministic
// and matches the official custom tokenizer's stable tool declaration form.
func marshalKimiToolDefinitions(tools []ToolDefinition) ([]byte, error) {
	payload := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		function := map[string]any{"name": tool.Function.Name}
		if tool.Function.Description != "" {
			function["description"] = tool.Function.Description
		}
		if len(tool.Function.Parameters) > 0 {
			var parameters any
			if err := json.Unmarshal(tool.Function.Parameters, &parameters); err != nil {
				return nil, fmt.Errorf("invalid parameters for Kimi tool %q: %w", tool.Function.Name, err)
			}
			function["parameters"] = parameters
		}
		payload = append(payload, map[string]any{
			"function": function,
			"type":     tool.Type,
		})
	}
	return json.Marshal(payload)
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
	// Some ChatML checkpoints (e.g. Nemotron-H) default enable_thinking to
	// true when a caller does not pass it, and their own add_generation_prompt
	// branch then opens an explicit, unclosed <think> tag rather than leaving
	// the assistant turn bare. Skipping that tag is out-of-distribution for
	// those checkpoints and derails generation; mirror it only for models
	// whose own chat_template actually does this by default.
	if think, ok := r.tok.SpecialID("<think>"); ok && r.chatMLOpensThinkTagByDefault() {
		tokens = append(tokens, think)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	return tokens, true
}

// chatMLOpensThinkTagByDefault reports whether this checkpoint's own
// chat_template emits '<|im_start|>assistant\n<think>\n' in its
// add_generation_prompt branch when enable_thinking is left at the
// template's own default (i.e. the literal appears unconditionally on one
// side of the enable_thinking check, not only when a caller opts in).
func (r *Runner) chatMLOpensThinkTagByDefault() bool {
	if r.gguf == nil {
		return false
	}
	v, ok := r.gguf.Metadata["tokenizer.chat_template"]
	if !ok {
		return false
	}
	s, ok := v.AsString()
	if !ok {
		return false
	}
	return strings.Contains(s, `'<|im_start|>assistant\n<think>\n'`) &&
		strings.Contains(s, `enable_thinking = enable_thinking if enable_thinking is defined else True`)
}

// renderQwen35Messages mirrors the text-only parts of Qwen3.5/3.6's embedded
// ChatML template. In addition to its thinking prompt it uses Qwen's native
// XML-like tool protocol; generic JSON tool blocks noticeably degrade tool
// calling because the model was trained on <function=...>/<parameter=...>.
// The optional tools argument keeps the simple two-argument form convenient
// for renderer tests and callers that do not expose tools.
func (r *Runner) renderQwen35Messages(messages []ChatMessage, systemPrompt string, toolSets ...[]ToolDefinition) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imEnd, ok2 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2) {
		return nil, false
	}
	var tools []ToolDefinition
	if len(toolSets) > 0 {
		tools = toolSets[0]
	}
	tokens := make([]uint32, 0)
	appendText := func(text string) {
		tokens = append(tokens, r.tok.EncodeWithoutBOS(text)...)
	}
	appendTurn := func(role, content string) {
		tokens = append(tokens, imStart)
		appendText(role + "\n" + strings.TrimSpace(content))
		tokens = append(tokens, imEnd)
		appendText("\n")
	}

	hasSystem := false
	var systemParts []string
	for _, message := range messages {
		if message.Role == ChatRoleSystem {
			hasSystem = true
			if content := strings.TrimSpace(message.Content); content != "" {
				systemParts = append(systemParts, content)
			}
		}
	}
	// The embedded Qwen template permits a system turn only at the beginning.
	// Normalize multiple API-level system messages into that one turn rather
	// than silently dropping a later instruction.
	explicitSystem := strings.Join(systemParts, "\n\n")
	if len(tools) > 0 {
		content := qwen35ToolSystemPrompt(tools)
		if hasSystem {
			content = appendSection(content, explicitSystem)
		} else if strings.TrimSpace(systemPrompt) != "" {
			content = appendSection(content, systemPrompt)
		}
		appendTurn("system", content)
	} else if hasSystem {
		appendTurn("system", explicitSystem)
	} else if strings.TrimSpace(systemPrompt) != "" {
		appendTurn("system", systemPrompt)
	}

	for i, message := range messages {
		if message.Role == ChatRoleSystem {
			// The system turn was rendered above, including the native tool list.
			continue
		}
		switch message.Role {
		case ChatRoleAssistant:
			content := strings.TrimSpace(message.Content)
			if len(message.ToolCalls) > 0 {
				content = renderQwen35AssistantToolCalls(content, message.ToolCalls)
			}
			appendTurn("assistant", content)
		case ChatRoleTool:
			// Qwen groups consecutive tool results into one user turn.
			if i == 0 || messages[i-1].Role != ChatRoleTool {
				tokens = append(tokens, imStart)
				appendText("user")
			}
			appendText("\n<tool_response>\n")
			appendText(strings.TrimSpace(message.Content))
			appendText("\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != ChatRoleTool {
				tokens = append(tokens, imEnd)
				appendText("\n")
			}
		default:
			appendTurn("user", message.Content)
		}
	}
	// add_generation_prompt=True with thinking enabled (the Qwen default).
	tokens = append(tokens, imStart)
	appendText("assistant\n<think>\n")
	return tokens, true
}

func qwen35ToolSystemPrompt(tools []ToolDefinition) string {
	var sb strings.Builder
	sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
	for _, tool := range tools {
		if encoded, err := json.Marshal(tool); err == nil {
			sb.WriteByte('\n')
			sb.Write(encoded)
		}
	}
	sb.WriteString("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n")
	// Keep the native text portion byte-for-byte aligned with Qwen3.5/3.6's
	// embedded chat template. These models are sensitive to the exact native
	// XML example and its explicit instruction block.
	sb.WriteString("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")
	return sb.String()
}

func renderQwen35AssistantToolCalls(content string, calls []ToolCall) string {
	var sb strings.Builder
	if content != "" {
		sb.WriteString(content)
	}
	for i, call := range calls {
		if i == 0 && sb.Len() > 0 {
			sb.WriteString("\n\n")
		} else if i > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString("<tool_call>\n<function=")
		sb.WriteString(call.Function.Name)
		sb.WriteString(">\n")
		var args map[string]any
		if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil {
			args = map[string]any{}
		}
		keys := make([]string, 0, len(args))
		for key := range args {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			sb.WriteString("<parameter=")
			sb.WriteString(key)
			sb.WriteString(">\n")
			if value, ok := args[key].(string); ok {
				sb.WriteString(value)
			} else if encoded, err := json.Marshal(args[key]); err == nil {
				sb.Write(encoded)
			}
			sb.WriteString("\n</parameter>\n")
		}
		sb.WriteString("</function>\n</tool_call>")
	}
	return sb.String()
}

// renderSoofiIsarMessages mirrors the text-only portion of the embedded
// Soofi-S-Isar Jinja template. In particular, it provides the model identity
// prompt and deliberately opens the assistant's <think> section; generic
// ChatML would omit both and produces a markedly weaker, non-reasoning turn.
func (r *Runner) renderSoofiIsarMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imEnd, ok2 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2) {
		return nil, false
	}
	const defaultSystem = "You are Soofi (Sovereign Open Source Foundation Models), an open-source AI assistant built for reasoning, developed by a German research consortium.\n\nArchitecture: Hybrid Mixture-of-Experts (MoE) with 23 Mamba-2/MoE layers and 6 Attention layers. 128 experts + 1 shared expert per MoE layer, 6 activated per token. 3.5B active parameters, 30B total.\n\nTraining: Trained from scratch on 25 trillion freely available tokens. Primary languages: English and German. Limited capability in French, Italian, and Spanish. English is the pivot language.\n\nBehaviour:\n- Answer identity questions naturally as Soofi.\n- For non-identity questions, respond normally and helpfully.\n- Match the language the user writes in.\n- Knowledge cutoff: 2025-12"
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, imStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n"+strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, imEnd)
			tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
		}
	}
	appendTurn("system", strings.TrimSpace(defaultSystem+"\n\n"+systemPrompt), true)
	for _, message := range messages {
		role := "user"
		if message.Role == ChatRoleAssistant {
			role = "assistant"
			// The Isar template keeps a closed empty thought section in history
			// when callers supplied only visible assistant content.
			if !strings.Contains(message.Content, "<think>") && !strings.Contains(message.Content, "</think>") {
				message.Content = "<think></think>" + message.Content
			}
		} else if message.Role == ChatRoleSystem {
			role = "system"
		}
		appendTurn(role, message.Content, true)
	}
	tokens = append(tokens, imStart)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant\n<think>\n")...)
	return tokens, true
}

// renderPhi4Messages mirrors Phi-4's embedded template:
//
//	<|im_start|>role<|im_sep|>content<|im_end|>
//
// It is deliberately not ChatML: ChatML places a newline after the role,
// whereas Phi-4 uses its dedicated <|im_sep|> control token. The final
// assistant turn remains open after that separator for generation.
func (r *Runner) renderPhi4Messages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imSep, ok2 := r.tok.SpecialID("<|im_sep|>")
	imEnd, ok3 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string) {
		tokens = append(tokens, imStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, imSep)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
		tokens = append(tokens, imEnd)
	}
	hasSystem := false
	for _, message := range messages {
		hasSystem = hasSystem || message.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt)
	}
	for _, message := range messages {
		role := "user"
		switch message.Role {
		case ChatRoleSystem:
			role = "system"
		case ChatRoleAssistant:
			role = "assistant"
		}
		appendTurn(role, message.Content)
	}
	tokens = append(tokens, imStart)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant")...)
	tokens = append(tokens, imSep)
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
	if r == nil || r.tok == nil {
		return ""
	}
	if r.gguf != nil {
		if v, ok := r.gguf.Metadata["tokenizer.chat_template"]; ok {
			if s, ok := v.AsString(); ok {
				switch {
				case strings.Contains(s, "<|turn>") && strings.Contains(s, "<turn|>"):
					return "gemma4-chat"
				case strings.Contains(s, "[INST]") && strings.Contains(s, "[/INST]"):
					return "mistral-inst"
				case r.arch == "llama" && strings.Contains(s, "<|python_tag|>") && strings.Contains(s, "<|eom_id|>") && strings.Contains(s, "tools_in_user_message"):
					// This is Meta's Llama-3.1-Instruct template, which has a
					// native custom-function / ipython exchange on top of the
					// otherwise shared Llama header protocol.
					return "llama31-chat"
				case strings.Contains(s, "<|start_header_id|>") && strings.Contains(s, "<|eot_id|>"):
					return "header-chat"
				case strings.Contains(s, "<|im_user|>") && strings.Contains(s, "<|im_assistant|>") && strings.Contains(s, "<|im_middle|>") && strings.Contains(s, "<|im_end|>"):
					return "kimi-chat"
				case r.arch == "phi3" && strings.Contains(s, "<|im_start|>") && strings.Contains(s, "<|im_sep|>") && strings.Contains(s, "<|im_end|>"):
					return "phi4-chat"
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
	}
	if _, ok := r.tok.SpecialID("<|im_user|>"); ok {
		if _, ok := r.tok.SpecialID("<|im_assistant|>"); ok {
			if _, ok := r.tok.SpecialID("<|im_middle|>"); ok {
				if _, ok := r.tok.SpecialID("<|im_end|>"); ok {
					return "kimi-chat"
				}
			}
		}
	}
	if _, ok := r.tok.SpecialID("[INST]"); ok {
		if _, ok := r.tok.SpecialID("[/INST]"); ok {
			return "mistral-inst"
		}
	}
	if _, ok := r.tok.SpecialID("<|turn>"); ok {
		if _, ok := r.tok.SpecialID("<turn|>"); ok {
			return "gemma4-chat"
		}
	}
	if r.arch == "phi3" {
		if _, ok := r.tok.SpecialID("<|im_start|>"); ok {
			if _, ok := r.tok.SpecialID("<|im_sep|>"); ok {
				if _, ok := r.tok.SpecialID("<|im_end|>"); ok {
					return "phi4-chat"
				}
			}
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
