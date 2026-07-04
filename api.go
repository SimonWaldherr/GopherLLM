package gopherllm

import (
	"context"
	"fmt"
	"io"
	"strings"
)

// This file is the primary embedding API for Go applications: a context-first,
// functional-options veneer over the lower-level Runner. Typical use:
//
//	model, err := gopherllm.Open(ctx, "model.gguf")
//	defer model.Close()
//	res, err := model.Generate(ctx, "Why is the sky blue?",
//	    gopherllm.WithMaxTokens(256), gopherllm.WithTemperature(0.7))
//	fmt.Println(res.Text)
//
// The lower-level Runner and its Generate* methods remain available for
// callers that need them, but new code should prefer Model.

// Model is a loaded LLM ready for generation, chat, embeddings, and
// tokenization. It wraps a Runner; the same concurrency contract applies
// (safe to share across goroutines, requests execute one at a time).
type Model struct {
	r    *Runner
	info LoadInfo
}

// Option configures model loading.
type Option func(*loadSettings)

type loadSettings struct {
	logw    io.Writer
	threads int
}

// WithLogWriter directs load-progress and warning diagnostics (GGUF summary,
// per-layer load progress, experimental-architecture warnings) to w. The
// default is io.Discard: as a library, GopherLLM produces no output unless
// asked to.
func WithLogWriter(w io.Writer) Option {
	return func(s *loadSettings) {
		if w != nil {
			s.logw = w
		}
	}
}

// WithThreads sets the compute worker count. NOTE: the worker pool is shared
// process-wide (all Models in the process use one pool), so this is a global
// knob despite being a load option; the last value set wins.
func WithThreads(n int) Option {
	return func(s *loadSettings) { s.threads = n }
}

// Open memory-maps and loads a GGUF model file. The returned Model borrows
// quantized weights from the mapping zero-copy; Close releases it. ctx only
// gates the start of the load (weight loading itself is not interruptible).
func Open(ctx context.Context, path string, opts ...Option) (*Model, error) {
	settings := applyLoadOptions(opts)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r, info, err := runnerFromPath(path, settings.logw)
	if err != nil {
		return nil, err
	}
	return &Model{r: r, info: info}, nil
}

// OpenBytes loads a model from an in-memory GGUF image, copying quantized
// tensors into owned memory (data may be released afterwards).
func OpenBytes(ctx context.Context, data []byte, opts ...Option) (*Model, error) {
	settings := applyLoadOptions(opts)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r, err := runnerFromGGUFBytes(data, false, settings.logw)
	if err != nil {
		return nil, err
	}
	return &Model{r: r, info: LoadInfo{FileSizeBytes: len(data)}}, nil
}

func applyLoadOptions(opts []Option) loadSettings {
	settings := loadSettings{logw: io.Discard}
	for _, opt := range opts {
		opt(&settings)
	}
	if settings.threads > 0 {
		SetNumThreads(settings.threads)
	}
	return settings
}

// Close releases the model's memory-mapped weight file. No Model method may
// be called afterwards.
func (m *Model) Close() error { return m.r.Close() }

// Runner exposes the underlying low-level Runner for callers that need
// capabilities the Model API doesn't surface (custom forward passes, the
// agentic skill loop, the HTTP handler).
func (m *Model) Runner() *Runner { return m.r }

// Info returns file size and load timing. Config, Tokenizer, GGUF, and Name
// expose the model's static properties.
func (m *Model) Info() LoadInfo        { return m.info }
func (m *Model) Config() Config        { return m.r.Config() }
func (m *Model) Tokenizer() *Tokenizer { return m.r.Tokenizer() }
func (m *Model) GGUF() *GGUFFile       { return m.r.GGUF() }

// Name returns the model's self-declared name (general.name), or the empty
// string.
func (m *Model) Name() string {
	name, _ := m.r.ModelName()
	return name
}

// GenOption configures a single generation request on top of
// DefaultGenerationOptions.
type GenOption func(*GenerationOptions)

// WithMaxTokens caps the number of generated tokens (default 256).
func WithMaxTokens(n int) GenOption { return func(o *GenerationOptions) { o.MaxTokens = n } }

// WithTemperature sets the sampling temperature; 0 selects greedy decoding.
func WithTemperature(t float32) GenOption {
	return func(o *GenerationOptions) { o.Sampler.Temperature = t }
}

// WithTopP sets nucleus sampling mass in (0, 1]; 1 disables it.
func WithTopP(p float32) GenOption { return func(o *GenerationOptions) { o.Sampler.TopP = p } }

// WithTopK restricts sampling to the k most likely tokens; 0 disables it.
func WithTopK(k int) GenOption { return func(o *GenerationOptions) { o.Sampler.TopK = k } }

// WithMinP drops candidates below fraction p of the best token's probability.
func WithMinP(p float32) GenOption { return func(o *GenerationOptions) { o.Sampler.MinP = p } }

// WithRepeatPenalty penalizes recently generated tokens; 1 disables it.
func WithRepeatPenalty(p float32) GenOption {
	return func(o *GenerationOptions) { o.Sampler.RepeatPenalty = p }
}

// WithSeed makes sampling deterministic for a fixed seed (0 = time-based).
func WithSeed(seed uint64) GenOption { return func(o *GenerationOptions) { o.Seed = seed } }

// WithSystemPrompt replaces the default system prompt; pass "" for none.
func WithSystemPrompt(s string) GenOption {
	return func(o *GenerationOptions) { o.SystemPrompt = s }
}

// WithStop adds stop sequences that end generation when they appear.
func WithStop(sequences ...string) GenOption {
	return func(o *GenerationOptions) { o.StopSequences = append(o.StopSequences, sequences...) }
}

// WithTools offers OpenAI-shaped tool definitions to the model; calls the
// model makes come back in Result.ToolCalls with FinishReason "tool_calls".
func WithTools(tools ...ToolDefinition) GenOption {
	return func(o *GenerationOptions) { o.Tools = append(o.Tools, tools...) }
}

// WithToolChoice sets the tool_choice behavior ("none" suppresses tools).
func WithToolChoice(choice string) GenOption {
	return func(o *GenerationOptions) { o.ToolChoice = choice }
}

// WithGenerationOptions replaces the entire options struct (escape hatch for
// callers that already hold a GenerationOptions); later GenOptions still
// apply on top.
func WithGenerationOptions(base GenerationOptions) GenOption {
	return func(o *GenerationOptions) {
		ctx := o.ctx
		*o = base
		o.ctx = ctx
	}
}

func buildGenOptions(ctx context.Context, opts []GenOption) GenerationOptions {
	options := DefaultGenerationOptions()
	options.ctx = ctx
	for _, opt := range opts {
		opt(&options)
	}
	return options
}

// Generate runs a single-prompt completion. Cancellation via ctx takes effect
// between prefill chunks and between decoded tokens; the returned error is
// then the context's error.
func (m *Model) Generate(ctx context.Context, prompt string, opts ...GenOption) (GenerationResult, error) {
	return m.Chat(ctx, []ChatMessage{UserMessage(prompt)}, opts...)
}

// Chat runs a multi-turn conversation through the model's chat template.
func (m *Model) Chat(ctx context.Context, messages []ChatMessage, opts ...GenOption) (GenerationResult, error) {
	return m.r.GenerateChatStreamUntil(messages, buildGenOptions(ctx, opts), func(string) bool { return true })
}

// Stream is Chat with incremental delivery: onDelta receives each new chunk
// of valid-UTF-8 output text as it is generated. Returning a non-nil error
// from onDelta stops generation; that error is returned (wrapped) alongside
// the partial result. When tools are offered, output is delivered in a single
// final call instead of incrementally so raw tool-call syntax never leaks
// (see RunAgenticChat for why).
func (m *Model) Stream(ctx context.Context, messages []ChatMessage, onDelta func(delta string) error, opts ...GenOption) (GenerationResult, error) {
	var deltaErr error
	result, err := m.r.GenerateChatStreamUntil(messages, buildGenOptions(ctx, opts), func(text string) bool {
		if onDelta == nil {
			return true
		}
		if err := onDelta(text); err != nil {
			deltaErr = err
			return false
		}
		return true
	})
	if deltaErr != nil {
		return result, fmt.Errorf("stream callback: %w", deltaErr)
	}
	return result, err
}

// Embed returns the mean-pooled, L2-normalized final-hidden-state embedding
// of text. Cancellation granularity is the whole call (embedding runs a
// prompt-length forward pass).
func (m *Model) Embed(ctx context.Context, text string) (EmbeddingResult, error) {
	if err := ctx.Err(); err != nil {
		return EmbeddingResult{}, err
	}
	return m.r.Embed(text)
}

// Tokenize encodes text with the model's tokenizer (including BOS when the
// model declares it). Detokenize is its inverse over generated ids.
func (m *Model) Tokenize(text string) []uint32 { return m.r.Tokenizer().Encode(text) }

// Detokenize decodes token ids back to text.
func (m *Model) Detokenize(ids []uint32) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(m.r.Tokenizer().DecodeToken(id))
	}
	return sb.String()
}