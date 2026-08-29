package gopherllm

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"
)

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
	// Images attaches image content to a user message for a model loaded
	// with a vision projector (Runner.HasVision). Capped at one image per
	// message for now — see ImageContent.
	Images []ImageContent
	// ToolCalls is set on an assistant message that is replaying a prior turn
	// in which the model requested one or more tool calls.
	ToolCalls []ToolCall
	// ToolCallID and Name identify which prior tool call a ChatRoleTool
	// message is answering.
	ToolCallID string
	Name       string
}

// ImageContent is one encoded image (PNG/JPEG/anything image.Decode
// understands) attached to a ChatMessage. Decoding and vision-tower encoding
// happen lazily during rendering, not when the message is constructed.
type ImageContent struct {
	Bytes []byte
}

func UserMessage(content string) ChatMessage {
	return ChatMessage{Role: ChatRoleUser, Content: content}
}

// UserMessageWithImages is UserMessage plus one or more attached images.
// Only the first image is used today — a message carrying more than one is
// rejected at render time with a clear error rather than silently dropping
// the extras.
func UserMessageWithImages(content string, images ...ImageContent) ChatMessage {
	return ChatMessage{Role: ChatRoleUser, Content: content, Images: images}
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
	// MTPDraftTokens enables Qwen3.5/3.6/3.8's on-model NextN draft head for
	// exact greedy speculation. Zero (the default) leaves MTP inactive in the
	// decode path. The target remains the authority for every emitted
	// token; non-greedy sampling is rejected rather than silently changing its
	// distribution.
	MTPDraftTokens int
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
	// MaxToolRounds bounds how many times the agent loop will feed tool
	// results back to the model before forcing a final, tools-withdrawn pass.
	// Zero means DefaultToolRounds; values are clamped into
	// [1, MaxToolRoundsCeiling]. Zero-value behavior is byte-identical to the
	// loop's historical fixed budget, so no existing caller moves.
	MaxToolRounds int
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
	if o.MTPDraftTokens < 0 || o.MTPDraftTokens > 32 {
		return fmt.Errorf("mtp_draft_tokens must be in the range [0, 32]")
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
