package server

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type GenerateRequest struct {
	Prompt        string                     `json:"prompt"`
	Messages      []APIMessage               `json:"messages"`
	MaxTokens     *int                       `json:"max_tokens"`
	Temp          *float32                   `json:"temp"`
	Temperature   *float32                   `json:"temperature"`
	TopP          *float32                   `json:"top_p"`
	TopK          *int                       `json:"top_k"`
	MinP          *float32                   `json:"min_p"`
	RepeatPenalty *float32                   `json:"repeat_penalty"`
	Seed          *uint64                    `json:"seed"`
	SystemPrompt  *string                    `json:"system_prompt"`
	Stop          any                        `json:"stop"`
	Tools         []gopherllm.ToolDefinition `json:"tools"`
	ToolChoice    any                        `json:"tool_choice"`
	Wikimedia     bool                       `json:"gopherllm_wikimedia"`
	OpenStreetMap bool                       `json:"gopherllm_openstreetmap"`
	RAG           bool                       `json:"gopherllm_rag"`
}

func (g GenerateRequest) ToMessagesAndOptions(def gopherllm.GenerationOptions) ([]gopherllm.ChatMessage, gopherllm.GenerationOptions) {
	options := applyRequestOptions(def, g.MaxTokens, firstFloat(g.Temp, g.Temperature), g.TopP, g.TopK, g.MinP, g.RepeatPenalty, g.Seed, g.SystemPrompt, g.Stop, g.Tools, normalizeToolChoice(g.ToolChoice))
	if len(g.Messages) > 0 {
		return apiMessages(g.Messages), options
	}
	return []gopherllm.ChatMessage{gopherllm.UserMessage(g.Prompt)}, options
}

type APIMessage struct {
	Role       string               `json:"role"`
	Content    any                  `json:"content"`
	ToolCalls  []gopherllm.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
	Name       string               `json:"name,omitempty"`
}

func apiMessages(items []APIMessage) []gopherllm.ChatMessage {
	out := make([]gopherllm.ChatMessage, 0, len(items))
	for _, item := range items {
		role := gopherllm.ChatRoleUser
		switch strings.ToLower(item.Role) {
		case "system", "developer":
			// "developer" is the OpenAI o1/gpt-oss-era replacement for
			// "system"; GopherLLM has no separate developer-instruction
			// channel, so it renders the same as a system message.
			role = gopherllm.ChatRoleSystem
		case "assistant":
			role = gopherllm.ChatRoleAssistant
		case "tool", "function", "ipython":
			role = gopherllm.ChatRoleTool
		}
		out = append(out, gopherllm.ChatMessage{Role: role, Content: contentText(item.Content), Images: contentImages(item.Content), ToolCalls: item.ToolCalls, ToolCallID: item.ToolCallID, Name: item.Name})
	}
	return out
}

// contentImages extracts image content from an OpenAI-style multi-part
// content array's "image_url" parts (mirroring contentText's "text" parts).
// Only base64 data: URLs are decoded (data:image/png;base64,... — the
// common form for local/self-hosted API clients, requiring no network
// access). A plain http(s):// image URL is intentionally NOT fetched here:
// having the server fetch an arbitrary caller-supplied URL is a
// server-side-request-forgery-shaped feature that needs an explicit,
// off-by-default opt-in and its own review, not a default-on convenience;
// such a part is simply skipped rather than silently treated as unusable
// text; a future revision can add an explicit opt-in flag for it.
func contentImages(v any) []gopherllm.ImageContent {
	parts, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []gopherllm.ImageContent
	for _, p := range parts {
		m, ok := p.(map[string]any)
		if !ok || m["type"] != "image_url" {
			continue
		}
		urlField, ok := m["image_url"].(map[string]any)
		if !ok {
			continue
		}
		url, ok := urlField["url"].(string)
		if !ok {
			continue
		}
		const dataPrefix = "data:"
		if !strings.HasPrefix(url, dataPrefix) {
			continue
		}
		comma := strings.IndexByte(url, ',')
		if comma < 0 || !strings.Contains(url[:comma], ";base64") {
			continue
		}
		decoded, err := base64.StdEncoding.DecodeString(url[comma+1:])
		if err != nil {
			continue
		}
		out = append(out, gopherllm.ImageContent{Bytes: decoded})
	}
	return out
}

// alwaysContinue is passed to gopherllm.RunAgenticChat by non-streaming handlers, which
// only care about the returned gopherllm.GenerationResult, not incremental delivery.
func alwaysContinue(string) bool { return true }

// normalizeToolChoice extracts the OpenAI-compatible "tool_choice" value's
// meaning that this server actually acts on. A literal "none" suppresses tool
// offering; "auto"/"required" pass through unchanged (GopherLLM has no
// constrained decoding, so both just mean "offer the tools"); an object
// naming one function (`{"type":"function","function":{"name":"..."}}`)
// becomes "function:<name>", which gopherllm.GenerationOptions.ActiveTools narrows
// offering to. An object missing a usable name degrades to "" (== auto).
func normalizeToolChoice(raw any) string {
	switch v := raw.(type) {
	case string:
		return v
	case map[string]any:
		if v["type"] != "function" {
			return ""
		}
		fn, ok := v["function"].(map[string]any)
		if !ok {
			return ""
		}
		name, ok := fn["name"].(string)
		if !ok || name == "" {
			return ""
		}
		return "function:" + name
	default:
		return ""
	}
}

func contentText(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []any:
		parts := []string{}
		for _, p := range x {
			if m, ok := p.(map[string]any); ok {
				if m["type"] == "text" {
					if s, ok := m["text"].(string); ok {
						parts = append(parts, s)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

type OpenAIChatRequest struct {
	Model               string                     `json:"model"`
	Messages            []APIMessage               `json:"messages"`
	Stream              bool                       `json:"stream"`
	StreamOptions       *OpenAIStreamOpts          `json:"stream_options"`
	MaxTokens           *int                       `json:"max_tokens"`
	MaxCompletionTokens *int                       `json:"max_completion_tokens"`
	Temperature         *float32                   `json:"temperature"`
	TopP                *float32                   `json:"top_p"`
	TopK                *int                       `json:"top_k"`
	MinP                *float32                   `json:"min_p"`
	RepeatPenalty       *float32                   `json:"repeat_penalty"`
	Seed                *uint64                    `json:"seed"`
	SystemPrompt        *string                    `json:"system_prompt"`
	Stop                any                        `json:"stop"`
	Tools               []gopherllm.ToolDefinition `json:"tools"`
	ToolChoice          any                        `json:"tool_choice"`
	// GopherLLMContextMode is an opt-in extension for local clients. Omitting
	// it preserves normal OpenAI-compatible full-history semantics.
	GopherLLMContextMode string `json:"gopherllm_context_mode"`
	Wikimedia            bool   `json:"gopherllm_wikimedia"`
	OpenStreetMap        bool   `json:"gopherllm_openstreetmap"`
	// RAG requests the search_documents tool, offered only when
	// Features.RAG is enabled and the server's knowledge base is non-empty.
	RAG bool `json:"gopherllm_rag"`
	// Skills is a pointer so an absent field keeps the historical default
	// (skills offered whenever --skills-dir is configured) while a client that
	// wants them off can say so.
	Skills *bool `json:"gopherllm_skills"`
}

// SkillsEnabled reports whether this request wants the load_skill tool offered.
func (o OpenAIChatRequest) SkillsEnabled() bool { return o.Skills == nil || *o.Skills }

// OpenAIStreamOpts is the OpenAI "stream_options" object; IncludeUsage gates
// whether the final SSE chunk carries a "usage" field (off by default, per
// spec — unlike a non-streaming response, which always includes usage).
type OpenAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
}

func (o OpenAIChatRequest) Options(def gopherllm.GenerationOptions) gopherllm.GenerationOptions {
	maxTokens := o.MaxTokens
	if maxTokens == nil {
		maxTokens = o.MaxCompletionTokens
	}
	return applyRequestOptions(def, maxTokens, o.Temperature, o.TopP, o.TopK, o.MinP, o.RepeatPenalty, o.Seed, o.SystemPrompt, o.Stop, o.Tools, normalizeToolChoice(o.ToolChoice))
}

// gopherllm.ContextWindowMode parses GopherLLM's local context-window extension. It is
// deliberately separate from Options so existing callers that use Options
// directly retain the zero-value (full history) behavior.
func (o OpenAIChatRequest) ContextWindowMode() (gopherllm.ContextWindowMode, error) {
	mode := gopherllm.ContextWindowMode(strings.ToLower(strings.TrimSpace(o.GopherLLMContextMode)))
	if mode == "" || mode == gopherllm.ContextWindowFull {
		return gopherllm.ContextWindowFull, nil
	}
	if mode == gopherllm.ContextWindowRecent {
		return gopherllm.ContextWindowRecent, nil
	}
	if mode == "autocompress" {
		return gopherllm.ContextWindowAutoCompress, nil
	}
	return "", fmt.Errorf("gopherllm_context_mode must be full, recent, or autoCompress")
}

func (o OpenAIChatRequest) ChatMessages() []gopherllm.ChatMessage { return apiMessages(o.Messages) }

type OpenAICompletionRequest struct {
	Model               string   `json:"model"`
	Prompt              any      `json:"prompt"`
	MaxTokens           *int     `json:"max_tokens"`
	MaxCompletionTokens *int     `json:"max_completion_tokens"`
	Temperature         *float32 `json:"temperature"`
	TopP                *float32 `json:"top_p"`
	TopK                *int     `json:"top_k"`
	MinP                *float32 `json:"min_p"`
	RepeatPenalty       *float32 `json:"repeat_penalty"`
	Seed                *uint64  `json:"seed"`
	SystemPrompt        *string  `json:"system_prompt"`
	Stop                any      `json:"stop"`
}

func (o OpenAICompletionRequest) PromptString() string {
	switch p := o.Prompt.(type) {
	case string:
		return p
	case []any:
		if len(p) > 0 {
			if s, ok := p[0].(string); ok {
				return s
			}
		}
	}
	return ""
}

func (o OpenAICompletionRequest) Options(def gopherllm.GenerationOptions) gopherllm.GenerationOptions {
	maxTokens := o.MaxTokens
	if maxTokens == nil {
		maxTokens = o.MaxCompletionTokens
	}
	return applyRequestOptions(def, maxTokens, o.Temperature, o.TopP, o.TopK, o.MinP, o.RepeatPenalty, o.Seed, o.SystemPrompt, o.Stop, nil, "")
}

type EmbeddingsRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"`
}

func (e EmbeddingsRequest) Inputs() []string {
	switch x := e.Input.(type) {
	case string:
		return []string{x}
	case []any:
		out := []string{}
		for _, v := range x {
			if s, ok := v.(string); ok {
				out = append(out, s)
			}
		}
		return out
	case nil:
		return nil
	default:
		return []string{fmt.Sprint(x)}
	}
}

// embedTexts is the shared compatibility-layer implementation for OpenAI,
// Ollama, and the browser RAG endpoint. In particular, it makes their empty
// input behaviour consistent instead of silently returning an empty success
// response from one endpoint and rejecting the same request from another.
func embedTexts(r *gopherllm.Runner, inputs []string) ([][]float32, int, error) {
	if r == nil {
		return nil, 0, errors.New("no embedding model is loaded")
	}
	if len(inputs) == 0 {
		return nil, 0, errors.New("embedding input must contain at least one text")
	}
	vectors := make([][]float32, 0, len(inputs))
	tokens := 0
	for _, input := range inputs {
		if strings.TrimSpace(input) == "" {
			return nil, 0, errors.New("embedding input must not be empty")
		}
		result, err := r.Embed(input)
		if err != nil {
			return nil, 0, err
		}
		vectors = append(vectors, result.Embedding)
		tokens += result.TokenCount
	}
	return vectors, tokens, nil
}
