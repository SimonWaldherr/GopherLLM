package server

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type OllamaGenerateRequest struct {
	Model         string        `json:"model"`
	Prompt        string        `json:"prompt"`
	System        string        `json:"system"`
	Stream        *bool         `json:"stream"`
	Options       OllamaOptions `json:"options"`
	Stop          any           `json:"stop"`
	Wikimedia     bool          `json:"gopherllm_wikimedia"`
	OpenStreetMap bool          `json:"gopherllm_openstreetmap"`
	RAG           bool          `json:"gopherllm_rag"`
}

func (o OllamaGenerateRequest) GenerationOptions(def gopherllm.GenerationOptions) gopherllm.GenerationOptions {
	system := (*string)(nil)
	if o.System != "" {
		system = &o.System
	}
	return applyRequestOptions(def, o.Options.NumPredict, o.Options.Temperature, o.Options.TopP, o.Options.TopK, o.Options.MinP, o.Options.RepeatPenalty, o.Options.Seed, system, firstStop(o.Stop, o.Options.Stop), nil, "")
}

type OllamaChatRequest struct {
	Model         string                     `json:"model"`
	Messages      []OllamaMessage            `json:"messages"`
	Stream        *bool                      `json:"stream"`
	Options       OllamaOptions              `json:"options"`
	Tools         []gopherllm.ToolDefinition `json:"tools"`
	Wikimedia     bool                       `json:"gopherllm_wikimedia"`
	OpenStreetMap bool                       `json:"gopherllm_openstreetmap"`
	RAG           bool                       `json:"gopherllm_rag"`
}

// streamEnabled implements Ollama's default-true streaming semantics: the
// request only turns streaming off when the "stream" field is explicitly
// present and false; omitting it (nil) streams, matching real Ollama.
func streamEnabled(b *bool) bool {
	return b == nil || *b
}

func (o OllamaChatRequest) GenerationOptions(def gopherllm.GenerationOptions) gopherllm.GenerationOptions {
	return applyRequestOptions(def, o.Options.NumPredict, o.Options.Temperature, o.Options.TopP, o.Options.TopK, o.Options.MinP, o.Options.RepeatPenalty, o.Options.Seed, nil, o.Options.Stop, o.Tools, "")
}

func (o OllamaChatRequest) ChatMessages() []gopherllm.ChatMessage {
	items := make([]APIMessage, len(o.Messages))
	for i, message := range o.Messages {
		items[i] = APIMessage{Role: message.Role, Content: message.Content}
	}
	return apiMessages(items)
}

type OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaOptions struct {
	// NumCtx is accepted for wire compatibility (real Ollama clients set it
	// routinely) but not actionable here: a gopherllm.Runner's KV cache is sized once
	// from the loaded GGUF's context_length at model-load time, and this
	// server has no per-request context-window resize.
	NumCtx        *int     `json:"num_ctx"`
	NumPredict    *int     `json:"num_predict"`
	Temperature   *float32 `json:"temperature"`
	TopP          *float32 `json:"top_p"`
	TopK          *int     `json:"top_k"`
	MinP          *float32 `json:"min_p"`
	RepeatPenalty *float32 `json:"repeat_penalty"`
	Seed          *uint64  `json:"seed"`
	Stop          any      `json:"stop"`
}

type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Input  any    `json:"input"`
}

func (o OllamaEmbeddingRequest) Inputs() []string {
	return EmbeddingsRequest{Input: o.Input}.Inputs()
}

// OllamaEmbedRequest is the request body for /api/embed, the batched
// successor to the deprecated single-prompt /api/embeddings.
type OllamaEmbedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"`
}

func (o OllamaEmbedRequest) Inputs() []string {
	return EmbeddingsRequest{Input: o.Input}.Inputs()
}
