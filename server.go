package gopherllm

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

//go:embed web_ui/chat.html
var chatHTMLTmpl string

//go:embed web_ui/style.css
var chatCSS string

//go:embed web_ui/script.js
var chatJS string

var chatTemplate = template.Must(template.New("chat").Parse(chatHTMLTmpl))

var inferenceRequestSeq atomic.Uint64

type chatTemplateData struct {
	Title       string
	Model       string
	MaxTokens   int
	Temperature float32
}

type runnerState struct {
	mu   sync.RWMutex
	r    *Runner
	path string
}

func (s *runnerState) get() *Runner {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.r
}

func (s *runnerState) withRunner(fn func(*Runner)) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	fn(s.r)
}

func (s *runnerState) swap(r *Runner, path string) {
	var old *Runner
	s.mu.Lock()
	old = s.r
	s.r = r
	s.path = path
	s.mu.Unlock()
	if old != nil {
		_ = old.Close()
	}
}

func (s *runnerState) getPath() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.path
}

// HandlerOptions configures the mountable HTTP API handler.
type HandlerOptions struct {
	// Defaults are the generation settings requests inherit unless they
	// override individual fields.
	Defaults GenerationOptions
	// MaxConcurrentRequests bounds in-flight generation requests (default 8).
	// Requests beyond the bound queue rather than failing.
	MaxConcurrentRequests int
	// ChatUI serves the embedded browser chat at /chat (plus its assets).
	ChatUI bool
	// ModelDir enables GET /models discovery and POST /models/load hot-swap
	// within that directory.
	ModelDir string
	// ModelPath is the initially loaded model's path (reported by /models).
	ModelPath string
	// SkillsDir, if set, is scanned once at handler construction for SKILL.md
	// files (see skills.go). Every chat/generate endpoint offers a load_skill
	// tool and resolves it server-side via RunAgenticChat.
	SkillsDir string
	// LogWriter receives handler diagnostics (skill load notes). Defaults to
	// io.Discard.
	LogWriter io.Writer
}

// ServeOptions is HandlerOptions plus the listen address, for the Serve
// convenience wrapper (used by the CLI). ChatHistoryPath/ChatHistoryLock are
// retained for compatibility but unused.
type ServeOptions struct {
	Addr                     string
	Defaults                 GenerationOptions
	MaxConcurrentConnections int
	ChatUI                   bool
	ChatHistoryPath          string
	ChatHistoryLock          *sync.Mutex
	ModelDir                 string
	ModelPath                string
	SkillsDir                string
	// LogWriter receives startup and handler diagnostics; Serve defaults it
	// to os.Stderr (CLI behavior), unlike NewHandler's io.Discard.
	LogWriter io.Writer
}

// Serve builds the API handler and runs a blocking http.Server on opts.Addr.
// Library consumers who want to control the server lifecycle, add middleware,
// TLS, or mount the API under a path prefix should use NewHandler instead:
//
//	handler := gopherllm.NewHandler(model.Runner(), gopherllm.HandlerOptions{...})
//	mux.Handle("/llm/", http.StripPrefix("/llm", handler))
func Serve(initialRunner *Runner, opts ServeOptions) error {
	logw := opts.LogWriter
	if logw == nil {
		logw = os.Stderr
	}
	handler := NewHandler(initialRunner, HandlerOptions{
		Defaults:              opts.Defaults,
		MaxConcurrentRequests: opts.MaxConcurrentConnections,
		ChatUI:                opts.ChatUI,
		ModelDir:              opts.ModelDir,
		ModelPath:             opts.ModelPath,
		SkillsDir:             opts.SkillsDir,
		LogWriter:             logw,
	})
	server := &http.Server{Addr: opts.Addr, Handler: handler, ReadHeaderTimeout: 30 * time.Second}
	fmt.Fprintf(logw, "Serving on %s\n", displayServerURL(opts.Addr, opts.ChatUI))
	return server.ListenAndServe()
}

// NewHandler returns the complete GopherLLM HTTP API (OpenAI-compatible,
// Ollama-compatible, and native endpoints — see the README's endpoint table)
// as a mountable http.Handler. It owns no listener and writes nothing except
// to opts.LogWriter, so it composes with any router, middleware stack, or
// server the host application already has.
func NewHandler(initialRunner *Runner, opts HandlerOptions) http.Handler {
	logw := opts.LogWriter
	if logw == nil {
		logw = io.Discard
	}
	if opts.MaxConcurrentRequests <= 0 {
		opts.MaxConcurrentRequests = 8
	}
	skills, err := LoadSkills(opts.SkillsDir)
	if err != nil {
		fmt.Fprintf(logw, "Warning: skills: %v (continuing without skills)\n", err)
	} else if len(skills) > 0 {
		names := make([]string, len(skills))
		for i, s := range skills {
			names[i] = s.Name
		}
		fmt.Fprintf(logw, "Skills: loaded %d (%s)\n", len(skills), strings.Join(names, ", "))
	}
	state := &runnerState{r: initialRunner, path: opts.ModelPath}
	sem := make(chan struct{}, opts.MaxConcurrentRequests)
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"ok": true, "model": modelID(state.get())})
	})
	mux.HandleFunc("/generate", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		requestID := ensureRequestID(w, req)
		var body GenerateRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		messages, options := body.ToMessagesAndOptions(opts.Defaults)
		options = withRequestContext(options, req)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			result, err := RunAgenticChat(r, messages, options, skills, alwaysContinue)
			logInferenceResult(logw, requestID, "/generate", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, generateResponse(result))
		})
	}))
	mux.HandleFunc("/v1/chat/completions", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		requestID := ensureRequestID(w, req)
		var body OpenAIChatRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.Options(opts.Defaults)
		options = withRequestContext(options, req)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			if body.Stream {
				streamOpenAIChat(w, req, logw, requestID, r, model, body.ChatMessages(), options, skills)
				return
			}
			result, err := RunAgenticChat(r, body.ChatMessages(), options, skills, alwaysContinue)
			logInferenceResult(logw, requestID, "/v1/chat/completions", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, openAIChatResponse(model, result))
		})
	}))
	mux.HandleFunc("/v1/completions", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		requestID := ensureRequestID(w, req)
		var body OpenAICompletionRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.Options(opts.Defaults)
		options = withRequestContext(options, req)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			result, err := r.Generate(body.PromptString(), options)
			logInferenceResult(logw, requestID, "/v1/completions", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, map[string]any{"id": "cmpl-gopherllm", "object": "text_completion", "model": model, "choices": []any{map[string]any{"index": 0, "text": result.Text, "finish_reason": "stop"}}, "usage": usage(result)})
		})
	}))
	mux.HandleFunc("/v1/embeddings", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body EmbeddingsRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		inputs := body.Inputs()
		data := []any{}
		total := 0
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			for i, input := range inputs {
				emb, err := r.Embed(input)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				total += emb.TokenCount
				data = append(data, map[string]any{"object": "embedding", "index": i, "embedding": emb.Embedding})
			}
			writeJSON(w, map[string]any{"object": "list", "model": model, "data": data, "usage": map[string]int{"prompt_tokens": total, "total_tokens": total}})
		})
	}))
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		model := modelID(state.get())
		writeJSON(w, map[string]any{"object": "list", "data": []any{map[string]any{"id": model, "object": "model", "created": 0, "owned_by": "gopherllm"}}})
	})
	mux.HandleFunc("/v1/skills", func(w http.ResponseWriter, _ *http.Request) {
		// Only name/description are exposed here, matching the progressive
		// disclosure the load_skill tool itself uses: full bodies are loaded
		// on demand by the model, not dumped up front.
		list := make([]map[string]string, len(skills))
		for i, s := range skills {
			list[i] = map[string]string{"name": s.Name, "description": s.Description}
		}
		writeJSON(w, map[string]any{"skills": list})
	})
	mux.HandleFunc("/api/generate", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		requestID := ensureRequestID(w, req)
		var body OllamaGenerateRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.GenerationOptions(opts.Defaults)
		options = withRequestContext(options, req)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			result, err := RunAgenticChat(r, []ChatMessage{UserMessage(body.Prompt)}, options, skills, alwaysContinue)
			logInferenceResult(logw, requestID, "/api/generate", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "response": result.Text, "done": true, "prompt_eval_count": result.Stats.PromptTokens, "eval_count": result.Stats.GeneratedTokens})
		})
	}))
	mux.HandleFunc("/api/chat", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		requestID := ensureRequestID(w, req)
		var body OllamaChatRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.GenerationOptions(opts.Defaults)
		options = withRequestContext(options, req)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			result, err := RunAgenticChat(r, body.ChatMessages(), options, skills, alwaysContinue)
			logInferenceResult(logw, requestID, "/api/chat", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "message": map[string]any{"role": "assistant", "content": result.Text}, "done": true, "prompt_eval_count": result.Stats.PromptTokens, "eval_count": result.Stats.GeneratedTokens})
		})
	}))
	mux.HandleFunc("/api/embeddings", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body OllamaEmbeddingRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		text := body.Prompt
		if text == "" {
			inputs := body.Inputs()
			if len(inputs) > 0 {
				text = inputs[0]
			}
		}
		state.withRunner(func(r *Runner) {
			emb, err := r.Embed(text)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, map[string]any{"embedding": emb.Embedding})
		})
	}))
	mux.HandleFunc("/models", func(w http.ResponseWriter, _ *http.Request) {
		type modelInfo struct {
			ID           string  `json:"id"`
			Name         string  `json:"name"`
			Path         string  `json:"path"`
			Architecture string  `json:"architecture"`
			SizeGB       float64 `json:"size_gb"`
			Supported    bool    `json:"supported"`
			Loaded       bool    `json:"loaded"`
		}
		if opts.ModelDir == "" {
			writeJSON(w, map[string]any{"models": []modelInfo{}})
			return
		}
		entries, err := DiscoverModels(opts.ModelDir, io.Discard)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		loadedPath := state.getPath()
		models := make([]modelInfo, 0, len(entries))
		for _, e := range entries {
			if e.IsProjector {
				continue
			}
			name := e.ModelName
			if name == "" {
				name = e.FileName
			}
			models = append(models, modelInfo{
				ID:           e.ID,
				Name:         name,
				Path:         e.Path,
				Architecture: e.Architecture,
				SizeGB:       float64(e.SizeBytes) / (1024 * 1024 * 1024),
				Supported:    e.IsSupported,
				Loaded:       e.Path == loadedPath,
			})
		}
		writeJSON(w, map[string]any{"models": models})
	})
	mux.HandleFunc("/models/load", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Path string `json:"path"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if body.Path == "" {
			http.Error(w, "missing path", http.StatusBadRequest)
			return
		}
		newRunner, _, err := RunnerFromPath(body.Path)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		state.swap(newRunner, body.Path)
		writeJSON(w, map[string]any{"ok": true, "model": modelID(newRunner)})
	}))
	if opts.ChatUI {
		mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
			if req.URL.Path != "/" {
				http.NotFound(w, req)
				return
			}
			http.Redirect(w, req, "/chat", http.StatusFound)
		})
		mux.HandleFunc("/chat", func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("content-type", "text/html; charset=utf-8")
			data := chatTemplateData{
				Title:       "GopherLLM Chat",
				Model:       modelID(state.get()),
				MaxTokens:   opts.Defaults.MaxTokens,
				Temperature: opts.Defaults.Sampler.Temperature,
			}
			if err := chatTemplate.Execute(w, data); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		})
		mux.HandleFunc("/style.css", func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("content-type", "text/css; charset=utf-8")
			fmt.Fprint(w, chatCSS)
		})
		mux.HandleFunc("/script.js", func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("content-type", "text/javascript; charset=utf-8")
			fmt.Fprint(w, chatJS)
		})
	}
	return mux
}

func displayServerURL(addr string, chatUI bool) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		host = strings.Trim(addr, "[]")
		port = ""
	}
	if host == "" || host == "::" || host == "0.0.0.0" || host == "[::]" {
		host = "localhost"
	}
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") {
		host = "[" + host + "]"
	}
	url := "http://" + host
	if port != "" {
		url += ":" + port
	}
	if chatUI {
		url += "/chat"
	}
	return url
}

type GenerateRequest struct {
	Prompt        string           `json:"prompt"`
	Messages      []APIMessage     `json:"messages"`
	MaxTokens     *int             `json:"max_tokens"`
	Temp          *float32         `json:"temp"`
	Temperature   *float32         `json:"temperature"`
	TopP          *float32         `json:"top_p"`
	TopK          *int             `json:"top_k"`
	MinP          *float32         `json:"min_p"`
	RepeatPenalty *float32         `json:"repeat_penalty"`
	Seed          *uint64          `json:"seed"`
	SystemPrompt  *string          `json:"system_prompt"`
	Stop          any              `json:"stop"`
	Tools         []ToolDefinition `json:"tools"`
	ToolChoice    any              `json:"tool_choice"`
}

func (g GenerateRequest) ToMessagesAndOptions(def GenerationOptions) ([]ChatMessage, GenerationOptions) {
	options := applyRequestOptions(def, g.MaxTokens, firstFloat(g.Temp, g.Temperature), g.TopP, g.TopK, g.MinP, g.RepeatPenalty, g.Seed, g.SystemPrompt, g.Stop, g.Tools, normalizeToolChoice(g.ToolChoice))
	if len(g.Messages) > 0 {
		return apiMessages(g.Messages), options
	}
	return []ChatMessage{UserMessage(g.Prompt)}, options
}

type APIMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

func apiMessages(items []APIMessage) []ChatMessage {
	out := make([]ChatMessage, 0, len(items))
	for _, item := range items {
		role := ChatRoleUser
		switch strings.ToLower(item.Role) {
		case "system":
			role = ChatRoleSystem
		case "assistant":
			role = ChatRoleAssistant
		case "tool", "function", "ipython":
			role = ChatRoleTool
		}
		out = append(out, ChatMessage{Role: role, Content: contentText(item.Content), ToolCalls: item.ToolCalls, ToolCallID: item.ToolCallID, Name: item.Name})
	}
	return out
}

// alwaysContinue is passed to RunAgenticChat by non-streaming handlers, which
// only care about the returned GenerationResult, not incremental delivery.
func alwaysContinue(string) bool { return true }

// normalizeToolChoice extracts the OpenAI-compatible "tool_choice" value's
// meaning that this server actually acts on. A literal "none" suppresses tool
// offering; every other form (the default "auto", "required", or an object
// forcing a specific tool) is treated as "offer the tools" since GopherLLM
// has no constrained decoding to force a specific one.
func normalizeToolChoice(raw any) string {
	if s, ok := raw.(string); ok {
		return s
	}
	return ""
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
	Model               string           `json:"model"`
	Messages            []APIMessage     `json:"messages"`
	Stream              bool             `json:"stream"`
	MaxTokens           *int             `json:"max_tokens"`
	MaxCompletionTokens *int             `json:"max_completion_tokens"`
	Temperature         *float32         `json:"temperature"`
	TopP                *float32         `json:"top_p"`
	TopK                *int             `json:"top_k"`
	MinP                *float32         `json:"min_p"`
	RepeatPenalty       *float32         `json:"repeat_penalty"`
	Seed                *uint64          `json:"seed"`
	SystemPrompt        *string          `json:"system_prompt"`
	Stop                any              `json:"stop"`
	Tools               []ToolDefinition `json:"tools"`
	ToolChoice          any              `json:"tool_choice"`
}

func (o OpenAIChatRequest) Options(def GenerationOptions) GenerationOptions {
	maxTokens := o.MaxTokens
	if maxTokens == nil {
		maxTokens = o.MaxCompletionTokens
	}
	return applyRequestOptions(def, maxTokens, o.Temperature, o.TopP, o.TopK, o.MinP, o.RepeatPenalty, o.Seed, o.SystemPrompt, o.Stop, o.Tools, normalizeToolChoice(o.ToolChoice))
}

func (o OpenAIChatRequest) ChatMessages() []ChatMessage { return apiMessages(o.Messages) }

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

func (o OpenAICompletionRequest) Options(def GenerationOptions) GenerationOptions {
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
	default:
		return []string{fmt.Sprint(x)}
	}
}

type OllamaGenerateRequest struct {
	Model   string        `json:"model"`
	Prompt  string        `json:"prompt"`
	System  string        `json:"system"`
	Stream  bool          `json:"stream"`
	Options OllamaOptions `json:"options"`
	Stop    any           `json:"stop"`
}

func (o OllamaGenerateRequest) GenerationOptions(def GenerationOptions) GenerationOptions {
	system := (*string)(nil)
	if o.System != "" {
		system = &o.System
	}
	return applyRequestOptions(def, o.Options.NumPredict, o.Options.Temperature, o.Options.TopP, o.Options.TopK, o.Options.MinP, o.Options.RepeatPenalty, o.Options.Seed, system, firstStop(o.Stop, o.Options.Stop), nil, "")
}

type OllamaChatRequest struct {
	Model    string           `json:"model"`
	Messages []OllamaMessage  `json:"messages"`
	Stream   bool             `json:"stream"`
	Options  OllamaOptions    `json:"options"`
	Tools    []ToolDefinition `json:"tools"`
}

func (o OllamaChatRequest) GenerationOptions(def GenerationOptions) GenerationOptions {
	return applyRequestOptions(def, o.Options.NumPredict, o.Options.Temperature, o.Options.TopP, o.Options.TopK, o.Options.MinP, o.Options.RepeatPenalty, o.Options.Seed, nil, o.Options.Stop, o.Tools, "")
}

func (o OllamaChatRequest) ChatMessages() []ChatMessage {
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

func applyRequestOptions(def GenerationOptions, maxTokens *int, temp *float32, topP *float32, topK *int, minP *float32, repeat *float32, seed *uint64, system *string, stop any, tools []ToolDefinition, toolChoice string) GenerationOptions {
	o := def
	if maxTokens != nil {
		o.MaxTokens = *maxTokens
	}
	if temp != nil {
		o.Sampler.Temperature = *temp
	}
	if topP != nil {
		o.Sampler.TopP = *topP
	}
	if topK != nil {
		o.Sampler.TopK = *topK
	}
	if minP != nil {
		o.Sampler.MinP = *minP
	}
	if repeat != nil {
		o.Sampler.RepeatPenalty = *repeat
	}
	if seed != nil {
		o.Seed = *seed
	}
	if system != nil {
		o.SystemPrompt = *system
	}
	if parsed, ok := parseStop(stop); ok {
		o.StopSequences = parsed
	}
	if len(tools) > 0 {
		o.Tools = tools
	}
	if toolChoice != "" {
		o.ToolChoice = toolChoice
	}
	return o
}

func firstFloat(a, b *float32) *float32 {
	if a != nil {
		return a
	}
	return b
}

func parseStop(v any) ([]string, bool) {
	switch x := v.(type) {
	case string:
		return []string{x}, true
	case []any:
		out := []string{}
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out, true
	case nil:
		return nil, false
	default:
		return nil, false
	}
}

func firstStop(a, b any) any {
	if a != nil {
		return a
	}
	return b
}

func withLimit(sem chan struct{}, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sem <- struct{}{}
		defer func() { <-sem }()
		h(w, r)
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("content-type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func ensureRequestID(w http.ResponseWriter, req *http.Request) string {
	id := strings.TrimSpace(req.Header.Get("X-Request-ID"))
	if id == "" {
		id = fmt.Sprintf("gopherllm-%d-%d", time.Now().UnixNano(), inferenceRequestSeq.Add(1))
	}
	w.Header().Set("X-Request-ID", id)
	return id
}

func withRequestContext(options GenerationOptions, req *http.Request) GenerationOptions {
	options.ctx = req.Context()
	return options
}

type inferenceLogRecord struct {
	Event            string `json:"event"`
	RequestID        string `json:"request_id"`
	Endpoint         string `json:"endpoint"`
	Provider         string `json:"provider"`
	Model            string `json:"model"`
	Streaming        bool   `json:"streaming"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
	TTFTMS           int64  `json:"ttft_ms"`
	PrefillMS        int64  `json:"prefill_ms"`
	DecodeMS         int64  `json:"decode_ms"`
	TotalMS          int64  `json:"total_ms"`
	TokensPerSecond  string `json:"tokens_per_second"`
	Cache            string `json:"cache"`
	CacheHit         bool   `json:"cache_hit"`
	RetryCount       int    `json:"retry_count"`
	FinishReason     string `json:"finish_reason"`
	ErrorType        string `json:"error_type,omitempty"`
	Error            string `json:"error,omitempty"`
}

// Inference logs are deliberately emitted through the existing handler
// LogWriter instead of adding a metrics backend. Bottleneck: TTFT/decode
// regressions were not attributable per endpoint/request. Change: one
// structured JSON line per completed local inference. Effect: usable latency,
// throughput, token, cache, retry, and error dimensions for benchmarks and
// production logs. Risk: small log volume increase. Rollback: pass nil/Discard
// LogWriter or remove this helper call.
func logInferenceResult(logw io.Writer, requestID, endpoint, model string, streaming bool, result GenerationResult, err error) {
	if logw == nil || logw == io.Discard {
		return
	}
	errorType, errorText := "", ""
	if err != nil {
		errorType = fmt.Sprintf("%T", err)
		errorText = err.Error()
	}
	tps := float64(0)
	if result.Stats.DecodeTime > 0 {
		tps = float64(result.Stats.GeneratedTokens) / result.Stats.DecodeTime.Seconds()
	}
	rec := inferenceLogRecord{
		Event:            "inference",
		RequestID:        requestID,
		Endpoint:         endpoint,
		Provider:         "local",
		Model:            model,
		Streaming:        streaming,
		PromptTokens:     result.Stats.PromptTokens,
		CompletionTokens: result.Stats.GeneratedTokens,
		TTFTMS:           result.Stats.TTFT.Milliseconds(),
		PrefillMS:        result.Stats.PrefillTime.Milliseconds(),
		DecodeMS:         result.Stats.DecodeTime.Milliseconds(),
		TotalMS:          result.Stats.TotalTime.Milliseconds(),
		TokensPerSecond:  fmt.Sprintf("%.2f", tps),
		Cache:            "none",
		CacheHit:         false,
		RetryCount:       0,
		FinishReason:     finishReasonOrDefault(result.FinishReason),
		ErrorType:        errorType,
		Error:            errorText,
	}
	if b, jsonErr := json.Marshal(rec); jsonErr == nil {
		fmt.Fprintln(logw, string(b))
	}
}

// streamOpenAIChat streams a chat completion via SSE. Content deltas flow
// incrementally exactly as before whenever no tool call could possibly be in
// play; once skills or caller tools are active, RunAgenticChat buffers the
// winning turn and calls onToken once with the final, already-classified
// content, so raw tool-call syntax never leaks into a content delta (see
// RunAgenticChat's doc comment). Either way, the connection ends with one
// terminal chunk carrying finish_reason, usage, and — when present —
// reasoning_content and tool_calls, computed from the authoritative final
// GenerationResult rather than reconstructed from what was streamed.
func streamOpenAIChat(w http.ResponseWriter, req *http.Request, logw io.Writer, requestID string, r *Runner, model string, messages []ChatMessage, options GenerationOptions, skills []Skill) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("content-type", "text/event-stream; charset=utf-8")
	w.Header().Set("cache-control", "no-cache")
	w.Header().Set("connection", "keep-alive")
	w.Header().Set("x-accel-buffering", "no")

	id := fmt.Sprintf("chatcmpl-gopherllm-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	if err := writeOpenAIStreamChunk(w, flusher, id, model, created, map[string]any{"role": "assistant"}, nil); err != nil {
		return
	}

	var streamErr error
	result, err := RunAgenticChat(r, messages, options, skills, func(text string) bool {
		if ctxErr := req.Context().Err(); ctxErr != nil {
			streamErr = ctxErr
			return false
		}
		if err := writeOpenAIStreamChunk(w, flusher, id, model, created, map[string]any{"content": text}, nil); err != nil {
			streamErr = err
			return false
		}
		return true
	})
	if streamErr != nil {
		logInferenceResult(logw, requestID, "/v1/chat/completions", model, true, result, streamErr)
		return
	}
	logInferenceResult(logw, requestID, "/v1/chat/completions", model, true, result, err)
	if err != nil {
		if errors.Is(err, ErrGenerationCanceled) {
			return
		}
		writeSSE(w, flusher, "error", map[string]string{"error": err.Error()})
		return
	}
	finalDelta := map[string]any{}
	if result.ReasoningText != "" {
		finalDelta["reasoning_content"] = result.ReasoningText
	}
	if len(result.ToolCalls) > 0 {
		finalDelta["tool_calls"] = result.ToolCalls
	}
	_ = writeOpenAIStreamChunk(w, flusher, id, model, created, finalDelta, map[string]any{"finish_reason": finishReasonOrDefault(result.FinishReason), "usage": usage(result)})
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeOpenAIStreamChunk(w http.ResponseWriter, flusher http.Flusher, id, model string, created int64, delta map[string]any, extra map[string]any) error {
	choice := map[string]any{"index": 0, "delta": delta}
	for k, v := range extra {
		choice[k] = v
	}
	return writeSSE(w, flusher, "", map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []any{choice},
	})
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, event string, v any) error {
	if event != "" {
		if _, err := fmt.Fprintf(w, "event: %s\n", event); err != nil {
			return err
		}
	}
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func generateResponse(result GenerationResult) map[string]any {
	resp := map[string]any{"text": result.Text, "prompt_tokens": result.Stats.PromptTokens, "generated_tokens": result.Stats.GeneratedTokens, "ttft_ms": result.Stats.TTFT.Milliseconds(), "prefill_ms": result.Stats.PrefillTime.Milliseconds(), "decode_ms": result.Stats.DecodeTime.Milliseconds(), "total_ms": result.Stats.TotalTime.Milliseconds(), "finish_reason": finishReasonOrDefault(result.FinishReason)}
	if result.ReasoningText != "" {
		resp["reasoning"] = result.ReasoningText
	}
	if len(result.ToolCalls) > 0 {
		resp["tool_calls"] = result.ToolCalls
	}
	return resp
}

func openAIChatResponse(model string, result GenerationResult) map[string]any {
	message := map[string]any{"role": "assistant", "content": result.Text}
	if len(result.ToolCalls) > 0 {
		message["tool_calls"] = result.ToolCalls
		if result.Text == "" {
			message["content"] = nil
		}
	}
	if result.ReasoningText != "" {
		message["reasoning_content"] = result.ReasoningText
	}
	return map[string]any{"id": "chatcmpl-gopherllm", "object": "chat.completion", "created": time.Now().Unix(), "model": model, "choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finishReasonOrDefault(result.FinishReason)}}, "usage": usage(result)}
}

// finishReasonOrDefault falls back to "stop" for callers of GenerateResult
// that predate FinishReason (in-tree, only GenerationResult zero values hit
// this) so every response always carries a valid OpenAI-shaped finish_reason.
func finishReasonOrDefault(reason string) string {
	if reason == "" {
		return "stop"
	}
	return reason
}

func usage(result GenerationResult) map[string]int {
	return map[string]int{"prompt_tokens": result.Stats.PromptTokens, "completion_tokens": result.Stats.GeneratedTokens, "total_tokens": result.Stats.PromptTokens + result.Stats.GeneratedTokens}
}

func modelID(r *Runner) string {
	if name, ok := r.ModelName(); ok && name != "" {
		return name
	}
	return "gopherllm"
}
