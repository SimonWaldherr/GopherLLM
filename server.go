package gopherllm

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
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
				includeUsage := body.StreamOptions != nil && body.StreamOptions.IncludeUsage
				streamOpenAIChat(w, req, logw, requestID, r, model, body.ChatMessages(), options, skills, includeUsage)
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
			if streamEnabled(body.Stream) {
				streamOllamaGenerate(w, req, logw, requestID, r, model, body.Prompt, options, skills)
				return
			}
			result, err := RunAgenticChat(r, []ChatMessage{UserMessage(body.Prompt)}, options, skills, alwaysContinue)
			logInferenceResult(logw, requestID, "/api/generate", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			resp := map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "response": result.Text, "done": true, "done_reason": finishReasonOrDefault(result.FinishReason)}
			for k, v := range ollamaDurations(result.Stats) {
				resp[k] = v
			}
			writeJSON(w, resp)
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
			if streamEnabled(body.Stream) {
				streamOllamaChat(w, req, logw, requestID, r, model, body.ChatMessages(), options, skills)
				return
			}
			result, err := RunAgenticChat(r, body.ChatMessages(), options, skills, alwaysContinue)
			logInferenceResult(logw, requestID, "/api/chat", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			message := map[string]any{"role": "assistant", "content": result.Text}
			if len(result.ToolCalls) > 0 {
				message["tool_calls"] = result.ToolCalls
			}
			resp := map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "message": message, "done": true, "done_reason": finishReasonOrDefault(result.FinishReason)}
			for k, v := range ollamaDurations(result.Stats) {
				resp[k] = v
			}
			writeJSON(w, resp)
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
	mux.HandleFunc("/api/embed", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body OllamaEmbedRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		inputs := body.Inputs()
		embeddings := make([][]float32, 0, len(inputs))
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			promptTokens := 0
			for _, input := range inputs {
				emb, err := r.Embed(input)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				promptTokens += emb.TokenCount
				embeddings = append(embeddings, emb.Embedding)
			}
			writeJSON(w, map[string]any{"model": model, "embeddings": embeddings, "prompt_eval_count": promptTokens})
		})
	}))
	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"models": ollamaTagEntries(state, opts.ModelDir)})
	})
	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, _ *http.Request) {
		r := state.get()
		if r == nil {
			writeJSON(w, map[string]any{"models": []any{}})
			return
		}
		name := modelID(r)
		a := AnalyzeGGUF(r.GGUF(), r.Tokenizer())
		writeJSON(w, map[string]any{"models": []any{map[string]any{
			"name":       name,
			"model":      name,
			"size":       a.FileBytes,
			"size_vram":  a.FileBytes,
			"digest":     modelDigest(state.getPath()),
			"details":    ollamaModelDetails(a),
			"expires_at": time.Now().Add(5 * time.Minute).Format(time.RFC3339Nano),
		}}})
	})
	mux.HandleFunc("/api/show", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body struct {
			Model string `json:"model"`
			Name  string `json:"name"`
		}
		if req.Body != nil {
			_ = json.NewDecoder(req.Body).Decode(&body)
		}
		requested := body.Model
		if requested == "" {
			requested = body.Name
		}
		a, ok := resolveModelAnalysis(state, opts.ModelDir, requested)
		if !ok {
			http.Error(w, "model not found", http.StatusNotFound)
			return
		}
		writeJSON(w, map[string]any{
			"modelfile":  "",
			"parameters": "",
			"template":   "",
			"details":    ollamaModelDetails(a),
			"model_info": map[string]any{
				"general.architecture": a.Architecture,
				"general.parameter_count": a.Params,
				a.Architecture + ".context_length": a.ContextLength,
				a.Architecture + ".embedding_length": a.Dim,
				a.Architecture + ".block_count": a.Layers,
				a.Architecture + ".attention.head_count": a.Heads,
				a.Architecture + ".attention.head_count_kv": a.KVHeads,
			},
			"capabilities": []string{"completion"},
		})
	}))
	mux.HandleFunc("/api/version", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"version": "gopherllm-ollama-compat"})
	})
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
		case "system", "developer":
			// "developer" is the OpenAI o1/gpt-oss-era replacement for
			// "system"; GopherLLM has no separate developer-instruction
			// channel, so it renders the same as a system message.
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
// offering; "auto"/"required" pass through unchanged (GopherLLM has no
// constrained decoding, so both just mean "offer the tools"); an object
// naming one function (`{"type":"function","function":{"name":"..."}}`)
// becomes "function:<name>", which GenerationOptions.activeTools narrows
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
	Model               string             `json:"model"`
	Messages            []APIMessage       `json:"messages"`
	Stream              bool               `json:"stream"`
	StreamOptions       *OpenAIStreamOpts  `json:"stream_options"`
	MaxTokens           *int               `json:"max_tokens"`
	MaxCompletionTokens *int               `json:"max_completion_tokens"`
	Temperature         *float32           `json:"temperature"`
	TopP                *float32           `json:"top_p"`
	TopK                *int               `json:"top_k"`
	MinP                *float32           `json:"min_p"`
	RepeatPenalty       *float32           `json:"repeat_penalty"`
	Seed                *uint64            `json:"seed"`
	SystemPrompt        *string            `json:"system_prompt"`
	Stop                any                `json:"stop"`
	Tools               []ToolDefinition   `json:"tools"`
	ToolChoice          any                `json:"tool_choice"`
}

// OpenAIStreamOpts is the OpenAI "stream_options" object; IncludeUsage gates
// whether the final SSE chunk carries a "usage" field (off by default, per
// spec — unlike a non-streaming response, which always includes usage).
type OpenAIStreamOpts struct {
	IncludeUsage bool `json:"include_usage"`
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
	Stream  *bool         `json:"stream"`
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
	Stream   *bool            `json:"stream"`
	Options  OllamaOptions    `json:"options"`
	Tools    []ToolDefinition `json:"tools"`
}

// streamEnabled implements Ollama's default-true streaming semantics: the
// request only turns streaming off when the "stream" field is explicitly
// present and false; omitting it (nil) streams, matching real Ollama.
func streamEnabled(b *bool) bool {
	return b == nil || *b
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
	// NumCtx is accepted for wire compatibility (real Ollama clients set it
	// routinely) but not actionable here: a Runner's KV cache is sized once
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
func streamOpenAIChat(w http.ResponseWriter, req *http.Request, logw io.Writer, requestID string, r *Runner, model string, messages []ChatMessage, options GenerationOptions, skills []Skill, includeUsage bool) {
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
	extra := map[string]any{"finish_reason": finishReasonOrDefault(result.FinishReason)}
	if includeUsage {
		extra["usage"] = usage(result)
	}
	_ = writeOpenAIStreamChunk(w, flusher, id, model, created, finalDelta, extra)
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// systemFingerprint is a static stand-in for OpenAI's build-identifying
// system_fingerprint field: GopherLLM has no server-side config permutations
// that would make it vary per request, but clients (agent frameworks,
// caching layers) expect the field to be present.
const systemFingerprint = "fp_gopherllm"

func writeOpenAIStreamChunk(w http.ResponseWriter, flusher http.Flusher, id, model string, created int64, delta map[string]any, extra map[string]any) error {
	choice := map[string]any{"index": 0, "delta": delta}
	for k, v := range extra {
		choice[k] = v
	}
	return writeSSE(w, flusher, "", map[string]any{
		"id":                 id,
		"object":             "chat.completion.chunk",
		"created":            created,
		"model":              model,
		"system_fingerprint": systemFingerprint,
		"choices":            []any{choice},
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

// writeNDJSON writes one newline-delimited JSON object and flushes — Ollama's
// streaming wire format, distinct from OpenAI's "data: "-prefixed SSE.
func writeNDJSON(w http.ResponseWriter, flusher http.Flusher, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := w.Write(append(data, '\n')); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

// ollamaDurations reports GenerationStats using Ollama's nanosecond duration
// field names. load_duration is always 0: GopherLLM has no separate
// model-load phase inside a request (the model is already resident).
func ollamaDurations(stats GenerationStats) map[string]any {
	return map[string]any{
		"total_duration":       stats.TotalTime.Nanoseconds(),
		"load_duration":        int64(0),
		"prompt_eval_count":    stats.PromptTokens,
		"prompt_eval_duration": stats.PrefillTime.Nanoseconds(),
		"eval_count":           stats.GeneratedTokens,
		"eval_duration":        stats.DecodeTime.Nanoseconds(),
	}
}

// streamOllamaGenerate streams /api/generate as NDJSON: one {"done":false}
// line per token, then a final {"done":true} line carrying finish reason and
// timing, mirroring real Ollama's wire shape.
func streamOllamaGenerate(w http.ResponseWriter, req *http.Request, logw io.Writer, requestID string, r *Runner, model, prompt string, options GenerationOptions, skills []Skill) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("content-type", "application/x-ndjson")

	var streamErr error
	result, err := RunAgenticChat(r, []ChatMessage{UserMessage(prompt)}, options, skills, func(text string) bool {
		if ctxErr := req.Context().Err(); ctxErr != nil {
			streamErr = ctxErr
			return false
		}
		if err := writeNDJSON(w, flusher, map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "response": text, "done": false}); err != nil {
			streamErr = err
			return false
		}
		return true
	})
	if streamErr != nil {
		logInferenceResult(logw, requestID, "/api/generate", model, true, result, streamErr)
		return
	}
	logInferenceResult(logw, requestID, "/api/generate", model, true, result, err)
	if err != nil {
		if errors.Is(err, ErrGenerationCanceled) {
			return
		}
		_ = writeNDJSON(w, flusher, map[string]string{"error": err.Error()})
		return
	}
	final := map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "response": "", "done": true, "done_reason": finishReasonOrDefault(result.FinishReason)}
	for k, v := range ollamaDurations(result.Stats) {
		final[k] = v
	}
	_ = writeNDJSON(w, flusher, final)
}

// streamOllamaChat streams /api/chat as NDJSON, surfacing tool_calls on the
// final message the same way the non-streaming path does (previously dropped
// entirely on this endpoint).
func streamOllamaChat(w http.ResponseWriter, req *http.Request, logw io.Writer, requestID string, r *Runner, model string, messages []ChatMessage, options GenerationOptions, skills []Skill) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("content-type", "application/x-ndjson")

	var streamErr error
	result, err := RunAgenticChat(r, messages, options, skills, func(text string) bool {
		if ctxErr := req.Context().Err(); ctxErr != nil {
			streamErr = ctxErr
			return false
		}
		if err := writeNDJSON(w, flusher, map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "message": map[string]any{"role": "assistant", "content": text}, "done": false}); err != nil {
			streamErr = err
			return false
		}
		return true
	})
	if streamErr != nil {
		logInferenceResult(logw, requestID, "/api/chat", model, true, result, streamErr)
		return
	}
	logInferenceResult(logw, requestID, "/api/chat", model, true, result, err)
	if err != nil {
		if errors.Is(err, ErrGenerationCanceled) {
			return
		}
		_ = writeNDJSON(w, flusher, map[string]string{"error": err.Error()})
		return
	}
	message := map[string]any{"role": "assistant", "content": ""}
	if len(result.ToolCalls) > 0 {
		message["tool_calls"] = result.ToolCalls
	}
	final := map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "message": message, "done": true, "done_reason": finishReasonOrDefault(result.FinishReason)}
	for k, v := range ollamaDurations(result.Stats) {
		final[k] = v
	}
	_ = writeNDJSON(w, flusher, final)
}

// analyzeModelFile parses a GGUF file's header only (mmap'd, no weight
// bytes touched — the same cheap path DiscoverModels uses) into an Analysis,
// for building Ollama-shaped model metadata without loading a full Runner.
func analyzeModelFile(path string) (*Analysis, error) {
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, err
	}
	defer mmap.Close()
	gguf, err := ParseGGUFQuiet(mmap.Bytes())
	if err != nil {
		return nil, err
	}
	return AnalyzeGGUF(gguf, nil), nil
}

// resolveModelAnalysis answers /api/show's "which model": empty/matching
// name means the currently loaded Runner (full Analysis, tokenizer
// included); any other name is looked up in ModelDir (if configured) and
// header-analyzed on demand.
func resolveModelAnalysis(state *runnerState, modelDir, requested string) (*Analysis, bool) {
	r := state.get()
	if r != nil && (requested == "" || requested == modelID(r)) {
		return AnalyzeGGUF(r.GGUF(), r.Tokenizer()), true
	}
	if modelDir == "" {
		return nil, false
	}
	entries, err := DiscoverModels(modelDir, io.Discard)
	if err != nil {
		return nil, false
	}
	entry, err := SelectModel(entries, requested)
	if err != nil {
		return nil, false
	}
	a, err := analyzeModelFile(entry.Path)
	if err != nil {
		return nil, false
	}
	return a, true
}

// ollamaTagEntries builds /api/tags' model list: every entry under ModelDir
// (header-analyzed, cheap) if configured, else just the currently loaded
// model.
func ollamaTagEntries(state *runnerState, modelDir string) []map[string]any {
	if modelDir == "" {
		r := state.get()
		if r == nil {
			return []map[string]any{}
		}
		name := modelID(r)
		a := AnalyzeGGUF(r.GGUF(), r.Tokenizer())
		return []map[string]any{ollamaTagEntry(name, state.getPath(), a.FileBytes, a)}
	}
	entries, err := DiscoverModels(modelDir, io.Discard)
	if err != nil {
		return []map[string]any{}
	}
	tags := make([]map[string]any, 0, len(entries))
	for _, e := range entries {
		if e.IsProjector || !e.IsSupported {
			continue
		}
		name := e.ModelName
		if name == "" {
			name = e.FileName
		}
		a, err := analyzeModelFile(e.Path)
		if err != nil {
			continue
		}
		tags = append(tags, ollamaTagEntry(name, e.Path, e.SizeBytes, a))
	}
	return tags
}

func ollamaTagEntry(name, path string, sizeBytes int64, a *Analysis) map[string]any {
	return map[string]any{
		"name":        name,
		"model":       name,
		"modified_at": time.Now().Format(time.RFC3339Nano),
		"size":        sizeBytes,
		"digest":      modelDigest(path),
		"details":     ollamaModelDetails(a),
	}
}

// ollamaModelDetails builds Ollama's "details" object from a header Analysis.
func ollamaModelDetails(a *Analysis) map[string]any {
	quant := "unknown"
	if len(a.DTypes) > 0 {
		quant = a.DTypes[0].Type.String()
	}
	family := a.Architecture
	if family == "" {
		family = "unknown"
	}
	return map[string]any{
		"parent_model":       "",
		"format":             "gguf",
		"family":             family,
		"families":           []string{family},
		"parameter_size":     humanParamSize(a.Params),
		"quantization_level": quant,
	}
}

// humanParamSize formats a parameter count the way Ollama's "parameter_size"
// field does (e.g. "3.3B", "125M").
func humanParamSize(params int64) string {
	switch {
	case params >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(params)/1_000_000_000)
	case params >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(params)/1_000_000)
	default:
		return fmt.Sprintf("%d", params)
	}
}

// modelDigest returns a stable, cheap-to-compute "sha256:"-prefixed
// identifier for a model file: real Ollama content-addresses the whole blob,
// but sha256-ing a multi-gigabyte GGUF on every /api/tags or /api/show
// request would be far too slow. This hashes the path, size, and first 1MiB
// only — good enough as an opaque, stable client-facing id, not a real
// content hash.
func modelDigest(path string) string {
	h := sha256.New()
	io.WriteString(h, path)
	if st, err := os.Stat(path); err == nil {
		fmt.Fprintf(h, "%d", st.Size())
		if f, err := os.Open(path); err == nil {
			defer f.Close()
			_, _ = io.CopyN(h, f, 1<<20)
		}
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
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
	return map[string]any{"id": "chatcmpl-gopherllm", "object": "chat.completion", "created": time.Now().Unix(), "model": model, "system_fingerprint": systemFingerprint, "choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finishReasonOrDefault(result.FinishReason)}}, "usage": usage(result)}
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
