package main

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

//go:embed web_ui/chat.html
var chatHTMLTmpl string

//go:embed web_ui/style.css
var chatCSS string

//go:embed web_ui/script.js
var chatJS string

var chatTemplate = template.Must(template.New("chat").Parse(chatHTMLTmpl))

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

type ServeOptions struct {
	Addr                     string
	Defaults                 GenerationOptions
	MaxConcurrentConnections int
	ChatUI                   bool
	ChatHistoryPath          string
	ChatHistoryLock          *sync.Mutex
	ModelDir                 string
	ModelPath                string
}

func Serve(initialRunner *Runner, opts ServeOptions) error {
	if opts.MaxConcurrentConnections <= 0 {
		opts.MaxConcurrentConnections = 8
	}
	state := &runnerState{r: initialRunner, path: opts.ModelPath}
	sem := make(chan struct{}, opts.MaxConcurrentConnections)
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"ok": true, "model": modelID(state.get())})
	})
	mux.HandleFunc("/generate", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body GenerateRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		messages, options := body.ToMessagesAndOptions(opts.Defaults)
		state.withRunner(func(r *Runner) {
			result, err := r.GenerateChat(messages, options)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, generateResponse(result))
		})
	}))
	mux.HandleFunc("/v1/chat/completions", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body OpenAIChatRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.Options(opts.Defaults)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			if body.Stream {
				streamOpenAIChat(w, req, r, model, body.ChatMessages(), options)
				return
			}
			result, err := r.GenerateChat(body.ChatMessages(), options)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, openAIChatResponse(model, result))
		})
	}))
	mux.HandleFunc("/v1/completions", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body OpenAICompletionRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.Options(opts.Defaults)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			result, err := r.Generate(body.PromptString(), options)
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
	mux.HandleFunc("/api/generate", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body OllamaGenerateRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.GenerationOptions(opts.Defaults)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			result, err := r.Generate(body.Prompt, options)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, map[string]any{"model": model, "created_at": time.Now().Format(time.RFC3339Nano), "response": result.Text, "done": true, "prompt_eval_count": result.Stats.PromptTokens, "eval_count": result.Stats.GeneratedTokens})
		})
	}))
	mux.HandleFunc("/api/chat", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body OllamaChatRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.GenerationOptions(opts.Defaults)
		state.withRunner(func(r *Runner) {
			model := modelID(r)
			result, err := r.GenerateChat(body.ChatMessages(), options)
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
		entries, err := DiscoverModels(opts.ModelDir)
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
	server := &http.Server{Addr: opts.Addr, Handler: mux, ReadHeaderTimeout: 30 * time.Second}
	fmt.Fprintf(stderr(), "Serving on %s\n", displayServerURL(opts.Addr, opts.ChatUI))
	return server.ListenAndServe()
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
	Prompt        string       `json:"prompt"`
	Messages      []APIMessage `json:"messages"`
	MaxTokens     *int         `json:"max_tokens"`
	Temp          *float32     `json:"temp"`
	Temperature   *float32     `json:"temperature"`
	TopP          *float32     `json:"top_p"`
	TopK          *int         `json:"top_k"`
	RepeatPenalty *float32     `json:"repeat_penalty"`
	Seed          *uint64      `json:"seed"`
	SystemPrompt  *string      `json:"system_prompt"`
	Stop          any          `json:"stop"`
}

func (g GenerateRequest) ToMessagesAndOptions(def GenerationOptions) ([]ChatMessage, GenerationOptions) {
	options := applyRequestOptions(def, g.MaxTokens, firstFloat(g.Temp, g.Temperature), g.TopP, g.TopK, g.RepeatPenalty, g.Seed, g.SystemPrompt, g.Stop)
	if len(g.Messages) > 0 {
		return apiMessages(g.Messages), options
	}
	return []ChatMessage{UserMessage(g.Prompt)}, options
}

type APIMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
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
		}
		out = append(out, ChatMessage{Role: role, Content: contentText(item.Content)})
	}
	return out
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
	Model               string       `json:"model"`
	Messages            []APIMessage `json:"messages"`
	Stream              bool         `json:"stream"`
	MaxTokens           *int         `json:"max_tokens"`
	MaxCompletionTokens *int         `json:"max_completion_tokens"`
	Temperature         *float32     `json:"temperature"`
	TopP                *float32     `json:"top_p"`
	TopK                *int         `json:"top_k"`
	RepeatPenalty       *float32     `json:"repeat_penalty"`
	Seed                *uint64      `json:"seed"`
	SystemPrompt        *string      `json:"system_prompt"`
	Stop                any          `json:"stop"`
}

func (o OpenAIChatRequest) Options(def GenerationOptions) GenerationOptions {
	maxTokens := o.MaxTokens
	if maxTokens == nil {
		maxTokens = o.MaxCompletionTokens
	}
	return applyRequestOptions(def, maxTokens, o.Temperature, o.TopP, o.TopK, o.RepeatPenalty, o.Seed, o.SystemPrompt, o.Stop)
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
	return applyRequestOptions(def, maxTokens, o.Temperature, o.TopP, o.TopK, o.RepeatPenalty, o.Seed, o.SystemPrompt, o.Stop)
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
	return applyRequestOptions(def, o.Options.NumPredict, o.Options.Temperature, o.Options.TopP, o.Options.TopK, o.Options.RepeatPenalty, o.Options.Seed, system, firstStop(o.Stop, o.Options.Stop))
}

type OllamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  OllamaOptions   `json:"options"`
}

func (o OllamaChatRequest) GenerationOptions(def GenerationOptions) GenerationOptions {
	return applyRequestOptions(def, o.Options.NumPredict, o.Options.Temperature, o.Options.TopP, o.Options.TopK, o.Options.RepeatPenalty, o.Options.Seed, nil, o.Options.Stop)
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

func applyRequestOptions(def GenerationOptions, maxTokens *int, temp *float32, topP *float32, topK *int, repeat *float32, seed *uint64, system *string, stop any) GenerationOptions {
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

func streamOpenAIChat(w http.ResponseWriter, req *http.Request, r *Runner, model string, messages []ChatMessage, options GenerationOptions) {
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
	if err := writeOpenAIStreamChunk(w, flusher, id, model, created, map[string]string{"role": "assistant"}, nil); err != nil {
		return
	}

	var streamErr error
	result, err := r.GenerateChatStreamUntil(messages, options, func(text string) bool {
		if ctxErr := req.Context().Err(); ctxErr != nil {
			streamErr = ctxErr
			return false
		}
		if err := writeOpenAIStreamChunk(w, flusher, id, model, created, map[string]string{"content": text}, nil); err != nil {
			streamErr = err
			return false
		}
		return true
	})
	if streamErr != nil {
		return
	}
	if err != nil {
		if errors.Is(err, ErrGenerationCanceled) {
			return
		}
		writeSSE(w, flusher, "error", map[string]string{"error": err.Error()})
		return
	}
	usage := usage(result)
	_ = writeOpenAIStreamChunk(w, flusher, id, model, created, map[string]string{}, map[string]any{"finish_reason": "stop", "usage": usage})
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeOpenAIStreamChunk(w http.ResponseWriter, flusher http.Flusher, id, model string, created int64, delta map[string]string, extra map[string]any) error {
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
	return map[string]any{"text": result.Text, "prompt_tokens": result.Stats.PromptTokens, "generated_tokens": result.Stats.GeneratedTokens, "prefill_ms": result.Stats.PrefillTime.Milliseconds(), "decode_ms": result.Stats.DecodeTime.Milliseconds(), "total_ms": result.Stats.TotalTime.Milliseconds()}
}

func openAIChatResponse(model string, result GenerationResult) map[string]any {
	return map[string]any{"id": "chatcmpl-gopherllm", "object": "chat.completion", "created": time.Now().Unix(), "model": model, "choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": result.Text}, "finish_reason": "stop"}}, "usage": usage(result)}
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
