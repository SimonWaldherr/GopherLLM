package server

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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/agentos"
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
	Title         string
	Model         string
	MaxTokens     int
	Temperature   float32
	TopP          float32
	TopK          int
	MinP          float32
	RepeatPenalty float32
	// MermaidCDN is the validated CDN key ("" when diagrams are off) and
	// MermaidScript the full script URL the page should load.
	MermaidCDN    string
	MermaidScript string
}

type runnerState struct {
	mu   sync.RWMutex
	r    *gopherllm.Runner
	path string
	// baseline is captured before any server-side auto tuning. A model switch
	// restores it when the replacement has not been explicitly tuned, so
	// process-wide knobs measured for the previous model do not leak across.
	baseline gopherllm.RuntimeTuning
	// autoTune is the tuning result actually applied to r during this process
	// (nil if none has been). It is distinct from gopherllm.Runner.LoadAutoTune, which
	// only reports what is cached on disk and may not reflect what is
	// currently active if the server started without --auto.
	autoTune *gopherllm.AutoTuneResult
}

// embeddingState holds the optional, separately loaded model used by the
// browser RAG mode. Enabling RAG must never replace the chat generation model.
type embeddingState struct {
	mu sync.RWMutex
	r  *gopherllm.Runner
}

func (s *embeddingState) withRunner(fn func(*gopherllm.Runner)) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	fn(s.r)
}

func (s *embeddingState) swap(r *gopherllm.Runner) {
	s.mu.Lock()
	old := s.r
	s.r = r
	s.mu.Unlock()
	if old != nil {
		_ = old.Close()
	}
}

func (s *runnerState) get() *gopherllm.Runner {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.r
}

func (s *runnerState) withRunner(fn func(*gopherllm.Runner)) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	fn(s.r)
}

func (s *runnerState) swap(r *gopherllm.Runner, path string) {
	var old *gopherllm.Runner
	s.mu.Lock()
	old = s.r
	s.r = r
	s.path = path
	// A hot-swapped model has not had a model-specific calibration applied.
	// Restore the pre-auto baseline while holding the writer lock: all inference
	// paths hold its read lock for their whole run, so no generation observes a
	// mixture of old and restored process-global knobs.
	s.baseline.Apply()
	s.autoTune = nil
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

// runAutoTune serializes calibration with model hot-swaps and all inference.
// gopherllm.AutoTuneResult.Apply changes process-wide settings, so holding the writer
// lock is intentional: inference paths hold a read lock for the entire run.
func (s *runnerState) runAutoTune(opts gopherllm.AutoTuneOptions, refresh bool) (gopherllm.AutoTuneResult, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.r == nil {
		return gopherllm.AutoTuneResult{}, false, errors.New("no model is loaded; load a model first")
	}
	res, cached, err := s.r.AutoTuneOrCached(opts, refresh)
	if err == nil {
		s.autoTune = &res
	}
	return res, cached, err
}

// autoTuneStatus reports what GET /autotune needs to render the web UI's
// panel: whether a tuning is active this session, whether one is persisted on
// disk for the current model+host, and the most relevant result to show.
func (s *runnerState) autoTuneStatus() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	status := map[string]any{"active": false, "cached": false}
	if s.r == nil {
		status["metal_available"] = gopherllm.MetalAvailable()
		if !gopherllm.MetalAvailable() {
			status["metal_hint"] = gopherllm.MetalError()
		}
		return status
	}
	if s.autoTune != nil {
		_, cached := s.r.LoadAutoTune()
		status = map[string]any{"active": true, "cached": cached, "result": s.autoTune}
	} else if cached, ok := s.r.LoadAutoTune(); ok {
		status = map[string]any{"active": false, "cached": true, "result": cached}
	}
	// Metal is a build-time choice, not something auto-tuning can switch on,
	// so a build without it silently leaves a large speedup on the table
	// (measured ~1.8x on a 3B Q4_K_M). Report it so the UI can say so rather
	// than letting it stay invisible.
	status["metal_available"] = gopherllm.MetalAvailable()
	if !gopherllm.MetalAvailable() {
		status["metal_hint"] = gopherllm.MetalError()
	}
	return status
}

// HandlerOptions configures the mountable HTTP API handler.
type HandlerOptions struct {
	// Defaults are the generation settings requests inherit unless they
	// override individual fields.
	Defaults gopherllm.GenerationOptions
	// MaxConcurrentRequests bounds in-flight generation requests (default 8).
	// Requests beyond the bound queue rather than failing.
	MaxConcurrentRequests int
	// ChatUI serves the embedded browser chat at /chat (plus its assets).
	ChatUI bool
	// ModelDir enables GET /models discovery and POST /models/load hot-swap.
	// A load request is resolved against this directory's discovered,
	// supported GGUF files; arbitrary filesystem paths are never loaded.
	ModelDir string
	// ModelPath is the initially loaded model's path (reported by /models).
	ModelPath string
	// ModelLoadOptions are retained for catalog hot-swaps, so a server started
	// in out-of-core mode does not accidentally load the next model eagerly.
	ModelLoadOptions gopherllm.LoadOptions
	// ModelLoaded is called after a local model has been successfully loaded or
	// hot-swapped. It is useful for hosts that persist the active selection.
	// Callback failures are the host's responsibility and never undo a load.
	ModelLoaded func(path string)
	// SkillsDir, if set, is scanned once at handler construction for SKILL.md
	// files (see skills.go). Every chat/generate endpoint offers a load_skill
	// tool and resolves it server-side via gopherllm.RunAgenticChat.
	SkillsDir string
	// AppliedAutoTune, if set, is a tuning already applied to initialRunner
	// before the handler was built (e.g. by the CLI's --auto flag). GET
	// /autotune reports it as active from the very first request, rather than
	// only after someone hits POST /autotune/run through the web UI.
	AppliedAutoTune *gopherllm.AutoTuneResult
	// BaselineRuntimeTuning is the process-wide configuration to restore after
	// a hot-swap to an uncalibrated model. When absent, NewHandler captures the
	// settings visible at construction time.
	BaselineRuntimeTuning *gopherllm.RuntimeTuning
	// LogWriter receives handler diagnostics (skill load notes). Defaults to
	// io.Discard.
	LogWriter io.Writer
	// AgentOS enables the agentic OS-command feature (a model proposes a local
	// shell command, this Runner's Policy decides whether it needs a human
	// click before it runs) when non-nil. Nil, the default, registers no
	// /agentos endpoints at all — the feature does not exist on this server
	// unless an operator deliberately configured a policy for it. See the
	// agentos package for the safety model.
	AgentOS *agentos.Runner
}

// modelLoadRequest accepts the model catalog ID used by the browser UI and
// API clients. Path remains for compatibility with older clients, but is
// treated only as a selector for a model already discovered in ModelDir.
type modelLoadRequest struct {
	Model string `json:"model"`
	ID    string `json:"id"`
	Path  string `json:"path"`
}

func (r modelLoadRequest) selector() string {
	for _, value := range []string{r.Model, r.ID, r.Path} {
		if selector := strings.TrimSpace(value); selector != "" {
			return selector
		}
	}
	return ""
}

// ServeOptions is HandlerOptions plus the listen address, for the Serve
// convenience wrapper (used by the CLI). ChatHistoryPath/ChatHistoryLock are
// retained for compatibility but unused.
type ServeOptions struct {
	Addr                     string
	Defaults                 gopherllm.GenerationOptions
	MaxConcurrentConnections int
	ChatUI                   bool
	ChatHistoryPath          string
	ChatHistoryLock          *sync.Mutex
	ModelDir                 string
	ModelPath                string
	ModelLoadOptions         gopherllm.LoadOptions
	ModelLoaded              func(path string)
	SkillsDir                string
	// AppliedAutoTune carries forward a tuning already applied before Serve
	// was called (e.g. by --auto), so GET /autotune reports it from the start.
	AppliedAutoTune *gopherllm.AutoTuneResult
	// BaselineRuntimeTuning forwards the pre-auto runtime settings captured by
	// a host such as the CLI, so a later model hot-swap can restore them.
	BaselineRuntimeTuning *gopherllm.RuntimeTuning
	// LogWriter receives startup and handler diagnostics; Serve defaults it
	// to os.Stderr (CLI behavior), unlike NewHandler's io.Discard.
	LogWriter io.Writer
	// AgentOS enables the agentic OS-command feature; see HandlerOptions.AgentOS.
	AgentOS *agentos.Runner
}

// Serve builds the API handler and runs a blocking http.Server on opts.Addr.
// Library consumers who want to control the server lifecycle, add middleware,
// TLS, or mount the API under a path prefix should use NewHandler instead:
//
//	handler := gopherllm.NewHandler(model.Runner(), gopherllm.HandlerOptions{...})
//	mux.Handle("/llm/", http.StripPrefix("/llm", handler))
func Serve(initialRunner *gopherllm.Runner, opts ServeOptions) error {
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
		ModelLoadOptions:      opts.ModelLoadOptions,
		ModelLoaded:           opts.ModelLoaded,
		SkillsDir:             opts.SkillsDir,
		AppliedAutoTune:       opts.AppliedAutoTune,
		BaselineRuntimeTuning: opts.BaselineRuntimeTuning,
		LogWriter:             logw,
		AgentOS:               opts.AgentOS,
	})
	server := &http.Server{Addr: opts.Addr, Handler: handler, ReadHeaderTimeout: 30 * time.Second}
	fmt.Fprintf(logw, "Serving on %s\n", displayServerURL(opts.Addr, opts.ChatUI))
	return server.ListenAndServe()
}

// HandlerForModel serves a Model opened through the high-level API. It is the
// replacement for the former gopherllm.Model.HTTPHandler method, which had to
// go when HTTP serving moved out of the inference package: the handler shares
// the Model's underlying Runner, so requests serialize with direct Model calls.
func HandlerForModel(m *gopherllm.Model, opts HandlerOptions) http.Handler {
	return NewHandler(m.Runner(), opts)
}

// NewHandler returns the complete GopherLLM HTTP API (OpenAI-compatible,
// Ollama-compatible, and native endpoints — see the README's endpoint table)
// as a mountable http.Handler. It owns no listener and writes nothing except
// to opts.LogWriter, so it composes with any router, middleware stack, or
// server the host application already has.
func NewHandler(initialRunner *gopherllm.Runner, opts HandlerOptions) http.Handler {
	logw := opts.LogWriter
	if logw == nil {
		logw = io.Discard
	}
	opts.ModelDir = strings.TrimSpace(opts.ModelDir)
	if opts.MaxConcurrentRequests <= 0 {
		opts.MaxConcurrentRequests = 8
	}
	skills, err := gopherllm.LoadSkills(opts.SkillsDir)
	if err != nil {
		fmt.Fprintf(logw, "Warning: skills: %v (continuing without skills)\n", err)
	} else if len(skills) > 0 {
		names := make([]string, len(skills))
		for i, s := range skills {
			names[i] = s.Name
		}
		fmt.Fprintf(logw, "Skills: loaded %d (%s)\n", len(skills), strings.Join(names, ", "))
	}
	wikimediaTools := newWikimediaClient(nil).tools()
	skillsFor := func(enabled bool) []gopherllm.Skill {
		if !enabled {
			return nil
		}
		return skills
	}
	agenticToolsFor := func(enabled bool) []gopherllm.AgenticTool {
		if !enabled {
			return nil
		}
		return wikimediaTools
	}
	baseline := gopherllm.CaptureRuntimeTuning()
	if opts.BaselineRuntimeTuning != nil {
		baseline = *opts.BaselineRuntimeTuning
	}
	state := &runnerState{r: initialRunner, path: opts.ModelPath, baseline: baseline, autoTune: opts.AppliedAutoTune}
	embedder := &embeddingState{}
	remote := newRemoteState()
	sem := make(chan struct{}, opts.MaxConcurrentRequests)
	var autoTuneMu sync.Mutex
	// Serializes replacement plus its host callback. This prevents two nearly
	// simultaneous hot-swaps from recording the models out of their actual
	// swap order (for example, in a host that persists the active model).
	var modelLoadMu sync.Mutex
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"ok": true, "model": modelID(state.get()), "remote": remote.enabled()})
	})
	mux.HandleFunc("/remote", func(w http.ResponseWriter, req *http.Request) {
		switch req.Method {
		case http.MethodGet:
			writeJSON(w, remote.publicConfig())
		case http.MethodDelete:
			remote.clear()
			writeJSON(w, remote.publicConfig())
		case http.MethodPost:
			var body remoteConfigRequest
			if err := json.NewDecoder(io.LimitReader(req.Body, 64<<10)).Decode(&body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if err := remote.configure(body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, remote.publicConfig())
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/remote/models", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		models, err := remote.listModels(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, models)
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
		state.withRunner(func(r *gopherllm.Runner) {
			model := modelID(r)
			result, err := gopherllm.RunAgenticChatWithTools(r, messages, options, skills, agenticToolsFor(body.Wikimedia), alwaysContinue)
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
		contextMode, err := body.ContextWindowMode()
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		options := body.Options(opts.Defaults)
		options.ContextWindowMode = contextMode
		options = withRequestContext(options, req)
		messages := body.ChatMessages()
		state.withRunner(func(r *gopherllm.Runner) {
			model := modelID(r)
			if contextMode != gopherllm.ContextWindowFull {
				effectiveOptions, _ := gopherllm.AgenticOptionsForTools(options, skills, agenticToolsFor(body.Wikimedia))
				_, _, err := r.PrepareChatContext(messages, effectiveOptions)
				if err != nil {
					logInferenceResult(logw, requestID, "/v1/chat/completions", model, body.Stream, gopherllm.GenerationResult{}, err)
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
			}
			if body.Stream {
				includeUsage := body.StreamOptions != nil && body.StreamOptions.IncludeUsage
				streamOpenAIChat(w, req, logw, requestID, r, model, messages, options, skillsFor(body.SkillsEnabled()), agenticToolsFor(body.Wikimedia), includeUsage)
				return
			}
			var timeline []gopherllm.AgentEvent
			observe := func(e gopherllm.AgentEvent) { timeline = append(timeline, e) }
			result, err := gopherllm.RunAgenticChatObserved(r, messages, options, skillsFor(body.SkillsEnabled()), agenticToolsFor(body.Wikimedia), alwaysContinue, observe)
			logInferenceResult(logw, requestID, "/v1/chat/completions", model, false, result, err)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			response := openAIChatResponse(model, result)
			if result.ContextWindow != nil {
				writeContextWindowHeaders(w, *result.ContextWindow)
				response["gopherllm_context"] = result.ContextWindow
			}
			response = withAgentTimeline(response, timeline)
			writeJSON(w, response)
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
		state.withRunner(func(r *gopherllm.Runner) {
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
		state.withRunner(func(r *gopherllm.Runner) {
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
		if model == "" {
			writeJSON(w, map[string]any{"object": "list", "data": []any{}})
			return
		}
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
		state.withRunner(func(r *gopherllm.Runner) {
			model := modelID(r)
			if streamEnabled(body.Stream) {
				streamOllamaGenerate(w, req, logw, requestID, r, model, body.Prompt, options, skills, agenticToolsFor(body.Wikimedia))
				return
			}
			result, err := gopherllm.RunAgenticChatWithTools(r, []gopherllm.ChatMessage{gopherllm.UserMessage(body.Prompt)}, options, skills, agenticToolsFor(body.Wikimedia), alwaysContinue)
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
		state.withRunner(func(r *gopherllm.Runner) {
			model := modelID(r)
			if streamEnabled(body.Stream) {
				streamOllamaChat(w, req, logw, requestID, r, model, body.ChatMessages(), options, skills, agenticToolsFor(body.Wikimedia))
				return
			}
			result, err := gopherllm.RunAgenticChatWithTools(r, body.ChatMessages(), options, skills, agenticToolsFor(body.Wikimedia), alwaysContinue)
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
		state.withRunner(func(r *gopherllm.Runner) {
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
		state.withRunner(func(r *gopherllm.Runner) {
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
		a := gopherllm.AnalyzeGGUF(r.GGUF(), r.Tokenizer())
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
				"general.architecture":                      a.Architecture,
				"general.parameter_count":                   a.Params,
				a.Architecture + ".context_length":          a.ContextLength,
				a.Architecture + ".embedding_length":        a.Dim,
				a.Architecture + ".block_count":             a.Layers,
				a.Architecture + ".attention.head_count":    a.Heads,
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
			ID            string  `json:"id"`
			Name          string  `json:"name"`
			Path          string  `json:"path"`
			Architecture  string  `json:"architecture"`
			ContextLength int     `json:"context_length"`
			SizeGB        float64 `json:"size_gb"`
			Supported     bool    `json:"supported"`
			Loaded        bool    `json:"loaded"`
			Embedding     bool    `json:"embedding"`
		}
		if opts.ModelDir == "" {
			writeJSON(w, map[string]any{"models": []modelInfo{}})
			return
		}
		entries, err := gopherllm.DiscoverModels(opts.ModelDir, io.Discard)
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
				ID:            e.ID,
				Name:          name,
				Path:          e.Path,
				Architecture:  e.Architecture,
				ContextLength: e.ContextLength,
				SizeGB:        float64(e.SizeBytes) / (1024 * 1024 * 1024),
				Supported:     e.IsSupported,
				Loaded:        e.Path == loadedPath,
				Embedding:     e.IsEmbedding && e.IsSupported && !e.IsProjector,
			})
		}
		writeJSON(w, map[string]any{"models": models})
	})
	mux.HandleFunc("/models/load", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.ModelDir == "" {
			http.Error(w, "model hot-swap is disabled: configure HandlerOptions.ModelDir", http.StatusNotFound)
			return
		}
		var body modelLoadRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		selector := body.selector()
		if selector == "" {
			http.Error(w, "missing model selector", http.StatusBadRequest)
			return
		}
		entries, err := gopherllm.DiscoverModels(opts.ModelDir, io.Discard)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		entry, err := gopherllm.SelectModel(entries, selector)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		newRunner, _, err := gopherllm.RunnerFromPathWithOptions(entry.Path, opts.ModelLoadOptions)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		func() {
			modelLoadMu.Lock()
			defer modelLoadMu.Unlock()
			state.swap(newRunner, entry.Path)
			if opts.ModelLoaded != nil {
				opts.ModelLoaded(entry.Path)
			}
		}()
		writeJSON(w, map[string]any{"ok": true, "id": entry.ID, "model": modelID(newRunner), "context_length": newRunner.Config().MaxSeqLen, "out_of_core": newRunner.OutOfCore()})
	}))
	mux.HandleFunc("/models/embed/load", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.ModelDir == "" {
			http.Error(w, "embedding-model loading is disabled: configure HandlerOptions.ModelDir", http.StatusNotFound)
			return
		}
		var body modelLoadRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		entries, err := gopherllm.DiscoverModels(opts.ModelDir, io.Discard)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		entry, err := gopherllm.SelectModel(entries, body.selector())
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if !entry.IsEmbedding {
			http.Error(w, "selected model is not an embedding model", http.StatusBadRequest)
			return
		}
		runner, _, err := gopherllm.RunnerFromPathWithOptions(entry.Path, opts.ModelLoadOptions)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		embedder.swap(runner)
		writeJSON(w, map[string]any{"ok": true, "id": entry.ID, "model": modelID(runner), "context_length": runner.Config().MaxSeqLen, "out_of_core": runner.OutOfCore()})
	}))
	mux.HandleFunc("/models/embed", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(body.Input) == 0 || len(body.Input) > 256 {
			http.Error(w, "input must contain between 1 and 256 texts", http.StatusBadRequest)
			return
		}
		var vectors [][]float32
		var model string
		var embedErr error
		embedder.withRunner(func(runner *gopherllm.Runner) {
			if runner == nil {
				embedErr = errors.New("no embedding model is loaded")
				return
			}
			model = modelID(runner)
			vectors = make([][]float32, 0, len(body.Input))
			for _, input := range body.Input {
				if strings.TrimSpace(input) == "" {
					embedErr = errors.New("embedding input must not be empty")
					return
				}
				result, err := runner.Embed(input)
				if err != nil {
					embedErr = err
					return
				}
				vectors = append(vectors, result.Embedding)
			}
		})
		if embedErr != nil {
			status := http.StatusBadRequest
			if embedErr.Error() == "no embedding model is loaded" {
				status = http.StatusConflict
			}
			http.Error(w, embedErr.Error(), status)
			return
		}
		writeJSON(w, map[string]any{"model": model, "embeddings": vectors})
	}))
	mux.HandleFunc("/autotune", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, state.autoTuneStatus())
	})
	mux.HandleFunc("/autotune/run", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Effort  string `json:"effort"`
			Refresh bool   `json:"refresh"`
		}
		if req.Body != nil {
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil && err != io.EOF {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		if !autoTuneMu.TryLock() {
			http.Error(w, "auto-tuning is already running", http.StatusConflict)
			return
		}
		defer autoTuneMu.Unlock()
		body.Effort = strings.TrimSpace(body.Effort)
		if !gopherllm.ValidAutoTuneEffort(body.Effort) {
			http.Error(w, "effort must be quick, balanced, or thorough", http.StatusBadRequest)
			return
		}
		runOpts := gopherllm.AutoTuneOptionsForEffort(body.Effort)
		runOpts.LogWriter = logw
		res, cached, err := state.runAutoTune(runOpts, body.Refresh)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(logw, "Auto-tune via web UI: %s\n", res.SettingsLine())
		writeJSON(w, map[string]any{"cached": cached, "result": res})
	}))
	mux.HandleFunc("/agentos/status", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.AgentOS == nil {
			writeJSON(w, map[string]any{"enabled": false})
			return
		}
		writeJSON(w, map[string]any{
			"enabled": true,
			"policy":  string(opts.AgentOS.Policy),
			"allowed": opts.AgentOS.Allowed,
		})
	})
	mux.HandleFunc("/agentos/propose", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.AgentOS == nil {
			http.Error(w, "the agentic OS-command feature is not enabled on this server", http.StatusNotFound)
			return
		}
		var body struct {
			Instruction string `json:"instruction"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		body.Instruction = strings.TrimSpace(body.Instruction)
		if body.Instruction == "" {
			http.Error(w, "instruction must not be empty", http.StatusBadRequest)
			return
		}
		state.withRunner(func(r *gopherllm.Runner) {
			options := opts.Defaults
			options.SystemPrompt = agentos.SystemPrompt
			options.Tools = nil
			options.ToolChoice = "none"
			options = withRequestContext(options, req)
			result, err := r.GenerateChat([]gopherllm.ChatMessage{gopherllm.UserMessage(body.Instruction)}, options)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			proposal, err := agentos.ParseProposal(result.Text)
			if err != nil {
				http.Error(w, "model did not return a usable proposal: "+err.Error(), http.StatusBadGateway)
				return
			}
			decision := opts.AgentOS.Evaluate(proposal)
			response := map[string]any{"proposal": proposal, "decision": decision}
			// AutoRun means the operator's own policy (whitelist/allow), not the
			// model's self-reported "safe" field, already authorizes this — see
			// the agentos package comment on why Evaluate never reads Safe.
			if decision.AutoRun {
				res, dec, execErr := opts.AgentOS.Execute(req.Context(), proposal, false)
				response["decision"] = dec
				if execErr != nil {
					response["error"] = execErr.Error()
				} else {
					response["result"] = res
				}
			}
			writeJSON(w, response)
		})
	}))
	mux.HandleFunc("/agentos/execute", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.AgentOS == nil {
			http.Error(w, "the agentic OS-command feature is not enabled on this server", http.StatusNotFound)
			return
		}
		var body struct {
			Proposal agentos.Proposal `json:"proposal"`
			// Approved is the human's out-of-band decision — a button click in
			// the browser, never anything read from the model's own output. See
			// Runner.Execute's doc comment.
			Approved bool `json:"approved"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		res, dec, err := opts.AgentOS.Execute(req.Context(), body.Proposal, body.Approved)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"decision": dec, "result": res})
	}))
	if opts.ChatUI {
		mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
			if req.URL.Path != "/" {
				http.NotFound(w, req)
				return
			}
			http.Redirect(w, req, "/chat", http.StatusFound)
		})
		mux.HandleFunc("/chat", func(w http.ResponseWriter, req *http.Request) {
			// The diagram renderer is chosen in the browser's Settings, but the
			// CSP that permits it is a response header, so the choice travels
			// back as a query parameter and is validated against the map above.
			mermaid := mermaidChoice(req.URL.Query().Get("mermaid"))
			setChatUIHeaders(w, mermaid)
			w.Header().Set("content-type", "text/html; charset=utf-8")
			model := modelID(state.get())
			if model == "" {
				model = "No model selected"
			}
			data := chatTemplateData{
				Title:         "GopherLLM Chat",
				Model:         model,
				MaxTokens:     opts.Defaults.MaxTokens,
				Temperature:   opts.Defaults.Sampler.Temperature,
				TopP:          opts.Defaults.Sampler.TopP,
				TopK:          opts.Defaults.Sampler.TopK,
				MinP:          opts.Defaults.Sampler.MinP,
				RepeatPenalty: opts.Defaults.Sampler.RepeatPenalty,
				MermaidCDN:    mermaid,
			}
			if cdn, ok := mermaidCDNs[mermaid]; ok {
				data.MermaidScript = cdn.Script
			}
			if err := chatTemplate.Execute(w, data); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		})
		mux.HandleFunc("/style.css", func(w http.ResponseWriter, _ *http.Request) {
			setChatUIHeaders(w, "")
			w.Header().Set("content-type", "text/css; charset=utf-8")
			fmt.Fprint(w, chatCSS)
		})
		mux.HandleFunc("/script.js", func(w http.ResponseWriter, _ *http.Request) {
			setChatUIHeaders(w, "")
			w.Header().Set("content-type", "text/javascript; charset=utf-8")
			fmt.Fprint(w, chatJS)
		})
	}
	return remoteOrLoadedModel(state, remote, mux)
}

// requireLoadedModel keeps catalog, UI and model-loading routes available when
// the server starts without weights. Generation-shaped routes return a clear
// 503 until the user chooses a model instead of dereferencing a nil Runner.
func remoteOrLoadedModel(state *runnerState, remote *remoteState, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/v1/chat/completions" && remote.enabled() {
			remote.proxyChat(w, req)
			return
		}
		if state.get() == nil && needsLoadedModel(req.URL.Path) {
			http.Error(w, "no model is loaded; choose one in the Web UI or POST /models/load", http.StatusServiceUnavailable)
			return
		}
		next.ServeHTTP(w, req)
	})
}

func needsLoadedModel(path string) bool {
	switch path {
	case "/generate", "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/generate", "/api/chat", "/api/embeddings", "/api/embed", "/autotune/run":
		return true
	default:
		return false
	}
}

// setChatUIHeaders keeps the local browser workspace private to this origin:
// chat HTML and its assets are never cached and cannot load third-party code.
// mermaidCDNs are the only origins the chat page may load a diagram renderer
// from. Mermaid is ~2.8 MB, so embedding it would inflate every binary for a
// feature most sessions never use; loading it from a CDN is the alternative,
// and that means punching a hole in the CSP. The hole is kept as small as
// possible: one operator-chosen origin, named here rather than assembled from
// user input, and absent entirely unless a choice was made.
var mermaidCDNs = map[string]struct {
	Origin string
	Script string
}{
	"jsdelivr": {"https://cdn.jsdelivr.net", "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"},
	"unpkg":    {"https://unpkg.com", "https://unpkg.com/mermaid@11/dist/mermaid.min.js"},
	"cdnjs":    {"https://cdnjs.cloudflare.com", "https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.4.1/mermaid.min.js"},
}

// mermaidChoice validates a requested CDN key, returning "" for "no diagrams".
func mermaidChoice(raw string) string {
	key := strings.ToLower(strings.TrimSpace(raw))
	if _, ok := mermaidCDNs[key]; ok {
		return key
	}
	return ""
}

func setChatUIHeaders(w http.ResponseWriter, mermaidCDN string) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Referrer-Policy", "same-origin")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	script, style := "'self'", "'self'"
	if cdn, ok := mermaidCDNs[mermaidCDN]; ok {
		script += " " + cdn.Origin
		// Mermaid styles the SVG it builds with inline <style>, so drawing a
		// diagram needs this. It is scoped to the page that opted in: with no
		// CDN chosen the policy stays strict.
		style += " 'unsafe-inline'"
	}
	w.Header().Set("Content-Security-Policy", "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; img-src 'self' data:; script-src "+script+"; style-src "+style)
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
		out = append(out, gopherllm.ChatMessage{Role: role, Content: contentText(item.Content), ToolCalls: item.ToolCalls, ToolCallID: item.ToolCallID, Name: item.Name})
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
	default:
		return []string{fmt.Sprint(x)}
	}
}

type OllamaGenerateRequest struct {
	Model     string        `json:"model"`
	Prompt    string        `json:"prompt"`
	System    string        `json:"system"`
	Stream    *bool         `json:"stream"`
	Options   OllamaOptions `json:"options"`
	Stop      any           `json:"stop"`
	Wikimedia bool          `json:"gopherllm_wikimedia"`
}

func (o OllamaGenerateRequest) GenerationOptions(def gopherllm.GenerationOptions) gopherllm.GenerationOptions {
	system := (*string)(nil)
	if o.System != "" {
		system = &o.System
	}
	return applyRequestOptions(def, o.Options.NumPredict, o.Options.Temperature, o.Options.TopP, o.Options.TopK, o.Options.MinP, o.Options.RepeatPenalty, o.Options.Seed, system, firstStop(o.Stop, o.Options.Stop), nil, "")
}

type OllamaChatRequest struct {
	Model     string                     `json:"model"`
	Messages  []OllamaMessage            `json:"messages"`
	Stream    *bool                      `json:"stream"`
	Options   OllamaOptions              `json:"options"`
	Tools     []gopherllm.ToolDefinition `json:"tools"`
	Wikimedia bool                       `json:"gopherllm_wikimedia"`
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

func applyRequestOptions(def gopherllm.GenerationOptions, maxTokens *int, temp *float32, topP *float32, topK *int, minP *float32, repeat *float32, seed *uint64, system *string, stop any, tools []gopherllm.ToolDefinition, toolChoice string) gopherllm.GenerationOptions {
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
		select {
		case sem <- struct{}{}:
		case <-r.Context().Done():
			return
		}
		defer func() { <-sem }()
		h(w, r)
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("content-type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// writeContextWindowHeaders exposes the exact rendered prompt accounting to
// the bundled web UI without changing OpenAI's streaming event schema. They
// are only emitted for explicit bounded-context extensions.
func writeContextWindowHeaders(w http.ResponseWriter, info gopherllm.ContextWindowInfo) {
	w.Header().Set("X-GopherLLM-Context-Mode", string(info.Mode))
	w.Header().Set("X-GopherLLM-Context-Length", strconv.Itoa(info.ContextLength))
	w.Header().Set("X-GopherLLM-Context-Budget", strconv.Itoa(info.PromptBudget))
	w.Header().Set("X-GopherLLM-Context-Prompt-Tokens", strconv.Itoa(info.PromptTokens))
	w.Header().Set("X-GopherLLM-Context-Input-Messages", strconv.Itoa(info.InputMessages))
	w.Header().Set("X-GopherLLM-Context-Retained-Messages", strconv.Itoa(info.RetainedMessages))
	w.Header().Set("X-GopherLLM-Context-Dropped-Messages", strconv.Itoa(info.DroppedMessages))
	w.Header().Set("X-GopherLLM-Context-Compressed-Messages", strconv.Itoa(info.CompressedMessages))
}

func ensureRequestID(w http.ResponseWriter, req *http.Request) string {
	id := strings.TrimSpace(req.Header.Get("X-Request-ID"))
	if id == "" {
		id = fmt.Sprintf("gopherllm-%d-%d", time.Now().UnixNano(), inferenceRequestSeq.Add(1))
	}
	w.Header().Set("X-Request-ID", id)
	return id
}

func withRequestContext(options gopherllm.GenerationOptions, req *http.Request) gopherllm.GenerationOptions {
	return options
}

type inferenceLogRecord struct {
	Event              string `json:"event"`
	RequestID          string `json:"request_id"`
	Endpoint           string `json:"endpoint"`
	Provider           string `json:"provider"`
	Model              string `json:"model"`
	Streaming          bool   `json:"streaming"`
	PromptTokens       int    `json:"prompt_tokens"`
	CompletionTokens   int    `json:"completion_tokens"`
	TTFTMS             int64  `json:"ttft_ms"`
	PrefillMS          int64  `json:"prefill_ms"`
	DecodeMS           int64  `json:"decode_ms"`
	TotalMS            int64  `json:"total_ms"`
	TokensPerSecond    string `json:"tokens_per_second"`
	Cache              string `json:"cache"`
	CacheHit           bool   `json:"cache_hit"`
	CachedPromptTokens int    `json:"cached_prompt_tokens"`
	RetryCount         int    `json:"retry_count"`
	FinishReason       string `json:"finish_reason"`
	ErrorType          string `json:"error_type,omitempty"`
	Error              string `json:"error,omitempty"`
}

// Inference logs are deliberately emitted through the existing handler
// LogWriter instead of adding a metrics backend. Bottleneck: TTFT/decode
// regressions were not attributable per endpoint/request. Change: one
// structured JSON line per completed local inference. Effect: usable latency,
// throughput, token, cache, retry, and error dimensions for benchmarks and
// production logs. Risk: small log volume increase. Rollback: pass nil/Discard
// LogWriter or remove this helper call.
func logInferenceResult(logw io.Writer, requestID, endpoint, model string, streaming bool, result gopherllm.GenerationResult, err error) {
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
	cacheMode, cacheHit, cachedPromptTokens := "none", false, 0
	if result.PromptCache != nil {
		cacheMode = result.PromptCache.Mode
		cacheHit = result.PromptCache.Hit
		cachedPromptTokens = result.PromptCache.ReusedTokens
	}
	rec := inferenceLogRecord{
		Event:              "inference",
		RequestID:          requestID,
		Endpoint:           endpoint,
		Provider:           "local",
		Model:              model,
		Streaming:          streaming,
		PromptTokens:       result.Stats.PromptTokens,
		CompletionTokens:   result.Stats.GeneratedTokens,
		TTFTMS:             result.Stats.TTFT.Milliseconds(),
		PrefillMS:          result.Stats.PrefillTime.Milliseconds(),
		DecodeMS:           result.Stats.DecodeTime.Milliseconds(),
		TotalMS:            result.Stats.TotalTime.Milliseconds(),
		TokensPerSecond:    fmt.Sprintf("%.2f", tps),
		Cache:              cacheMode,
		CacheHit:           cacheHit,
		CachedPromptTokens: cachedPromptTokens,
		RetryCount:         0,
		FinishReason:       finishReasonOrDefault(result.FinishReason),
		ErrorType:          errorType,
		Error:              errorText,
	}
	if b, jsonErr := json.Marshal(rec); jsonErr == nil {
		fmt.Fprintln(logw, string(b))
	}
}

// streamOpenAIChat streams a chat completion via SSE. Content deltas flow
// incrementally exactly as before whenever no tool call could possibly be in
// play; once skills or caller tools are active, gopherllm.RunAgenticChat buffers the
// winning turn and calls onToken once with the final, already-classified
// content, so raw tool-call syntax never leaks into a content delta (see
// gopherllm.RunAgenticChat's doc comment). Either way, the connection ends with one
// terminal chunk carrying finish_reason, usage, and tool_calls. <think>
// reasoning is separated into reasoning_content deltas as soon as its tokens
// arrive; the final gopherllm.GenerationResult remains authoritative for tool calls and
// for the buffered agentic path.
func streamOpenAIChat(w http.ResponseWriter, req *http.Request, logw io.Writer, requestID string, r *gopherllm.Runner, model string, messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions, skills []gopherllm.Skill, tools []gopherllm.AgenticTool, includeUsage bool) {
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
	// Soofi Isar's native template opens <think> in the prompt, so its first
	// generated characters are reasoning and the first marker received is the
	// closing tag. Other models emit the opening marker themselves.
	thinkSplitter := gopherllm.NewThinkStreamSplitter(r.Architecture() == "nemotron_h_moe")
	streamedReasoning := false
	emit := func(reasoning bool, text string) bool {
		if text == "" {
			return true
		}
		field := "content"
		if reasoning {
			field = "reasoning_content"
			streamedReasoning = true
		}
		if err := writeOpenAIStreamChunk(w, flusher, id, model, created, map[string]any{field: text}, nil); err != nil {
			streamErr = err
			return false
		}
		return true
	}
	// Tool activity is streamed as it happens, in its own chunk field, so the
	// browser can show a live timeline instead of an unexplained pause. It
	// rides alongside the content deltas rather than replacing them.
	observe := func(event gopherllm.AgentEvent) {
		if streamErr != nil || req.Context().Err() != nil {
			return
		}
		if err := writeOpenAIStreamChunk(w, flusher, id, model, created,
			map[string]any{}, map[string]any{"gopherllm_agent": event}); err != nil {
			streamErr = err
		}
	}
	result, err := gopherllm.RunAgenticChatObserved(r, messages, options, skills, tools, func(text string) bool {
		if ctxErr := req.Context().Err(); ctxErr != nil {
			streamErr = ctxErr
			return false
		}
		return thinkSplitter.Push(text, emit)
	}, observe)
	if streamErr != nil {
		logInferenceResult(logw, requestID, "/v1/chat/completions", model, true, result, streamErr)
		return
	}
	if err != nil {
		logInferenceResult(logw, requestID, "/v1/chat/completions", model, true, result, err)
		if errors.Is(err, gopherllm.ErrGenerationCanceled) {
			return
		}
		writeSSE(w, flusher, "error", map[string]string{"error": err.Error()})
		return
	}
	if !thinkSplitter.Flush(emit) {
		logInferenceResult(logw, requestID, "/v1/chat/completions", model, true, result, streamErr)
		return
	}
	logInferenceResult(logw, requestID, "/v1/chat/completions", model, true, result, nil)
	finalDelta := map[string]any{}
	if result.ReasoningText != "" && !streamedReasoning {
		finalDelta["reasoning_content"] = result.ReasoningText
	}
	if len(result.ToolCalls) > 0 {
		finalDelta["tool_calls"] = result.ToolCalls
	}
	extra := map[string]any{"finish_reason": finishReasonOrDefault(result.FinishReason)}
	if includeUsage {
		extra["usage"] = usage(result)
	}
	// Headers are already committed once an SSE stream begins. Put the final
	// loop iteration's context accounting in the terminal choice instead, so a
	// skill/tool loop cannot leave the UI reporting the initial prompt as if it
	// were the final one.
	if result.ContextWindow != nil {
		extra["gopherllm_context"] = result.ContextWindow
	}
	if result.PromptCache != nil {
		extra["gopherllm_cache"] = result.PromptCache
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

// ollamaDurations reports gopherllm.GenerationStats using Ollama's nanosecond duration
// field names. load_duration is always 0: GopherLLM has no separate
// model-load phase inside a request (the model is already resident).
func ollamaDurations(stats gopherllm.GenerationStats) map[string]any {
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
func streamOllamaGenerate(w http.ResponseWriter, req *http.Request, logw io.Writer, requestID string, r *gopherllm.Runner, model, prompt string, options gopherllm.GenerationOptions, skills []gopherllm.Skill, tools []gopherllm.AgenticTool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("content-type", "application/x-ndjson")

	var streamErr error
	result, err := gopherllm.RunAgenticChatWithTools(r, []gopherllm.ChatMessage{gopherllm.UserMessage(prompt)}, options, skills, tools, func(text string) bool {
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
		if errors.Is(err, gopherllm.ErrGenerationCanceled) {
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
func streamOllamaChat(w http.ResponseWriter, req *http.Request, logw io.Writer, requestID string, r *gopherllm.Runner, model string, messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions, skills []gopherllm.Skill, tools []gopherllm.AgenticTool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("content-type", "application/x-ndjson")

	var streamErr error
	result, err := gopherllm.RunAgenticChatWithTools(r, messages, options, skills, tools, func(text string) bool {
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
		if errors.Is(err, gopherllm.ErrGenerationCanceled) {
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
// bytes touched — the same cheap path gopherllm.DiscoverModels uses) into an gopherllm.Analysis,
// for building Ollama-shaped model metadata without loading a full gopherllm.Runner.
func analyzeModelFile(path string) (*gopherllm.Analysis, error) {
	mmap, err := gopherllm.OpenMmap(path)
	if err != nil {
		return nil, err
	}
	defer mmap.Close()
	gguf, err := gopherllm.ParseGGUFQuiet(mmap.Bytes())
	if err != nil {
		return nil, err
	}
	return gopherllm.AnalyzeGGUF(gguf, nil), nil
}

// resolveModelAnalysis answers /api/show's "which model": empty/matching
// name means the currently loaded gopherllm.Runner (full gopherllm.Analysis, tokenizer
// included); any other name is looked up in ModelDir (if configured) and
// header-analyzed on demand.
func resolveModelAnalysis(state *runnerState, modelDir, requested string) (*gopherllm.Analysis, bool) {
	r := state.get()
	if r != nil && (requested == "" || requested == modelID(r)) {
		return gopherllm.AnalyzeGGUF(r.GGUF(), r.Tokenizer()), true
	}
	if modelDir == "" {
		return nil, false
	}
	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
	if err != nil {
		return nil, false
	}
	entry, err := gopherllm.SelectModel(entries, requested)
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
		a := gopherllm.AnalyzeGGUF(r.GGUF(), r.Tokenizer())
		return []map[string]any{ollamaTagEntry(name, state.getPath(), a.FileBytes, a)}
	}
	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
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

func ollamaTagEntry(name, path string, sizeBytes int64, a *gopherllm.Analysis) map[string]any {
	return map[string]any{
		"name":        name,
		"model":       name,
		"modified_at": time.Now().Format(time.RFC3339Nano),
		"size":        sizeBytes,
		"digest":      modelDigest(path),
		"details":     ollamaModelDetails(a),
	}
}

// ollamaModelDetails builds Ollama's "details" object from a header gopherllm.Analysis.
func ollamaModelDetails(a *gopherllm.Analysis) map[string]any {
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

func generateResponse(result gopherllm.GenerationResult) map[string]any {
	resp := map[string]any{"text": result.Text, "prompt_tokens": result.Stats.PromptTokens, "generated_tokens": result.Stats.GeneratedTokens, "ttft_ms": result.Stats.TTFT.Milliseconds(), "prefill_ms": result.Stats.PrefillTime.Milliseconds(), "decode_ms": result.Stats.DecodeTime.Milliseconds(), "total_ms": result.Stats.TotalTime.Milliseconds(), "finish_reason": finishReasonOrDefault(result.FinishReason)}
	if result.ReasoningText != "" {
		resp["reasoning"] = result.ReasoningText
	}
	if len(result.ToolCalls) > 0 {
		resp["tool_calls"] = result.ToolCalls
	}
	if result.PromptCache != nil {
		resp["gopherllm_cache"] = result.PromptCache
	}
	return resp
}

func openAIChatResponse(model string, result gopherllm.GenerationResult) map[string]any {
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
	response := map[string]any{"id": "chatcmpl-gopherllm", "object": "chat.completion", "created": time.Now().Unix(), "model": model, "system_fingerprint": systemFingerprint, "choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finishReasonOrDefault(result.FinishReason)}}, "usage": usage(result)}
	if result.PromptCache != nil {
		response["gopherllm_cache"] = result.PromptCache
	}
	return response
}

// withAgentTimeline attaches the turn's tool activity to a non-streaming
// response, in the same shape (kind/iteration/tool/result/duration_ms) the
// streaming path already sends per-chunk as gopherllm_agent — so a client
// gets the same visibility into what ran and how long it took, regardless of
// whether it asked for a stream. A nil or empty timeline (the common,
// tool-free case) leaves the response exactly as it was.
func withAgentTimeline(response map[string]any, timeline []gopherllm.AgentEvent) map[string]any {
	if len(timeline) > 0 {
		response["gopherllm_agent"] = timeline
	}
	return response
}

// finishReasonOrDefault falls back to "stop" for callers of GenerateResult
// that predate FinishReason (in-tree, only gopherllm.GenerationResult zero values hit
// this) so every response always carries a valid OpenAI-shaped finish_reason.
func finishReasonOrDefault(reason string) string {
	if reason == "" {
		return "stop"
	}
	return reason
}

func usage(result gopherllm.GenerationResult) map[string]int {
	return map[string]int{"prompt_tokens": result.Stats.PromptTokens, "completion_tokens": result.Stats.GeneratedTokens, "total_tokens": result.Stats.PromptTokens + result.Stats.GeneratedTokens}
}

func modelID(r *gopherllm.Runner) string {
	if r == nil {
		return ""
	}
	if name, ok := r.ModelName(); ok && name != "" {
		return name
	}
	return "gopherllm"
}
