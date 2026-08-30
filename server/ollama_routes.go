package server

import (
	"encoding/json"
	"io"
	"net/http"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// registerOllamaRoutes registers the Ollama-compatible endpoints:
// /api/generate, /api/chat, /api/embeddings, /api/embed, /api/tags, /api/ps,
// /api/show, and /api/version. Extracted from NewHandler's inline handlers
// for these routes.
func registerOllamaRoutes(mux *http.ServeMux, state *runnerState, embedder *embeddingState, sem chan struct{}, opts HandlerOptions, skills []gopherllm.Skill, agenticToolsFor func(wikimedia, openStreetMap, ragSearch bool) []gopherllm.AgenticTool, logw io.Writer) {
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
				streamOllamaGenerate(w, req, logw, requestID, r, model, body.Prompt, options, skills, agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG))
				return
			}
			result, err := gopherllm.RunAgenticChatWithTools(r, []gopherllm.ChatMessage{gopherllm.UserMessage(body.Prompt)}, options, skills, agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG), alwaysContinue)
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
				streamOllamaChat(w, req, logw, requestID, r, model, body.ChatMessages(), options, skills, agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG))
				return
			}
			result, err := gopherllm.RunAgenticChatWithTools(r, body.ChatMessages(), options, skills, agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG), alwaysContinue)
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
		var vectors [][]float32
		var embedErr error
		withEmbeddingRunner(state, embedder, func(r *gopherllm.Runner) {
			vectors, _, embedErr = embedTexts(r, []string{text})
		})
		if embedErr != nil {
			http.Error(w, embedErr.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"embedding": vectors[0]})
	}))
	mux.HandleFunc("/api/embed", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body OllamaEmbedRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		inputs := body.Inputs()
		var embeddings [][]float32
		var promptTokens int
		var embedErr error
		var model string
		withEmbeddingRunner(state, embedder, func(r *gopherllm.Runner) {
			model = modelID(r)
			embeddings, promptTokens, embedErr = embedTexts(r, inputs)
		})
		if embedErr != nil {
			http.Error(w, embedErr.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"model": model, "embeddings": embeddings, "prompt_eval_count": promptTokens})
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
}
