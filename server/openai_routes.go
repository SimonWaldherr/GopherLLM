package server

import (
	"encoding/json"
	"io"
	"net/http"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// registerOpenAIRoutes registers the OpenAI-compatible endpoints: /generate
// (GopherLLM's native shape), /v1/chat/completions, /v1/completions,
// /v1/embeddings, /v1/models, and /v1/skills. Extracted from NewHandler's
// inline handlers for these routes.
func registerOpenAIRoutes(mux *http.ServeMux, state *runnerState, embedder *embeddingState, sem chan struct{}, opts HandlerOptions, skills []gopherllm.Skill, skillsFor func(bool) []gopherllm.Skill, agenticToolsFor func(wikimedia, openStreetMap, ragSearch bool) []gopherllm.AgenticTool, logw io.Writer) {
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
			result, err := gopherllm.RunAgenticChatWithTools(r, messages, options, skills, agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG), alwaysContinue)
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
			// Only a stream needs this pre-flight. Headers are committed with an
			// SSE stream's first chunk, so a context error discovered during
			// generation could no longer be reported as a 4xx — it has to be
			// caught before anything is written. The non-streaming path below
			// has no such constraint: RunAgenticChatObserved surfaces the very
			// same error and turns it into the very same 400, so preparing the
			// context here would only do the work twice. That is not a
			// rounding error. Preparation re-renders and re-tokenizes the
			// retained history once per candidate turn boundary, which is
			// quadratic in conversation length: measured against the real
			// Ministral-3 3B, a 128-message history in "recent" mode costs
			// ~608 ms and a 256-message one ~2.3 s, all of it before the first
			// token. Dropping the duplicate cut a non-streaming request's
			// allocations by 48%.
			if body.Stream && contextMode != gopherllm.ContextWindowFull {
				effectiveOptions, _ := gopherllm.AgenticOptionsForTools(options, skills, agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG))
				_, _, err := r.PrepareChatContext(messages, effectiveOptions)
				if err != nil {
					logInferenceResult(logw, requestID, "/v1/chat/completions", model, body.Stream, gopherllm.GenerationResult{}, err)
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
			}
			if body.Stream {
				includeUsage := body.StreamOptions != nil && body.StreamOptions.IncludeUsage
				streamOpenAIChat(w, req, logw, requestID, r, model, messages, options, skillsFor(body.SkillsEnabled()), agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG), includeUsage)
				return
			}
			var timeline []gopherllm.AgentEvent
			observe := func(e gopherllm.AgentEvent) { timeline = append(timeline, e) }
			result, err := gopherllm.RunAgenticChatObserved(r, messages, options, skillsFor(body.SkillsEnabled()), agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG), alwaysContinue, observe)
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
		var vectors [][]float32
		var total int
		var embedErr error
		var model string
		withEmbeddingRunner(state, embedder, func(r *gopherllm.Runner) {
			model = modelID(r)
			vectors, total, embedErr = embedTexts(r, inputs)
		})
		if embedErr != nil {
			http.Error(w, embedErr.Error(), http.StatusBadRequest)
			return
		}
		data := make([]any, len(vectors))
		for i, vector := range vectors {
			data[i] = map[string]any{"object": "embedding", "index": i, "embedding": vector}
		}
		writeJSON(w, map[string]any{"object": "list", "model": model, "data": data, "usage": map[string]int{"prompt_tokens": total, "total_tokens": total}})
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
}
