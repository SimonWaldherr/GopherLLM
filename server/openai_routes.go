package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/chatmemory"
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
			writeAPIError(w, 400, "invalid_request", "", err.Error())
			return
		}
		messages, options := body.ToMessagesAndOptions(opts.Defaults)
		options = withRequestContext(options, req)
		if err := state.withRunnerContext(req.Context(), func(r *gopherllm.Runner) {
			model := modelID(r)
			result, err := gopherllm.RunAgenticChatWithTools(r, messages, options, skills, agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG), alwaysContinue)
			logInferenceResult(logw, requestID, "/generate", model, false, result, err)
			if err != nil {
				writeAPIError(w, 400, "invalid_request", "", err.Error())
				return
			}
			writeJSON(w, generateResponse(result))
		}); err != nil {
			inferenceAPIError(w, err)
		}
	}))
	mux.HandleFunc("/v1/chat/completions", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		requestID := ensureRequestID(w, req)
		var body OpenAIChatRequest
		if !decodeAPIRequest(w, req, &body) {
			return
		}
		schema, err := body.validateContract()
		if err != nil {
			writeAPIError(w, 400, "unsupported_request", "", err.Error())
			return
		}
		contextMode, err := body.ContextWindowMode()
		if err != nil {
			writeAPIError(w, 400, "invalid_request", "", err.Error())
			return
		}
		if body.GopherLLMContextRAG && contextMode == gopherllm.ContextWindowFull {
			writeAPIError(w, 400, "invalid_parameter", "gopherllm_context_rag", "context retrieval requires recent or autoCompress context mode")
			return
		}
		options := body.Options(opts.Defaults)
		options.ContextWindowMode = contextMode
		if body.ResponseFormat != nil && body.ResponseFormat.Type == "text" {
			options.JSONObject = false
		}
		structured := options.JSONObject || body.ResponseFormat != nil && body.ResponseFormat.Type != "text"
		if structured {
			options.JSONObject = true
			options.SystemPrompt += "\nRespond with one JSON object only."
			if schema != nil {
				options.SystemPrompt += "\nRequired JSON schema: " + string(body.ResponseFormat.JSONSchema.Schema)
			}
		}
		if err := options.Validate(); err != nil {
			logInferenceResult(logw, requestID, req.URL.Path, "", body.Stream, gopherllm.GenerationResult{}, err)
			writeAPIError(w, 400, "invalid_parameter", "", err.Error())
			return
		}
		options = withRequestContext(options, req)
		messages := body.ChatMessages()
		selectedSkills := skillsFor(body.SkillsEnabled())
		if structured || len(body.Tools) > 0 {
			selectedSkills = nil
		}
		if err := state.withRunnerContext(req.Context(), func(r *gopherllm.Runner) {
			model := modelID(r)
			if !validateModel(w, body.Model, model) {
				return
			}
			tools := agenticToolsFor(body.Wikimedia, body.OpenStreetMap, body.RAG)
			generate := gopherllm.ChatGenerator(r.GenerateChatStreamUntil)
			var memory *chatmemory.Session
			if body.GopherLLMContextRAG {
				var err error
				memory, err = chatmemory.New(req.Context(), r, chatmemory.Options{})
				if err != nil {
					inferenceAPIError(w, err)
					return
				}
				// This wrapper prepares each effective agent-loop prompt, after
				// new tool results and the current tool schemas have been added.
				generate = memory.Wrap(generate)
			}
			// Catch bounded-context errors before committing SSE headers, then
			// consume this exact request-local plan once instead of repeating the
			// suffix search/tokenization. Tool rounds use the ordinary generator
			// (including recall when enabled) and prepare their changed prompt.
			if body.Stream && contextMode != gopherllm.ContextWindowFull {
				effectiveOptions, _ := gopherllm.AgenticOptionsForTools(options, selectedSkills, tools)
				var prepare gopherllm.ChatContextPreparer
				if memory != nil {
					prepare = memory.Prepare
				}
				plan, err := r.PrepareChatGeneration(messages, effectiveOptions, prepare)
				if err != nil {
					logInferenceResult(logw, requestID, "/v1/chat/completions", model, body.Stream, gopherllm.GenerationResult{}, err)
					writeAPIError(w, 400, "invalid_request", "", err.Error())
					return
				}
				generate = plan.Wrap(generate)
			}
			if body.Stream && !structured {
				includeUsage := body.StreamOptions != nil && body.StreamOptions.IncludeUsage
				streamOpenAIChat(w, req, logw, requestID, r, model, messages, options, selectedSkills, tools, includeUsage, generate)
				return
			}
			var timeline []gopherllm.AgentEvent
			observe := func(e gopherllm.AgentEvent) { timeline = append(timeline, e) }
			result, err := gopherllm.RunAgenticChatWithGenerator(generate, messages, options, selectedSkills, tools, alwaysContinue, observe)
			if err == nil {
				if e := validateChatResult(result, options); e != nil {
					err = fmt.Errorf("%w: %v", errOutputContract, e)
				}
			}
			if err == nil && schema != nil {
				if e := schema.Validate([]byte(result.Text)); e != nil {
					err = fmt.Errorf("%w: %v", errSchemaOutput, e)
				}
			}
			logInferenceResult(logw, requestID, "/v1/chat/completions", model, false, result, err)
			if err != nil {
				inferenceAPIError(w, err)
				return
			}
			if body.Stream {
				streamValidatedChat(w, req, model, result, body.StreamOptions != nil && body.StreamOptions.IncludeUsage)
				return
			}
			response := openAIChatResponse(model, result)
			if result.ContextWindow != nil {
				writeContextWindowHeaders(w, *result.ContextWindow)
				response["gopherllm_context"] = result.ContextWindow
			}
			response = withAgentTimeline(response, timeline)
			writeJSON(w, response)
		}); err != nil {
			inferenceAPIError(w, err)
		}
	}))
	mux.HandleFunc("/v1/completions", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		requestID := ensureRequestID(w, req)
		var body OpenAICompletionRequest
		if !decodeAPIRequest(w, req, &body) {
			return
		}
		if body.MaxTokens != nil && body.MaxCompletionTokens != nil {
			writeAPIError(w, 400, "invalid_parameter", "max_tokens", "use only one token limit")
			return
		}
		if err := validateStop(body.Stop); err != nil {
			writeAPIError(w, 400, "invalid_parameter", "stop", err.Error())
			return
		}
		if _, ok := body.Prompt.(string); !ok {
			writeAPIError(w, 400, "invalid_prompt", "prompt", "only a single string prompt is supported")
			return
		}
		options := body.Options(opts.Defaults)
		if err := options.Validate(); err != nil {
			writeAPIError(w, 400, "invalid_parameter", "", err.Error())
			return
		}
		options = withRequestContext(options, req)
		if err := state.withRunnerContext(req.Context(), func(r *gopherllm.Runner) {
			model := modelID(r)
			if !validateModel(w, body.Model, model) {
				return
			}
			result, err := r.Generate(body.PromptString(), options)
			logInferenceResult(logw, requestID, "/v1/completions", model, false, result, err)
			if err != nil {
				inferenceAPIError(w, err)
				return
			}
			writeJSON(w, map[string]any{"id": newCompletionID("cmpl"), "object": "text_completion", "model": model, "choices": []any{map[string]any{"index": 0, "text": result.Text, "finish_reason": finishReasonOrDefault(result.FinishReason)}}, "usage": usage(result)})
		}); err != nil {
			inferenceAPIError(w, err)
		}
	}))
	mux.HandleFunc("/v1/embeddings", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		var body EmbeddingsRequest
		if !decodeAPIRequest(w, req, &body) {
			return
		}
		inputs, err := validatedEmbeddingInputs(body)
		if err != nil {
			writeAPIError(w, 400, "invalid_input", "input", err.Error())
			return
		}
		var vectors [][]float32
		var total int
		var embedErr error
		var model string
		if err := withEmbeddingRunnerContext(req.Context(), state, embedder, func(r *gopherllm.Runner) {
			model = modelID(r)
			if !validateModel(w, body.Model, model) {
				embedErr = errResponseWritten
				return
			}
			if body.Dimensions != nil && *body.Dimensions != r.Config().Dim {
				embedErr = fmt.Errorf("dimensions must equal model hidden size %d", r.Config().Dim)
				return
			}
			var results []gopherllm.EmbeddingResult
			results, embedErr = r.EmbedBatch(req.Context(), inputs)
			if embedErr == nil && len(results) != len(inputs) {
				embedErr = fmt.Errorf("%w: embedding batch size mismatch", errOutputContract)
			}
			for _, result := range results {
				if e := validateEmbeddingVector(result.Embedding, r.Config().Dim); e != nil {
					embedErr = fmt.Errorf("%w: %v", errOutputContract, e)
					break
				}
				vectors = append(vectors, result.Embedding)
				total += result.TokenCount
			}
		}); err != nil {
			inferenceAPIError(w, err)
			return
		}
		if embedErr == errResponseWritten {
			return
		}
		if embedErr != nil {
			inferenceAPIError(w, embedErr)
			return
		}
		data := make([]any, len(vectors))
		for i, vector := range vectors {
			data[i] = map[string]any{"object": "embedding", "index": i, "embedding": encodeEmbedding(vector, body.EncodingFormat)}
		}
		writeJSON(w, map[string]any{"object": "list", "model": model, "data": data, "usage": map[string]int{"prompt_tokens": total, "total_tokens": total}})
	}))
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			writeAPIError(w, 405, "method_not_allowed", "", "GET required")
			return
		}
		models := []any{}
		seen := map[string]bool{}
		add := func(r *gopherllm.Runner) {
			id := modelID(r)
			if id != "" && !seen[id] {
				models = append(models, map[string]any{"id": id, "object": "model", "created": 0, "owned_by": "gopherllm"})
				seen[id] = true
			}
		}
		state.withRunner(add)
		embedder.withRunner(add)
		if opts.DecisionModel != nil && opts.DeploymentMode != DeploymentBrowser {
			id := opts.DecisionModelID
			if id == "" {
				id = "laya-rl-agent"
			}
			if !seen[id] {
				models = append(models, map[string]any{"id": id, "object": "model", "created": 0, "owned_by": "gopherllm", "capabilities": []string{"classification"}})
			}
		}
		writeJSON(w, map[string]any{"object": "list", "data": models})
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
