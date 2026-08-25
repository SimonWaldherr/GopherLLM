package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

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
