package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type reasoningStreamSplitter interface {
	Push(string, func(reasoning bool, text string) bool) bool
	Flush(func(reasoning bool, text string) bool) bool
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
	// Soofi Isar and Qwen3.5/3.6's native templates open <think> in the
	// prompt, so their first generated characters are reasoning and the first
	// marker received is the closing tag. Other models emit the opening marker
	// themselves.
	arch := r.Architecture()
	var thinkSplitter reasoningStreamSplitter
	if arch == "mistral3" {
		// Ministral-3-Reasoning uses the native [THINK] protocol recorded in
		// its GGUF template, rather than the <think> protocol below.
		splitter := gopherllm.NewMistralThinkStreamSplitter()
		thinkSplitter = &splitter
	} else {
		startsInReasoning := arch == "nemotron_h_moe" || arch == "qwen35" || arch == "qwen35moe"
		splitter := gopherllm.NewThinkStreamSplitter(startsInReasoning)
		thinkSplitter = &splitter
	}
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
	if includeUsage {
		_ = writeOpenAIUsageChunk(w, flusher, id, model, created, usage(result))
	}
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

// OpenAI emits requested streaming usage as a final, top-level usage object
// with no choices. Keeping it separate from the terminal finish-reason chunk
// lets compatible SDKs account for cached input tokens without understanding
// GopherLLM's choice-level extensions.
func writeOpenAIUsageChunk(w http.ResponseWriter, flusher http.Flusher, id, model string, created int64, usage map[string]any) error {
	return writeSSE(w, flusher, "", map[string]any{
		"id":                 id,
		"object":             "chat.completion.chunk",
		"created":            created,
		"model":              model,
		"system_fingerprint": systemFingerprint,
		"choices":            []any{},
		"usage":              usage,
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
