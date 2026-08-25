package server

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

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

func usage(result gopherllm.GenerationResult) map[string]any {
	cachedTokens := 0
	if result.PromptCache != nil {
		cachedTokens = min(max(result.PromptCache.ReusedTokens, 0), result.Stats.PromptTokens)
	}
	return map[string]any{
		"prompt_tokens":     result.Stats.PromptTokens,
		"completion_tokens": result.Stats.GeneratedTokens,
		"total_tokens":      result.Stats.PromptTokens + result.Stats.GeneratedTokens,
		"prompt_tokens_details": map[string]int{
			"cached_tokens": cachedTokens,
		},
	}
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
