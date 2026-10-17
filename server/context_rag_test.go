package server

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func contextRAGTestHandler(t *testing.T) *Handler {
	t.Helper()
	r, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUFWithContext(4096))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	defaults := gopherllm.DefaultGenerationOptions()
	defaults.SystemPrompt, defaults.MaxTokens = "", 1
	defaults.Sampler.Temperature, defaults.Sampler.TopK = 0, 1
	h := NewHandler(r, HandlerOptions{Defaults: defaults})
	t.Cleanup(func() { h.Close() })
	return h
}

func contextRAGTestMessages(topic string) []map[string]any {
	return []map[string]any{
		{"role": "user", "content": topic},
		{"role": "assistant", "content": "b"},
		{"role": "user", "content": strings.Repeat("f", 5000)},
		{"role": "assistant", "content": "g"},
		{"role": "user", "content": "cedar"},
	}
}

func postContextRAG(t *testing.T, h http.Handler, body map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(data))))
	return w
}

func contextRAGMetadata(t *testing.T, w *httptest.ResponseRecorder, stream bool) map[string]any {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d: %s", w.Code, w.Body.String())
	}
	if !stream {
		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		info, _ := body["gopherllm_context"].(map[string]any)
		return info
	}
	var info map[string]any
	scanner := bufio.NewScanner(strings.NewReader(w.Body.String()))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var payload struct {
			Choices []struct {
				Context map[string]any `json:"gopherllm_context"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err != nil {
			t.Fatal(err)
		}
		for _, choice := range payload.Choices {
			if choice.Context != nil {
				info = choice.Context
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return info
}

func TestOpenAIContextRAGOptInAndFinalMetadata(t *testing.T) {
	h := contextRAGTestHandler(t)
	for _, stream := range []bool{false, true} {
		body := map[string]any{"messages": contextRAGTestMessages("cedar"), "gopherllm_context_mode": "recent", "max_tokens": 1, "stream": stream}
		plain := contextRAGMetadata(t, postContextRAG(t, h, body), stream)
		if _, present := plain["recall"]; present {
			t.Fatal("ordinary context planning unexpectedly enabled retrieval")
		}
		body["gopherllm_context_rag"] = true
		info := contextRAGMetadata(t, postContextRAG(t, h, body), stream)
		recall, _ := info["recall"].(map[string]any)
		if recall == nil || recall["mode"] != "bm25" {
			t.Fatalf("stream=%v missing recall metadata: %+v", stream, info)
		}
		added, _ := recall["added_tokens"].(float64)
		budget, _ := recall["token_budget"].(float64)
		if added <= 0 || added > budget || info["prompt_tokens"].(float64) > info["prompt_budget"].(float64) {
			t.Fatalf("retrieval exceeded exact token budget: %+v", info)
		}
		sources, _ := recall["sources"].([]any)
		if len(sources) != 1 {
			t.Fatalf("source count=%d: %+v", len(sources), recall)
		}
		source := sources[0].(map[string]any)
		if source["start_message"] != float64(0) || source["end_message"] != float64(2) || source["hash"] == "" || source["id"] == "" {
			t.Fatalf("source does not identify a complete original turn: %+v", source)
		}
	}
}

func TestOpenAIContextRAGRequestAndToolIsolation(t *testing.T) {
	h := contextRAGTestHandler(t)
	body := map[string]any{"messages": contextRAGTestMessages("cedar"), "gopherllm_context_mode": "recent", "gopherllm_context_rag": true, "max_tokens": 1}
	info := contextRAGMetadata(t, postContextRAG(t, h, body), false)
	if info["recall"] == nil {
		t.Fatal("fixture did not retrieve its own older turn")
	}
	for _, messages := range [][]map[string]any{
		contextRAGTestMessages("birch"),        // Edited branch cannot read the earlier request's cedar turn.
		{{"role": "user", "content": "cedar"}}, // Reset conversation.
		{
			{"role": "user", "content": "cedar"},
			{"role": "assistant", "content": "", "tool_calls": []map[string]any{{"id": "old", "type": "function", "function": map[string]any{"name": "past_action", "arguments": "{}"}}}},
			{"role": "tool", "tool_call_id": "old", "content": "cedar"},
			{"role": "assistant", "content": "b"},
			{"role": "user", "content": strings.Repeat("f", 5000)},
			{"role": "assistant", "content": "g"},
			{"role": "user", "content": "cedar"},
		},
	} {
		body["messages"] = messages
		info := contextRAGMetadata(t, postContextRAG(t, h, body), false)
		if info["recall"] != nil {
			t.Fatalf("retrieval crossed a request boundary or replayed a historical tool turn: %+v", info)
		}
	}
}

func TestOpenAIContextRAGRejectsFullAndPreflightsStream(t *testing.T) {
	h := contextRAGTestHandler(t)
	for _, mode := range []string{"", "full"} {
		body := map[string]any{"messages": []map[string]string{{"role": "user", "content": "a"}}, "gopherllm_context_rag": true}
		if mode != "" {
			body["gopherllm_context_mode"] = mode
		}
		w := postContextRAG(t, h, body)
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "gopherllm_context_rag") {
			t.Fatalf("full+retrieval status=%d: %s", w.Code, w.Body.String())
		}
	}
	w := postContextRAG(t, h, map[string]any{"messages": []map[string]string{{"role": "user", "content": strings.Repeat("f", 5000)}}, "gopherllm_context_mode": "recent", "gopherllm_context_rag": true, "stream": true})
	if w.Code != http.StatusBadRequest || strings.Contains(w.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("oversized latest turn committed SSE: status=%d headers=%+v", w.Code, w.Header())
	}
}

func TestRemoteContextRAGRejectedBeforeForwarding(t *testing.T) {
	var calls atomic.Int32
	remote := newRemoteState()
	remote.client = &http.Client{Transport: contextRAGRoundTripper(func(req *http.Request) (*http.Response, error) {
		calls.Add(1)
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"choices":[]}`))}, nil
	})}
	if err := remote.configure(remoteConfigRequest{BaseURL: "https://remote.example.test"}); err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	remote.proxyChat(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[],"gopherllm_context_rag":true}`)))
	if w.Code != http.StatusBadRequest || calls.Load() != 0 || !strings.Contains(w.Body.String(), "gopherllm_context_rag") {
		t.Fatalf("remote retrieval reached upstream: status=%d calls=%d", w.Code, calls.Load())
	}
	w = httptest.NewRecorder()
	remote.proxyChat(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[]}`)))
	if w.Code != http.StatusOK || calls.Load() != 1 {
		t.Fatal("normal remote forwarding changed")
	}
}

type contextRAGRoundTripper func(*http.Request) (*http.Response, error)

func (f contextRAGRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func TestBrowserContextRAGRemainsUnavailable(t *testing.T) {
	h := NewHandler(nil, HandlerOptions{DeploymentMode: DeploymentBrowser})
	t.Cleanup(func() { h.Close() })
	w := postContextRAG(t, h, map[string]any{"messages": []map[string]string{{"role": "user", "content": "a"}}, "gopherllm_context_mode": "recent", "gopherllm_context_rag": true})
	if w.Code != http.StatusNotFound || !strings.Contains(w.Body.String(), "browser deployment") {
		t.Fatalf("browser deployment allowed server retrieval: status=%d body=%s", w.Code, w.Body.String())
	}
}

func TestContextRAGStructuredStreamPreservesFinalMetadata(t *testing.T) {
	result := gopherllm.GenerationResult{
		Text: `{"answer":"cedar"}`, FinishReason: "stop",
		ContextWindow: &gopherllm.ContextWindowInfo{
			Mode: gopherllm.ContextWindowRecent, PromptBudget: 800, PromptTokens: 750,
			Recall: &gopherllm.ContextRecallInfo{
				Mode: "bm25", IndexedTurns: 2, TokenBudget: 100, AddedTokens: 80,
				Sources: []gopherllm.ContextRecallSource{{ID: "history:1", Turn: 1, StartMessage: 0, EndMessage: 2, Hash: "935401a46a10", Score: 1}},
			},
		},
		PromptCache: &gopherllm.PromptCacheInfo{Mode: "prefix", Hit: true, ReusedTokens: 400, PromptTokens: 750},
	}
	w := httptest.NewRecorder()
	streamValidatedChat(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), "fixture", result, true)
	info := contextRAGMetadata(t, w, true)
	if info == nil || info["recall"].(map[string]any)["added_tokens"] != float64(80) {
		t.Fatalf("structured stream lost final recall metadata: %+v", info)
	}
	var terminal map[string]any
	var usagePresent bool
	for _, line := range strings.Split(w.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var payload map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err != nil {
			t.Fatal(err)
		}
		if payload["usage"] != nil {
			usagePresent = true
		}
		for _, raw := range payload["choices"].([]any) {
			choice := raw.(map[string]any)
			if choice["finish_reason"] != nil {
				terminal = choice
			}
		}
	}
	cache, _ := terminal["gopherllm_cache"].(map[string]any)
	if terminal["finish_reason"] != "stop" || cache["reused_tokens"] != float64(400) || !usagePresent || !strings.HasSuffix(w.Body.String(), "data: [DONE]\n\n") {
		t.Fatalf("structured terminal metadata/usage incomplete: %s", w.Body.String())
	}
}
