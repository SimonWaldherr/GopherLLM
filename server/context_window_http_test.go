package server

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"

	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAIRecentContextIsOptIn(t *testing.T) {
	opts := gopherllm.DefaultGenerationOptions()
	opts.MaxTokens = 1
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	system := gopherllm.ChatMessage{Role: gopherllm.ChatRoleSystem, Content: "c"}
	latest := gopherllm.UserMessage("dddd")

	// Size the model's declared context so that [system, latest] fits exactly
	// and the full four-message history does not. Measured through the
	// exported context-window API and then baked into a rebuilt GGUF, because
	// the Runner's context length is not settable from outside the package.
	probe, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	probeOpts := opts
	probeOpts.ContextWindowMode = gopherllm.ContextWindowRecent // budget formula only applies in recent mode
	_, info, err := probe.PrepareChatContext([]gopherllm.ChatMessage{system, latest}, probeOpts)
	probe.Close()
	if err != nil {
		t.Fatal(err)
	}
	safety := info.ContextLength - info.PromptBudget - max(1, opts.MaxTokens)
	limit := info.PromptTokens + max(1, opts.MaxTokens) + safety

	r, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUFWithContext(uint32(limit)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })

	handler := NewHandler(r, HandlerOptions{Defaults: opts})
	post := func(mode string, stream bool) *httptest.ResponseRecorder {
		t.Helper()
		body := map[string]any{
			"messages": []map[string]string{
				{"role": "system", "content": "c"},
				{"role": "user", "content": strings.Repeat("a", 256)},
				{"role": "assistant", "content": strings.Repeat("b", 256)},
				{"role": "user", "content": "dddd"},
			},
			"max_tokens": 1,
		}
		if mode != "" {
			body["gopherllm_context_mode"] = mode
		}
		if stream {
			body["stream"] = true
		}
		data, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(data)))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		return rec
	}

	full := post("", false)
	if full.Code != http.StatusBadRequest {
		t.Fatalf("default full-history request status = %d, want %d", full.Code, http.StatusBadRequest)
	}

	recent := post("recent", false)
	if recent.Code != http.StatusOK {
		t.Fatalf("recent-context request status = %d, want %d", recent.Code, http.StatusOK)
	}
	if got := recent.Header().Get("X-GopherLLM-Context-Mode"); got != "recent" {
		t.Fatalf("context mode header = %q, want recent", got)
	}
	if got := recent.Header().Get("X-GopherLLM-Context-Dropped-Messages"); got != "2" {
		t.Fatalf("dropped messages header = %q, want 2", got)
	}
	if !strings.Contains(recent.Body.String(), `"gopherllm_context"`) {
		t.Fatalf("non-streaming response omitted context metadata: %s", recent.Body.String())
	}
	if !strings.Contains(recent.Body.String(), `"gopherllm_cache"`) {
		t.Fatalf("non-streaming response omitted cache metadata: %s", recent.Body.String())
	}

	stream := post("recent", true)
	if stream.Code != http.StatusOK {
		t.Fatalf("recent streaming request status = %d, want %d", stream.Code, http.StatusOK)
	}
	if got := stream.Header().Get("X-GopherLLM-Context-Mode"); got != "" {
		t.Fatalf("stream sent initial context header %q instead of final SSE metadata", got)
	}
	if !strings.Contains(stream.Body.String(), `"gopherllm_context"`) {
		t.Fatalf("streaming response omitted terminal context metadata: %s", stream.Body.String())
	}
	if !strings.Contains(stream.Body.String(), `"gopherllm_cache"`) {
		t.Fatalf("streaming response omitted terminal cache metadata: %s", stream.Body.String())
	}

	invalid := post("everything", false)
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid context mode status = %d, want %d", invalid.Code, http.StatusBadRequest)
	}
	if !strings.Contains(invalid.Body.String(), "gopherllm_context_mode") {
		t.Fatalf("invalid context mode response = %q", invalid.Body.String())
	}
}
