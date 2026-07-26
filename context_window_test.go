package gopherllm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func newContextWindowTestRunner(t *testing.T) *Runner {
	t.Helper()
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

func recentContextWindowOptions() GenerationOptions {
	opts := DefaultGenerationOptions()
	opts.ContextWindowMode = ContextWindowRecent
	opts.MaxTokens = 1
	opts.SystemPrompt = ""
	return opts
}

// contextWindowLimitFor makes the supplied messages fit exactly in the
// recent-mode input budget. It deliberately uses the loaded tiny GGUF's real
// tokenizer and renderer, rather than assuming a character-to-token ratio.
func contextWindowLimitFor(r *Runner, messages []ChatMessage, opts GenerationOptions) int {
	promptTokens := len(r.renderMessages(messages, opts.SystemPrompt, opts.activeTools()))
	return promptTokens + max(1, opts.MaxTokens) + contextWindowSafetyTokens
}

func TestPrepareChatContextDefaultFullPreservesOversizedHistory(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := DefaultGenerationOptions() // zero ContextWindowMode must stay full.
	opts.MaxTokens = 1
	opts.SystemPrompt = ""
	messages := []ChatMessage{UserMessage(strings.Repeat("a", 96))}
	r.config.MaxSeqLen = 16

	got, info, err := r.PrepareChatContext(messages, opts)
	if err != nil {
		t.Fatalf("PrepareChatContext(full) error = %v", err)
	}
	if !reflect.DeepEqual(got, messages) {
		t.Fatalf("full mode changed messages:\n got: %#v\nwant: %#v", got, messages)
	}
	if info.Mode != ContextWindowFull || info.DroppedMessages != 0 || info.RetainedMessages != len(messages) {
		t.Fatalf("full-mode info = %+v", info)
	}
	if info.PromptTokens < r.config.MaxSeqLen {
		t.Fatalf("test setup did not exceed context: prompt=%d context=%d", info.PromptTokens, r.config.MaxSeqLen)
	}
	if _, err := r.GenerateChat(got, opts); err == nil {
		t.Fatal("full mode should retain the historical over-context generation error")
	}
}

func TestPrepareChatContextRecentKeepsLeadingSystemAndNewestUserTurn(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := recentContextWindowOptions()
	system := ChatMessage{Role: ChatRoleSystem, Content: "c"}
	latest := UserMessage("dddd")
	messages := []ChatMessage{
		system,
		UserMessage(strings.Repeat("a", 256)),
		AssistantMessage(strings.Repeat("b", 256)),
		latest,
	}
	want := []ChatMessage{system, latest}
	r.config.MaxSeqLen = contextWindowLimitFor(r, want, opts)

	got, info, err := r.PrepareChatContext(messages, opts)
	if err != nil {
		t.Fatalf("PrepareChatContext(recent) error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("recent context:\n got: %#v\nwant: %#v", got, want)
	}
	if info.PromptTokens > info.PromptBudget {
		t.Fatalf("retained prompt has %d tokens, budget is %d", info.PromptTokens, info.PromptBudget)
	}
	if info.DroppedMessages != len(messages)-len(want) {
		t.Fatalf("dropped messages = %d, want %d", info.DroppedMessages, len(messages)-len(want))
	}
}

func TestGenerateChatRecentUsesPreparedContext(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := recentContextWindowOptions()
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	system := ChatMessage{Role: ChatRoleSystem, Content: "c"}
	latest := UserMessage("dddd")
	messages := []ChatMessage{
		system,
		UserMessage(strings.Repeat("a", 256)),
		AssistantMessage(strings.Repeat("b", 256)),
		latest,
	}
	want := []ChatMessage{system, latest}
	r.config.MaxSeqLen = contextWindowLimitFor(r, want, opts)

	result, err := r.GenerateChat(messages, opts)
	if err != nil {
		t.Fatalf("GenerateChat(recent) error = %v", err)
	}
	if wantPromptTokens := len(r.renderMessages(want, opts.SystemPrompt, opts.activeTools())); result.Stats.PromptTokens != wantPromptTokens {
		t.Fatalf("generation used %d prompt tokens, want retained context's %d", result.Stats.PromptTokens, wantPromptTokens)
	}
	if result.ContextWindow == nil || result.ContextWindow.Mode != ContextWindowRecent || result.ContextWindow.DroppedMessages != len(messages)-len(want) {
		t.Fatalf("generation context metadata = %+v", result.ContextWindow)
	}
}

func TestPrepareChatContextRecentKeepsNewestToolTurnWhole(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := recentContextWindowOptions()
	system := ChatMessage{Role: ChatRoleSystem, Content: "c"}
	calls := []ToolCall{
		{ID: "call-a", Type: "function", Function: ToolCallFunction{Name: "a", Arguments: `{"a":1}`}},
		{ID: "call-b", Type: "function", Function: ToolCallFunction{Name: "b", Arguments: `{"b":2}`}},
	}
	newestTurn := []ChatMessage{
		UserMessage("dddd"),
		{Role: ChatRoleAssistant, ToolCalls: calls},
		ToolResultMessage("call-a", "a", "eeee"),
		ToolResultMessage("call-b", "b", "ffff"),
	}
	messages := append([]ChatMessage{
		system,
		UserMessage(strings.Repeat("a", 256)),
		AssistantMessage(strings.Repeat("b", 256)),
	}, newestTurn...)
	want := append([]ChatMessage{system}, newestTurn...)
	r.config.MaxSeqLen = contextWindowLimitFor(r, want, opts)

	got, info, err := r.PrepareChatContext(messages, opts)
	if err != nil {
		t.Fatalf("PrepareChatContext(recent) error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tool turn was not retained as a complete group:\n got: %#v\nwant: %#v", got, want)
	}
	if info.DroppedMessages != len(messages)-len(want) {
		t.Fatalf("dropped messages = %d, want %d", info.DroppedMessages, len(messages)-len(want))
	}
}

func TestPrepareChatContextRecentDropsOversizedToolTurnAtomically(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := recentContextWindowOptions()
	system := ChatMessage{Role: ChatRoleSystem, Content: "c"}
	latest := UserMessage("dddd")
	calls := []ToolCall{
		{ID: "call-a", Type: "function", Function: ToolCallFunction{Name: "a", Arguments: `{"a":1}`}},
		{ID: "call-b", Type: "function", Function: ToolCallFunction{Name: "b", Arguments: `{"b":2}`}},
	}
	toolA := ToolResultMessage("call-a", "a", "eeeeeeee")
	toolB := ToolResultMessage("call-b", "b", "ffffffff")
	messages := []ChatMessage{
		system,
		UserMessage(strings.Repeat("a", 256)),
		{Role: ChatRoleAssistant, ToolCalls: calls},
		toolA,
		toolB,
		latest,
	}
	want := []ChatMessage{system, latest}

	// Leave enough room that an incorrectly message-by-message implementation
	// could keep the last tool result, but not enough for the complete prior
	// user/tool turn. Correct recent-mode grouping must drop that turn whole.
	partial := []ChatMessage{system, toolB, latest}
	partialLimit := contextWindowLimitFor(r, partial, opts)
	wholePreviousTurn := append([]ChatMessage{system}, messages[1:]...)
	if len(r.renderMessages(wholePreviousTurn, opts.SystemPrompt, opts.activeTools()))+max(1, opts.MaxTokens)+contextWindowSafetyTokens <= partialLimit {
		t.Fatal("test setup: complete tool turn unexpectedly fits")
	}
	r.config.MaxSeqLen = partialLimit

	got, info, err := r.PrepareChatContext(messages, opts)
	if err != nil {
		t.Fatalf("PrepareChatContext(recent) error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("oversized tool turn was split or retained:\n got: %#v\nwant: %#v", got, want)
	}
	if info.DroppedMessages != len(messages)-len(want) {
		t.Fatalf("dropped messages = %d, want %d", info.DroppedMessages, len(messages)-len(want))
	}
}

func TestPrepareChatContextRecentRejectsTooLargeLatestTurn(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := recentContextWindowOptions()
	messages := []ChatMessage{UserMessage(strings.Repeat("a", 256))}
	r.config.MaxSeqLen = 24

	got, info, err := r.PrepareChatContext(messages, opts)
	if err == nil {
		t.Fatalf("PrepareChatContext(recent) = %#v, %+v, nil; want latest-turn error", got, info)
	}
	if !strings.Contains(err.Error(), "latest user turn") {
		t.Fatalf("error = %q, want latest user turn context", err)
	}
	if got != nil {
		t.Fatalf("messages = %#v, want nil on an unfit latest turn", got)
	}
}

func TestOpenAIRecentContextIsOptIn(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 1
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	system := ChatMessage{Role: ChatRoleSystem, Content: "c"}
	latest := UserMessage("dddd")
	r.config.MaxSeqLen = contextWindowLimitFor(r, []ChatMessage{system, latest}, opts)

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
