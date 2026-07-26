package gopherllm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestApiMessagesMapsToolRole(t *testing.T) {
	msgs := apiMessages([]APIMessage{
		{Role: "tool", Content: "18C, sunny", ToolCallID: "abc123XYZ", Name: "get_weather"},
	})
	if len(msgs) != 1 {
		t.Fatalf("len = %d", len(msgs))
	}
	if msgs[0].Role != ChatRoleTool || msgs[0].ToolCallID != "abc123XYZ" || msgs[0].Name != "get_weather" {
		t.Fatalf("msg = %+v", msgs[0])
	}
}

func TestApiMessagesCarriesAssistantToolCalls(t *testing.T) {
	calls := []ToolCall{{ID: "id1", Type: "function", Function: ToolCallFunction{Name: "get_weather", Arguments: "{}"}}}
	msgs := apiMessages([]APIMessage{{Role: "assistant", ToolCalls: calls}})
	if len(msgs) != 1 || len(msgs[0].ToolCalls) != 1 || msgs[0].ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("msgs = %+v", msgs)
	}
}

func TestNormalizeToolChoice(t *testing.T) {
	if got := normalizeToolChoice("none"); got != "none" {
		t.Fatalf("normalizeToolChoice(none) = %q", got)
	}
	if got := normalizeToolChoice("auto"); got != "auto" {
		t.Fatalf("normalizeToolChoice(auto) = %q", got)
	}
	if got := normalizeToolChoice(nil); got != "" {
		t.Fatalf("normalizeToolChoice(nil) = %q, want empty", got)
	}
	if got := normalizeToolChoice(map[string]any{"type": "function"}); got != "" {
		t.Fatalf("normalizeToolChoice(object) = %q, want empty (degrades to auto)", got)
	}
}

func TestOpenAIChatResponseIncludesToolCallsAndFinishReason(t *testing.T) {
	result := GenerationResult{
		Text:         "",
		FinishReason: "tool_calls",
		ToolCalls:    []ToolCall{{ID: "id1", Type: "function", Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Berlin"}`}}},
	}
	resp := openAIChatResponse("test-model", result)
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, `"finish_reason":"tool_calls"`) {
		t.Fatalf("missing finish_reason: %s", s)
	}
	if !strings.Contains(s, `"tool_calls"`) || !strings.Contains(s, "get_weather") {
		t.Fatalf("missing tool_calls: %s", s)
	}
	if !strings.Contains(s, `"content":null`) {
		t.Fatalf("expected null content when text is empty: %s", s)
	}
}

func TestOpenAIChatResponseIncludesReasoningContent(t *testing.T) {
	result := GenerationResult{Text: "42", ReasoningText: "let me think", FinishReason: "stop"}
	resp := openAIChatResponse("test-model", result)
	b, _ := json.Marshal(resp)
	if !strings.Contains(string(b), "let me think") {
		t.Fatalf("missing reasoning_content: %s", b)
	}
}

func TestOpenAIChatResponseDefaultsFinishReason(t *testing.T) {
	resp := openAIChatResponse("m", GenerationResult{Text: "hi"})
	choices := resp["choices"].([]any)
	choice := choices[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %v, want stop", choice["finish_reason"])
	}
}

func TestGenerateResponseIncludesToolCallsAndReasoning(t *testing.T) {
	result := GenerationResult{
		Text:          "",
		ReasoningText: "thinking",
		ToolCalls:     []ToolCall{{ID: "id1", Function: ToolCallFunction{Name: "f", Arguments: "{}"}}},
		FinishReason:  "tool_calls",
	}
	resp := generateResponse(result)
	if resp["reasoning"] != "thinking" {
		t.Fatalf("reasoning = %v", resp["reasoning"])
	}
	if resp["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v", resp["finish_reason"])
	}
	calls, ok := resp["tool_calls"].([]ToolCall)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %v", resp["tool_calls"])
	}
}
