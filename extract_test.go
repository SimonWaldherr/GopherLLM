package gopherllm

import "testing"

func TestExtractThinkBasic(t *testing.T) {
	content, reasoning := extractThink("<think>let me work this out</think>The answer is 4.")
	if content != "The answer is 4." {
		t.Fatalf("content = %q", content)
	}
	if reasoning != "let me work this out" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

func TestExtractThinkNoTags(t *testing.T) {
	content, reasoning := extractThink("just a plain answer")
	if content != "just a plain answer" || reasoning != "" {
		t.Fatalf("content=%q reasoning=%q", content, reasoning)
	}
}

func TestExtractThinkMultipleBlocks(t *testing.T) {
	content, reasoning := extractThink("<think>first</think>middle<think>second</think>end")
	if content != "middleend" {
		t.Fatalf("content = %q", content)
	}
	if reasoning != "first\n\nsecond" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

func TestExtractThinkUnterminated(t *testing.T) {
	// Generation cut off mid-thought (e.g. hit max_tokens): everything from
	// <think> onward is reasoning, not leaked into content.
	content, reasoning := extractThink("<think>still thinking and it never ends")
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if reasoning != "still thinking and it never ends" {
		t.Fatalf("reasoning = %q", reasoning)
	}
}

func TestExtractToolCallsMistralSingle(t *testing.T) {
	text := `[TOOL_CALLS]get_weather[ARGS]{"city": "Berlin"}`
	content, calls := extractToolCallsMistral(text)
	if content != "" {
		t.Fatalf("content = %q, want empty", content)
	}
	if len(calls) != 1 {
		t.Fatalf("calls = %d, want 1", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("name = %q", calls[0].Function.Name)
	}
	if calls[0].Function.Arguments != `{"city": "Berlin"}` {
		t.Fatalf("arguments = %q", calls[0].Function.Arguments)
	}
}

func TestExtractToolCallsMistralMultipleAndPrefix(t *testing.T) {
	text := `Sure, let me check that.[TOOL_CALLS]a[ARGS]{}[TOOL_CALLS]b[ARGS]{"x": 1}`
	content, calls := extractToolCallsMistral(text)
	if content != "Sure, let me check that." {
		t.Fatalf("content = %q", content)
	}
	if len(calls) != 2 || calls[0].Function.Name != "a" || calls[1].Function.Name != "b" {
		t.Fatalf("calls = %+v", calls)
	}
	if calls[0].Function.Arguments != "{}" || calls[1].Function.Arguments != `{"x": 1}` {
		t.Fatalf("arguments = %q / %q", calls[0].Function.Arguments, calls[1].Function.Arguments)
	}
}

func TestExtractToolCallsMistralNoMarker(t *testing.T) {
	content, calls := extractToolCallsMistral("no tool calls here")
	if content != "no tool calls here" || calls != nil {
		t.Fatalf("content=%q calls=%v", content, calls)
	}
}

func TestExtractToolCallsMistralMissingArgsMarkerIsMalformed(t *testing.T) {
	content, calls := extractToolCallsMistral("before[TOOL_CALLS]nameButNoArgsMarker")
	if calls != nil {
		t.Fatalf("calls = %v, want nil when [ARGS] is missing", calls)
	}
	if content != "before" {
		t.Fatalf("content = %q", content)
	}
}

func TestExtractToolCallsGenericSingle(t *testing.T) {
	text := `<tool_call>{"name": "search", "arguments": {"q": "go generics"}}</tool_call>`
	content, calls := extractToolCallsGeneric(text)
	if content != "" {
		t.Fatalf("content = %q", content)
	}
	if len(calls) != 1 || calls[0].Function.Name != "search" {
		t.Fatalf("calls = %+v", calls)
	}
	if calls[0].Function.Arguments != `{"q": "go generics"}` {
		t.Fatalf("arguments = %q", calls[0].Function.Arguments)
	}
}

func TestExtractToolCallsGenericInterleavedWithProse(t *testing.T) {
	text := "Let me look that up.\n<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>\nDone."
	content, calls := extractToolCallsGeneric(text)
	if content != "Let me look that up.\n\nDone." {
		t.Fatalf("content = %q", content)
	}
	if len(calls) != 1 || calls[0].Function.Name != "search" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestExtractToolCallsGenericMultipleBlocks(t *testing.T) {
	text := `<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>`
	_, calls := extractToolCallsGeneric(text)
	if len(calls) != 2 || calls[0].Function.Name != "a" || calls[1].Function.Name != "b" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestExtractToolCallsGenericMalformedBlockPreserved(t *testing.T) {
	text := "<tool_call>not json</tool_call>"
	content, calls := extractToolCallsGeneric(text)
	if calls != nil {
		t.Fatalf("calls = %v, want nil", calls)
	}
	if content != text {
		t.Fatalf("content = %q, want malformed block preserved verbatim", content)
	}
}

func TestExtractToolCallsGenericNoBlocks(t *testing.T) {
	content, calls := extractToolCallsGeneric("just a normal reply")
	if content != "just a normal reply" || calls != nil {
		t.Fatalf("content=%q calls=%v", content, calls)
	}
}

func TestExtractGptOssChannelsFinalOnly(t *testing.T) {
	text := "<|channel|>final<|message|>The answer is 4.<|end|>"
	content, reasoning, calls := extractGptOssChannels(text)
	if content != "The answer is 4." || reasoning != "" || calls != nil {
		t.Fatalf("content=%q reasoning=%q calls=%v", content, reasoning, calls)
	}
}

func TestExtractGptOssChannelsAnalysisAndFinal(t *testing.T) {
	text := "<|channel|>analysis<|message|>think think think<|end|><|channel|>final<|message|>done<|end|>"
	content, reasoning, calls := extractGptOssChannels(text)
	if content != "done" {
		t.Fatalf("content = %q", content)
	}
	if reasoning != "think think think" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	if calls != nil {
		t.Fatalf("calls = %v", calls)
	}
}

func TestExtractGptOssChannelsCommentaryToolCall(t *testing.T) {
	text := `<|channel|>commentary to=functions.get_weather<|message|>{"city": "Berlin"}<|call|>`
	content, reasoning, calls := extractGptOssChannels(text)
	if content != "" || reasoning != "" {
		t.Fatalf("content=%q reasoning=%q", content, reasoning)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("calls = %+v", calls)
	}
	if calls[0].Function.Arguments != `{"city": "Berlin"}` {
		t.Fatalf("arguments = %q", calls[0].Function.Arguments)
	}
}

func TestExtractGptOssChannelsNoChannelMarkersIsNoOp(t *testing.T) {
	// This is the current default: renderGptOssMessages forces channel=final
	// into the prompt, so generated text never contains channel markers.
	content, reasoning, calls := extractGptOssChannels("plain answer, no harmony structure")
	if content != "plain answer, no harmony structure" {
		t.Fatalf("content = %q", content)
	}
	if reasoning != "" || calls != nil {
		t.Fatalf("reasoning=%q calls=%v", reasoning, calls)
	}
}

func TestExtractGptOssChannelsUnterminatedTrailingSegment(t *testing.T) {
	text := "<|channel|>final<|message|>cut off mid"
	content, _, _ := extractGptOssChannels(text)
	if content != "cut off mid" {
		t.Fatalf("content = %q", content)
	}
}
