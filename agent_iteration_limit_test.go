package gopherllm

import (
	"context"
	"testing"
)

// A model that keeps insisting on calling a tool, forever. Real models don't
// literally never stop, but the point of the iteration cap is to survive one
// that effectively doesn't for a given turn.
func alwaysWantsToolGenerator(toolName string) chatGenerator {
	return func(messages []ChatMessage, options GenerationOptions, onToken func(string) bool) (GenerationResult, error) {
		if len(options.Tools) == 0 {
			// Tools were withdrawn for this call (the forced final pass):
			// behave like a real model that, with nothing left to call,
			// answers in words instead.
			return GenerationResult{Text: "I could not fully resolve this in time.", FinishReason: "stop"}, nil
		}
		return GenerationResult{
			ToolCalls:    []ToolCall{{ID: "1", Type: "function", Function: ToolCallFunction{Name: toolName, Arguments: "{}"}}},
			FinishReason: "tool_calls",
		}, nil
	}
}

// This is the second half of the "agents work great but the answer is wrong"
// bug report: even after grounding stops fabrication, a model that keeps
// requesting tools right up to the iteration cap used to have its last,
// unresolved tool_calls request returned to the caller as if it were the
// final answer — empty Text, finish_reason "tool_calls", nothing a chat UI
// can render. The loop must instead force one more pass with tools withdrawn
// so the model is made to answer in words.
func TestRunAgenticChatForcesATextAnswerWhenIterationsRunOutMidToolCall(t *testing.T) {
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "lookup"}},
		Execute:    func(ctx context.Context, c ToolCall) (string, error) { return "some partial fact", nil },
	}

	var events []AgentEvent
	observe := func(e AgentEvent) { events = append(events, e) }

	result, err := runAgenticChatWith(alwaysWantsToolGenerator("lookup"),
		[]ChatMessage{UserMessage("hi")}, DefaultGenerationOptions(), nil, []AgenticTool{tool}, nil, observe)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text == "" {
		t.Fatal("expected a non-empty text answer once the iteration budget was exhausted")
	}
	if len(result.ToolCalls) != 0 {
		t.Fatalf("final result must not carry an unresolved tool call: %+v", result.ToolCalls)
	}

	toolCalls := 0
	for _, e := range events {
		if e.Kind == AgentEventToolStart {
			toolCalls++
		}
	}
	if toolCalls != maxAgenticIterations {
		t.Fatalf("expected exactly %d tool calls (one per allowed iteration), got %d", maxAgenticIterations, toolCalls)
	}

	last := events[len(events)-1]
	if last.Kind != AgentEventIteration || last.Iteration != maxAgenticIterations+1 {
		t.Fatalf("expected a final iteration event marking the forced answer pass, got %+v", last)
	}
}

// When the model naturally stops asking for tools before the cap, nothing
// about the new exhaustion handling should kick in — this is a plain
// regression guard against the forced final pass firing unconditionally.
func TestRunAgenticChatDoesNotForceAnExtraPassWhenTheLoopEndsNaturally(t *testing.T) {
	calls := 0
	generate := func(messages []ChatMessage, options GenerationOptions, onToken func(string) bool) (GenerationResult, error) {
		calls++
		if calls == 1 {
			return GenerationResult{
				ToolCalls:    []ToolCall{{ID: "1", Type: "function", Function: ToolCallFunction{Name: "lookup", Arguments: "{}"}}},
				FinishReason: "tool_calls",
			}, nil
		}
		return GenerationResult{Text: "final answer", FinishReason: "stop"}, nil
	}
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "lookup"}},
		Execute:    func(ctx context.Context, c ToolCall) (string, error) { return "fact", nil },
	}

	result, err := runAgenticChatWith(generate, []ChatMessage{UserMessage("hi")}, DefaultGenerationOptions(), nil, []AgenticTool{tool}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "final answer" {
		t.Fatalf("result.Text = %q, want the model's natural final answer", result.Text)
	}
	if calls != 2 {
		t.Fatalf("generate called %d times, want exactly 2 (one tool round, one final answer)", calls)
	}
}
