package gopherllm

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func observedToolCalls(t *testing.T, tool AgenticTool, calls []ToolCall) []AgentEvent {
	t.Helper()
	var events []AgentEvent
	_, ok := resolveInternalToolCalls(context.Background(), calls, nil, []AgenticTool{tool}, 2,
		func(e AgentEvent) { events = append(events, e) })
	if !ok {
		t.Fatal("calls were not resolved internally")
	}
	return events
}

func testTool(name string, run func() (string, error)) AgenticTool {
	return AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: name}},
		Execute:    func(context.Context, ToolCall) (string, error) { return run() },
	}
}

// The point of the observer is that a caller can say what ran and how long it
// took, so both ends of every tool call must be reported with its identity.
func TestObserverReportsEachToolCallWithTiming(t *testing.T) {
	tool := testTool("wikipedia_search", func() (string, error) { return "an article extract", nil })
	events := observedToolCalls(t, tool, []ToolCall{{
		ID:       "call-1",
		Function: ToolCallFunction{Name: "wikipedia_search", Arguments: `{"query":"gopher"}`},
	}})

	if len(events) != 2 {
		t.Fatalf("got %d events, want a start and an end: %+v", len(events), events)
	}
	start, end := events[0], events[1]
	if start.Kind != AgentEventToolStart || end.Kind != AgentEventToolEnd {
		t.Fatalf("kinds = %q, %q", start.Kind, end.Kind)
	}
	if start.Tool != "wikipedia_search" || end.Tool != "wikipedia_search" {
		t.Fatalf("tool names = %q, %q", start.Tool, end.Tool)
	}
	if !strings.Contains(start.Arguments, "gopher") {
		t.Fatalf("start event lost the arguments: %q", start.Arguments)
	}
	if end.Result != "an article extract" {
		t.Fatalf("end result = %q", end.Result)
	}
	// The iteration number is what lets a UI group several calls into rounds.
	if start.Iteration != 2 || end.Iteration != 2 {
		t.Fatalf("iterations = %d, %d; want 2", start.Iteration, end.Iteration)
	}
	if end.DurationMS < 0 {
		t.Fatalf("negative duration: %d", end.DurationMS)
	}
}

// A failing tool must still be reported, and must not abort the loop: the
// error is fed back so the model can correct itself.
func TestObserverReportsToolFailuresWithoutStoppingTheLoop(t *testing.T) {
	tool := testTool("wikidata_sparql", func() (string, error) { return "", errors.New("upstream 503") })
	events := observedToolCalls(t, tool, []ToolCall{{
		ID: "call-1", Function: ToolCallFunction{Name: "wikidata_sparql", Arguments: "{}"},
	}})

	end := events[len(events)-1]
	if end.Kind != AgentEventToolEnd {
		t.Fatalf("last event kind = %q", end.Kind)
	}
	if !strings.Contains(end.Error, "upstream 503") {
		t.Fatalf("failure not reported: %+v", end)
	}
	if end.Result != "" {
		t.Fatalf("a failed call must not also report a result: %q", end.Result)
	}
}

// Events are for display, so a huge tool result must not be streamed verbatim
// — while the model still receives the untruncated version.
func TestObserverTruncatesLongResults(t *testing.T) {
	huge := strings.Repeat("x", agentEventTextLimit*3)
	tool := testTool("wikipedia_summary", func() (string, error) { return huge, nil })
	events := observedToolCalls(t, tool, []ToolCall{{
		ID: "call-1", Function: ToolCallFunction{Name: "wikipedia_summary", Arguments: "{}"},
	}})

	end := events[len(events)-1]
	if len(end.Result) > agentEventTextLimit+4 {
		t.Fatalf("result not truncated: %d chars", len(end.Result))
	}
	if !strings.HasSuffix(end.Result, "…") {
		t.Fatalf("truncation not marked: %q", end.Result[len(end.Result)-8:])
	}
}

// A nil observer is the ordinary path and must stay free of surprises.
func TestNilObserverIsSafe(t *testing.T) {
	tool := testTool("noop", func() (string, error) { return "ok", nil })
	if _, ok := resolveInternalToolCalls(context.Background(),
		[]ToolCall{{ID: "c", Function: ToolCallFunction{Name: "noop"}}}, nil, []AgenticTool{tool}, 1, nil); !ok {
		t.Fatal("nil observer changed resolution behaviour")
	}
}

// Several calls in one turn each get their own pair of events, in order.
func TestObserverReportsEveryCallInATurn(t *testing.T) {
	tool := testTool("lookup", func() (string, error) { return "r", nil })
	events := observedToolCalls(t, tool, []ToolCall{
		{ID: "a", Function: ToolCallFunction{Name: "lookup", Arguments: `{"n":1}`}},
		{ID: "b", Function: ToolCallFunction{Name: "lookup", Arguments: `{"n":2}`}},
	})
	if len(events) != 4 {
		t.Fatalf("got %d events for two calls, want 4: %+v", len(events), events)
	}
	if events[0].Kind != AgentEventToolStart || events[1].Kind != AgentEventToolEnd ||
		events[2].Kind != AgentEventToolStart || events[3].Kind != AgentEventToolEnd {
		t.Fatalf("events out of order: %+v", events)
	}
	if !strings.Contains(events[0].Arguments, `"n":1`) || !strings.Contains(events[2].Arguments, `"n":2`) {
		t.Fatalf("arguments not matched to their calls: %+v", events)
	}
}
