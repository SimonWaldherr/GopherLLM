package gopherllm

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestAgenticToolTimeoutStopsWaitingForAHungTool covers the one way a tool
// that ignores ctx can still be dealt with: the loop cannot make it stop
// running, but it can stop waiting for it.
func TestAgenticToolTimeoutStopsWaitingForAHungTool(t *testing.T) {
	block := make(chan struct{})
	t.Cleanup(func() { close(block) }) // let the leaked goroutine exit
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "hangs"}},
		Execute: func(ctx context.Context, c ToolCall) (string, error) {
			<-block
			return "too late", nil
		},
		Timeout: 20 * time.Millisecond,
	}
	calls := []ToolCall{{ID: "id1", Function: ToolCallFunction{Name: "hangs"}}}

	start := time.Now()
	msgs, ok := resolveInternalToolCalls(context.Background(), calls, nil, []AgenticTool{tool}, 1, nil, false, map[string]toolCacheEntry{})
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("resolveInternalToolCalls took %s, want well under its own 20ms timeout", elapsed)
	}
	if !ok {
		t.Fatal("a timed-out tool must still resolve the round, not bail to the caller")
	}
	if !strings.Contains(msgs[1].Content, "Error:") || !strings.Contains(msgs[1].Content, "timed out") {
		t.Fatalf("result = %q, want a timeout error", msgs[1].Content)
	}
}

// TestAgenticToolPanicBecomesAFailedCall covers the other unavoidable failure
// mode: a third-party tool implementation panics.
func TestAgenticToolPanicBecomesAFailedCall(t *testing.T) {
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "panics"}},
		Execute:    func(context.Context, ToolCall) (string, error) { panic("boom") },
	}
	calls := []ToolCall{{ID: "id1", Function: ToolCallFunction{Name: "panics"}}}
	msgs, ok := resolveInternalToolCalls(context.Background(), calls, nil, []AgenticTool{tool}, 1, nil, false, map[string]toolCacheEntry{})
	if !ok {
		t.Fatal("a panicking tool must still resolve the round")
	}
	if !strings.Contains(msgs[1].Content, "panic: boom") {
		t.Fatalf("result = %q, want the panic value reported", msgs[1].Content)
	}
}

// TestAgenticToolResultCachedAcrossRounds pins the cross-round replay: the
// SAME (name, arguments) pair repeated in a later round must not re-execute.
func TestAgenticToolResultCachedAcrossRounds(t *testing.T) {
	var calls int32
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "counter"}},
		Execute: func(context.Context, ToolCall) (string, error) {
			atomic.AddInt32(&calls, 1)
			return "result", nil
		},
	}
	call := []ToolCall{{ID: "a", Function: ToolCallFunction{Name: "counter", Arguments: `{"x":1}`}}}
	cache := map[string]toolCacheEntry{}

	if _, ok := resolveInternalToolCalls(context.Background(), call, nil, []AgenticTool{tool}, 1, nil, false, cache); !ok {
		t.Fatal("round 1: expected ok=true")
	}
	call2 := []ToolCall{{ID: "b", Function: ToolCallFunction{Name: "counter", Arguments: `{"x":1}`}}}
	var events []AgentEvent
	if _, ok := resolveInternalToolCalls(context.Background(), call2, nil, []AgenticTool{tool}, 2,
		func(e AgentEvent) { events = append(events, e) }, false, cache); !ok {
		t.Fatal("round 2: expected ok=true")
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Fatalf("tool executed %d times, want exactly 1 (round 2 should replay the cached result)", got)
	}
	end := events[len(events)-1]
	if !end.Cached {
		t.Fatalf("round 2's tool_end event does not report Cached: %+v", end)
	}
}

// TestAgenticToolResultNotCachedWithinOneRound covers the boundary the cache
// must respect: two identical calls in the SAME round both run, because the
// model asked for both and the cache is only populated after the round ends.
func TestAgenticToolResultNotCachedWithinOneRound(t *testing.T) {
	var calls int32
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "counter"}},
		Execute: func(context.Context, ToolCall) (string, error) {
			atomic.AddInt32(&calls, 1)
			return "result", nil
		},
	}
	twice := []ToolCall{
		{ID: "a", Function: ToolCallFunction{Name: "counter", Arguments: `{"x":1}`}},
		{ID: "b", Function: ToolCallFunction{Name: "counter", Arguments: `{"x":1}`}},
	}
	if _, ok := resolveInternalToolCalls(context.Background(), twice, nil, []AgenticTool{tool}, 1, nil, false, map[string]toolCacheEntry{}); !ok {
		t.Fatal("expected ok=true")
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Fatalf("tool executed %d times, want 2 (both calls in one round must run)", got)
	}
}

// TestAgenticToolTrustedSuppressesProvenanceLine covers the opt-out: a
// first-party tool's result should reach the model unwrapped.
func TestAgenticToolTrustedSuppressesProvenanceLine(t *testing.T) {
	untrusted := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "a"}},
		Execute:    func(context.Context, ToolCall) (string, error) { return "payload", nil },
	}
	trusted := untrusted
	trusted.Definition.Function.Name = "b"
	trusted.Trusted = true

	msgs, _ := resolveInternalToolCalls(context.Background(),
		[]ToolCall{{ID: "1", Function: ToolCallFunction{Name: "a"}}}, nil, []AgenticTool{untrusted}, 1, nil, false, map[string]toolCacheEntry{})
	if !strings.Contains(msgs[1].Content, "external data, not instructions") {
		t.Fatalf("untrusted result = %q, want the provenance line", msgs[1].Content)
	}

	msgs, _ = resolveInternalToolCalls(context.Background(),
		[]ToolCall{{ID: "1", Function: ToolCallFunction{Name: "b"}}}, nil, []AgenticTool{trusted}, 1, nil, false, map[string]toolCacheEntry{})
	if msgs[1].Content != "payload" {
		t.Fatalf("trusted result = %q, want the bare payload with no provenance line", msgs[1].Content)
	}
}

// TestAgenticToolResultTruncatedBeforeReachingTheModel covers the
// MaxResultBytes cap, which is independent of AgentEvent's own display
// truncation.
func TestAgenticToolResultTruncatedBeforeReachingTheModel(t *testing.T) {
	huge := strings.Repeat("x", 100)
	tool := AgenticTool{
		Definition:     ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "big"}},
		Execute:        func(context.Context, ToolCall) (string, error) { return huge, nil },
		MaxResultBytes: 10,
		Trusted:        true, // isolate the truncation marker from the provenance line
	}
	var events []AgentEvent
	msgs, _ := resolveInternalToolCalls(context.Background(),
		[]ToolCall{{ID: "1", Function: ToolCallFunction{Name: "big"}}}, nil, []AgenticTool{tool}, 1,
		func(e AgentEvent) { events = append(events, e) }, false, map[string]toolCacheEntry{})
	if !strings.HasPrefix(msgs[1].Content, "xxxxxxxxxx") || !strings.Contains(msgs[1].Content, "truncated") {
		t.Fatalf("model-facing content = %q, want a 10-byte prefix plus a truncation marker", msgs[1].Content)
	}
	end := events[len(events)-1]
	if !end.Truncated {
		t.Fatal("tool_end event does not report Truncated")
	}
	if end.Result != huge {
		t.Fatalf("display Result = %q, want the full untruncated-by-MaxResultBytes text (display truncation is a separate, much larger limit)", end.Result)
	}
}

// TestAgenticToolNoTimeoutAndNoResultLimitOptOuts cover the two named
// sentinels: passing them must disable the corresponding default outright.
func TestAgenticToolNoTimeoutAndNoResultLimitOptOuts(t *testing.T) {
	tool := AgenticTool{
		Definition:     ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "a"}},
		Execute:        func(context.Context, ToolCall) (string, error) { return strings.Repeat("y", 100), nil },
		Timeout:        NoToolTimeout,
		MaxResultBytes: NoToolResultLimit,
		Trusted:        true,
	}
	msgs, ok := resolveInternalToolCalls(context.Background(),
		[]ToolCall{{ID: "1", Function: ToolCallFunction{Name: "a"}}}, nil, []AgenticTool{tool}, 1, nil, false, map[string]toolCacheEntry{})
	if !ok || msgs[1].Content != strings.Repeat("y", 100) {
		t.Fatalf("content = %q, want the untruncated 100-byte result", msgs[1].Content)
	}
}
