package gopherllm

import "testing"

func TestToolHelpers(t *testing.T) {
	tools := []ToolDefinition{
		{Type: "function", Function: ToolFunctionDef{Name: "weather"}},
		{Type: "function", Function: ToolFunctionDef{Name: "calendar"}},
	}

	tool, ok := findTool(tools, "calendar")
	if !ok || tool.Name != "calendar" {
		t.Fatalf("findTool(calendar) = %+v, %v", tool, ok)
	}
	if _, ok := findTool(tools, "missing"); ok {
		t.Fatal("findTool(missing) unexpectedly found a tool")
	}
	names := toolNames(tools)
	if len(names) != 2 || names[0] != "weather" || names[1] != "calendar" {
		t.Fatalf("toolNames = %v", names)
	}
}

func TestToolCallIDsAreValidAndDeterministic(t *testing.T) {
	a, b := NewRng(42), NewRng(42)
	for range 4 {
		left, right := newToolCallID(a), newToolCallID(b)
		if left != right {
			t.Fatalf("same seed generated %q and %q", left, right)
		}
		if !validToolCallID(left) {
			t.Fatalf("generated invalid id %q", left)
		}
	}
	for _, id := range []string{"", "short", "abcdefghij", "abcd-1234", "abc defgh"} {
		if validToolCallID(id) {
			t.Fatalf("validToolCallID(%q) = true", id)
		}
	}
}
