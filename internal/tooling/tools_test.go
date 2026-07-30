package tooling

import "testing"

func TestToolHelpers(t *testing.T) {
	tools := []Definition{
		{Type: "function", Function: FunctionDefinition{Name: "weather"}},
		{Type: "function", Function: FunctionDefinition{Name: "calendar"}},
	}

	tool, ok := Find(tools, "calendar")
	if !ok || tool.Name != "calendar" {
		t.Fatalf("findTool(calendar) = %+v, %v", tool, ok)
	}
	if _, ok := Find(tools, "missing"); ok {
		t.Fatal("findTool(missing) unexpectedly found a tool")
	}
	names := Names(tools)
	if len(names) != 2 || names[0] != "weather" || names[1] != "calendar" {
		t.Fatalf("toolNames = %v", names)
	}
}

func TestToolCallIDsAreValidAndDeterministic(t *testing.T) {
	a, b := callIDSequence(42), callIDSequence(42)
	for range 4 {
		left, right := NewCallID(a), NewCallID(b)
		if left != right {
			t.Fatalf("same seed generated %q and %q", left, right)
		}
		if !ValidCallID(left) {
			t.Fatalf("generated invalid id %q", left)
		}
	}
	for _, id := range []string{"", "short", "abcdefghij", "abcd-1234", "abc defgh"} {
		if ValidCallID(id) {
			t.Fatalf("validToolCallID(%q) = true", id)
		}
	}
}

func callIDSequence(seed uint64) func() float32 {
	if seed == 0 {
		seed = 1
	}
	state := seed
	return func() float32 {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		return float32(state>>40) / float32(uint64(1)<<24)
	}
}
