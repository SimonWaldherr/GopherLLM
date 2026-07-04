package main

import "encoding/json"

// ToolCallFunction is the OpenAI-compatible function payload of a tool call.
// Arguments is a JSON-encoded object (a string, matching the OpenAI wire
// format), not a nested object, so it round-trips through JSON unchanged
// regardless of what the caller's argument schema looks like.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall is one function call requested by the assistant, OpenAI-compatible.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // always "function"
	Function ToolCallFunction `json:"function"`
}

// ToolFunctionDef describes a callable function, OpenAI-compatible.
type ToolFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolDefinition is one entry of an OpenAI-compatible "tools" array.
type ToolDefinition struct {
	Type     string          `json:"type"` // always "function"
	Function ToolFunctionDef `json:"function"`
}

const toolCallIDAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

// newToolCallID generates a tool-call id using the runtime's existing sampler
// RNG (so ids are reproducible under a fixed --seed, like everything else
// about a run). Mistral's template requires exactly 9 alphanumeric
// characters; other conventions accept any short opaque string, so a single
// generator satisfies every template family.
func newToolCallID(rng *Rng) string {
	b := make([]byte, 9)
	for i := range b {
		idx := int(rng.NextF32() * float32(len(toolCallIDAlphabet)))
		if idx < 0 || idx >= len(toolCallIDAlphabet) {
			idx = 0
		}
		b[i] = toolCallIDAlphabet[idx]
	}
	return string(b)
}

// validToolCallID reports whether id already satisfies Mistral's exactly-9
// alphanumeric-character requirement. Applied universally (not just for
// Mistral) since a conforming id is harmless for every other convention too.
func validToolCallID(id string) bool {
	if len(id) != 9 {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}

// findTool returns the definition named name, if present.
func findTool(tools []ToolDefinition, name string) (ToolFunctionDef, bool) {
	for _, t := range tools {
		if t.Function.Name == name {
			return t.Function, true
		}
	}
	return ToolFunctionDef{}, false
}

// toolNames returns the function names of tools, in order.
func toolNames(tools []ToolDefinition) []string {
	out := make([]string, len(tools))
	for i, t := range tools {
		out[i] = t.Function.Name
	}
	return out
}
