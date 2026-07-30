package tooling

import "encoding/json"

// ToolCallFunction is the OpenAI-compatible function payload of a tool call.
// Arguments is a JSON-encoded object (a string, matching the OpenAI wire
// format), not a nested object, so it round-trips through JSON unchanged
// regardless of what the caller's argument schema looks like.
type CallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall is one function call requested by the assistant, OpenAI-compatible.
type Call struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // always "function"
	Function CallFunction `json:"function"`
}

// ToolFunctionDef describes a callable function, OpenAI-compatible.
type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolDefinition is one entry of an OpenAI-compatible "tools" array.
type Definition struct {
	Type     string             `json:"type"` // always "function"
	Function FunctionDefinition `json:"function"`
}

const callIDAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

// newToolCallID generates a tool-call id using the runtime's existing sampler
// RNG (so ids are reproducible under a fixed --seed, like everything else
// about a run). Mistral's template requires exactly 9 alphanumeric
// characters; other conventions accept any short opaque string, so a single
// generator satisfies every template family.
func NewCallID(next func() float32) string {
	b := make([]byte, 9)
	for i := range b {
		idx := int(next() * float32(len(callIDAlphabet)))
		if idx < 0 || idx >= len(callIDAlphabet) {
			idx = 0
		}
		b[i] = callIDAlphabet[idx]
	}
	return string(b)
}

// validToolCallID reports whether id already satisfies Mistral's exactly-9
// alphanumeric-character requirement. Applied universally (not just for
// Mistral) since a conforming id is harmless for every other convention too.
func ValidCallID(id string) bool {
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
func Find(tools []Definition, name string) (FunctionDefinition, bool) {
	for _, t := range tools {
		if t.Function.Name == name {
			return t.Function, true
		}
	}
	return FunctionDefinition{}, false
}

// toolNames returns the function names of tools, in order.
func Names(tools []Definition) []string {
	out := make([]string, len(tools))
	for i, t := range tools {
		out[i] = t.Function.Name
	}
	return out
}
