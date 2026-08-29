package agent

import (
	"context"
	"fmt"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// Tool is gopherllm.AgenticTool under a package-local name, so this package
// can attach methods to it — Go does not allow adding methods to a type
// declared in another package. Every field gopherllm.AgenticTool has
// (Definition, Execute, Timeout, MaxResultBytes, Trusted) is present here
// too and directly settable: gopherllm.NewTool/agent.NewTool followed by
// `t.Timeout = 5 * time.Second` is the whole API, matching how every other
// field on every other exported struct in GopherLLM is set. There is
// deliberately no chainable With* method for any of them.
type Tool gopherllm.AgenticTool

// NewTool wraps gopherllm.NewTool, returning agent.Tool so the result is
// ready for WithHint and for a ToolSet without a manual conversion. See
// gopherllm.NewTool's doc comment for the full schema-derivation contract,
// including that it panics on an Args type it cannot describe.
func NewTool[Args any](name, description string, fn func(context.Context, Args) (string, error)) Tool {
	return Tool(gopherllm.NewTool(name, description, fn))
}

// RawTool lifts an existing gopherllm.AgenticTool — a hand-written schema, or
// one built elsewhere (e.g. server.NewResearchTools) — into this package's
// Tool so it can go through WithHint or a ToolSet.
func RawTool(t gopherllm.AgenticTool) Tool { return Tool(t) }

// WithHint appends text to this tool's RESULT (not its description) on
// success, or different text when the result is empty. The distinction: text
// that changes WHICH tool gets picked belongs in the description and is paid
// for on every request the model considers calling it; text that only helps
// once you are looking at THIS call's own outcome belongs in a hint and is
// paid for only when it actually fires.
func (t Tool) WithHint(onSuccess, onEmpty string) Tool {
	inner := t.Execute
	t.Execute = func(ctx context.Context, call gopherllm.ToolCall) (string, error) {
		result, err := inner(ctx, call)
		if err != nil {
			return result, err
		}
		switch {
		case strings.TrimSpace(result) == "" && onEmpty != "":
			return result + "\n" + onEmpty, nil
		case onSuccess != "":
			return result + "\n" + onSuccess, nil
		default:
			return result, nil
		}
	}
	return t
}

// ToolSet is a validated collection of tools. NewToolSet/Add turn the silent
// failures of a bare []gopherllm.AgenticTool into construction-time errors: a
// duplicate name (a bare slice + map is last-wins, so one tool silently never
// runs), an empty name (dispatched but never offered to the model, since
// gopherllm's offer-check requires a non-empty name but its dispatch does
// not), a Definition.Type that isn't "function" (same asymmetry), and a name
// reserved by this package (gopherllm.LoadSkillToolName,
// SearchDocumentsToolName) — rejected outright rather than silently shadowed.
type ToolSet struct {
	tools []Tool
	names map[string]bool
}

// NewToolSet validates and collects tools. An error here means at least one
// tool would have been broken in a way this package can detect statically;
// nothing is partially registered.
func NewToolSet(tools ...Tool) (*ToolSet, error) {
	s := &ToolSet{names: map[string]bool{}}
	if err := s.Add(tools...); err != nil {
		return nil, err
	}
	return s, nil
}

// Add validates and appends tools to the set. On the first invalid tool, the
// set is left exactly as it was before the call — either every tool in this
// call is added, or none are.
func (s *ToolSet) Add(tools ...Tool) error {
	seenInBatch := make(map[string]bool, len(tools))
	for _, t := range tools {
		name := t.Definition.Function.Name
		switch {
		case name == "":
			return fmt.Errorf("agent: tool has an empty name")
		case name == gopherllm.LoadSkillToolName || name == SearchDocumentsToolName:
			return fmt.Errorf("agent: %q is a reserved tool name", name)
		case s.names[name] || seenInBatch[name]:
			return fmt.Errorf("agent: duplicate tool name %q", name)
		case t.Definition.Type != "function":
			return fmt.Errorf("agent: tool %q has Definition.Type %q, want \"function\"", name, t.Definition.Type)
		case t.Execute == nil:
			return fmt.Errorf("agent: tool %q has no Execute function", name)
		}
		seenInBatch[name] = true
	}
	// Validated as a whole batch above before mutating anything, so a caller
	// adding several tools at once never ends up with half of them applied.
	for _, t := range tools {
		s.names[t.Definition.Function.Name] = true
		s.tools = append(s.tools, t)
	}
	return nil
}

// Names lists the registered tools' names, in registration order.
func (s *ToolSet) Names() []string {
	if s == nil {
		return nil
	}
	out := make([]string, len(s.tools))
	for i, t := range s.tools {
		out[i] = t.Definition.Function.Name
	}
	return out
}

// AgenticTools materializes the set for gopherllm.RunAgenticChatWithTools (or
// any other root-package entry point), converting each Tool back to a
// gopherllm.AgenticTool. Use it to get a validated tool set without an Agent.
func (s *ToolSet) AgenticTools() []gopherllm.AgenticTool {
	if s == nil {
		return nil
	}
	out := make([]gopherllm.AgenticTool, len(s.tools))
	for i, t := range s.tools {
		out[i] = gopherllm.AgenticTool(t)
	}
	return out
}
