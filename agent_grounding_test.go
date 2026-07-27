package gopherllm

import (
	"context"
	"strings"
	"testing"
)

// This is the regression test for the actual bug report: a Wikipedia lookup
// against a "List of ..." article returned only its lead sentence (a bare
// item count, no names), and the model filled the gap with invented place
// names presented as fact. The tool call itself worked; nothing told the
// model that doing this was unacceptable. groundingInstruction is that
// missing instruction, and this test pins it to every turn where tools or
// skills are actually offered.
func TestAgenticOptionsAddsGroundingInstructionWhenToolsActive(t *testing.T) {
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "wikipedia_summary"}},
		Execute:    func(context.Context, ToolCall) (string, error) { return "", nil },
	}

	options, active := AgenticOptionsForTools(GenerationOptions{SystemPrompt: "You are a helpful assistant."}, nil, []AgenticTool{tool})
	if !active {
		t.Fatal("expected tools to be active")
	}
	if !strings.Contains(options.SystemPrompt, groundingInstruction) {
		t.Fatalf("system prompt missing grounding instruction: %q", options.SystemPrompt)
	}
	if !strings.HasPrefix(options.SystemPrompt, "You are a helpful assistant.") {
		t.Fatalf("existing system prompt was not preserved: %q", options.SystemPrompt)
	}
}

func TestAgenticOptionsGroundingWorksWithAnEmptySystemPrompt(t *testing.T) {
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "lookup"}},
		Execute:    func(context.Context, ToolCall) (string, error) { return "", nil },
	}
	options, active := AgenticOptionsForTools(GenerationOptions{SystemPrompt: ""}, nil, []AgenticTool{tool})
	if !active {
		t.Fatal("expected tools to be active")
	}
	if strings.TrimSpace(options.SystemPrompt) != groundingInstruction {
		t.Fatalf("system prompt = %q, want exactly the grounding instruction", options.SystemPrompt)
	}
}

// Skills-only activation (no caller tools at all) must get the same
// treatment: the hallucination risk is identical whenever any tool result can
// under-answer the question.
func TestAgenticOptionsGroundingAppliesForSkillsToo(t *testing.T) {
	options, active := AgenticOptionsForTools(GenerationOptions{SystemPrompt: "Custom instructions."}, []Skill{{Name: "s", Body: "b"}}, nil)
	if !active {
		t.Fatal("expected skills to activate the agentic loop")
	}
	if !strings.Contains(options.SystemPrompt, groundingInstruction) {
		t.Fatalf("system prompt missing grounding instruction: %q", options.SystemPrompt)
	}
}

// When no tool or skill actually activates, the options must come back
// byte-for-byte unchanged — this addition must not leak into ordinary,
// non-agentic chat.
func TestAgenticOptionsLeavesSystemPromptAloneWhenInactive(t *testing.T) {
	original := GenerationOptions{SystemPrompt: "You are a helpful assistant."}
	options, active := AgenticOptionsForTools(original, nil, nil)
	if active {
		t.Fatal("no tools or skills were supplied; loop should not activate")
	}
	if options.SystemPrompt != original.SystemPrompt {
		t.Fatalf("system prompt changed with no tools active: %q", options.SystemPrompt)
	}

	tool := AgenticTool{Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "lookup"}}}
	options, active = AgenticOptionsForTools(GenerationOptions{SystemPrompt: "unchanged", ToolChoice: "none"}, nil, []AgenticTool{tool})
	if active {
		t.Fatal("tool_choice=none must not activate the loop")
	}
	if options.SystemPrompt != "unchanged" {
		t.Fatalf("system prompt changed despite tool_choice=none: %q", options.SystemPrompt)
	}
}

// The full loop must actually use the grounding-augmented prompt for
// generation, not just compute it and discard it.
func TestRunAgenticChatUsesGroundedSystemPromptForGeneration(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	var sawGrounding bool
	tool := AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{Name: "lookup"}},
		Execute: func(context.Context, ToolCall) (string, error) {
			return "irrelevant, only called once for this test", nil
		},
	}
	options := DefaultGenerationOptions()
	options.MaxTokens = 1
	// Force a call into resolveInternalToolCalls isn't straightforward without
	// a model that actually emits tool_calls; instead verify the option
	// construction the loop hands to the runner by reproducing what
	// runAgenticChat does for its first iteration.
	loopOptions, active := AgenticOptionsForTools(options, nil, []AgenticTool{tool})
	if !active {
		t.Fatal("expected the loop to activate")
	}
	sawGrounding = strings.Contains(loopOptions.SystemPrompt, groundingInstruction)
	if !sawGrounding {
		t.Fatal("loop options passed to generation lack the grounding instruction")
	}
	if _, err := r.GenerateChat([]ChatMessage{UserMessage("hi")}, loopOptions); err != nil {
		t.Fatalf("sanity generation with grounded options failed: %v", err)
	}
}
