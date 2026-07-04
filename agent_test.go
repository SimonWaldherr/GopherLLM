package gopherllm

import (
	"strings"
	"testing"
)

func TestResolveSkillCallsAllKnown(t *testing.T) {
	skills := []Skill{{Name: "pdf-fill", Body: "fill instructions"}, {Name: "git-review", Body: "review instructions"}}
	calls := []ToolCall{
		{ID: "id1", Function: ToolCallFunction{Name: LoadSkillToolName, Arguments: `{"name":"pdf-fill"}`}},
		{ID: "id2", Function: ToolCallFunction{Name: LoadSkillToolName, Arguments: `{"name":"git-review"}`}},
	}
	msgs, ok := resolveSkillCalls(calls, skills)
	if !ok {
		t.Fatal("ok=false, want true")
	}
	if len(msgs) != 3 {
		t.Fatalf("len = %d, want 3 (1 assistant + 2 tool results)", len(msgs))
	}
	if msgs[0].Role != ChatRoleAssistant || len(msgs[0].ToolCalls) != 2 {
		t.Fatalf("msgs[0] = %+v", msgs[0])
	}
	if msgs[1].Role != ChatRoleTool || msgs[1].ToolCallID != "id1" || msgs[1].Content != "fill instructions" {
		t.Fatalf("msgs[1] = %+v", msgs[1])
	}
	if msgs[2].Role != ChatRoleTool || msgs[2].ToolCallID != "id2" || msgs[2].Content != "review instructions" {
		t.Fatalf("msgs[2] = %+v", msgs[2])
	}
}

func TestResolveSkillCallsUnknownNameSelfCorrects(t *testing.T) {
	// A load_skill call for a name that doesn't exist is still resolved
	// internally (not punted to the caller), with an error result the model
	// can use to try again.
	skills := []Skill{{Name: "pdf-fill", Body: "fill instructions"}}
	calls := []ToolCall{{ID: "id1", Function: ToolCallFunction{Name: LoadSkillToolName, Arguments: `{"name":"nope"}`}}}
	msgs, ok := resolveSkillCalls(calls, skills)
	if !ok {
		t.Fatal("ok=false, want true (still resolved internally)")
	}
	if len(msgs) != 2 || !strings.Contains(msgs[1].Content, "nope") || !strings.Contains(msgs[1].Content, "pdf-fill") {
		t.Fatalf("msgs = %+v", msgs)
	}
}

func TestResolveSkillCallsNonLoadSkillCallGoesToCaller(t *testing.T) {
	skills := []Skill{{Name: "pdf-fill", Body: "fill instructions"}}
	calls := []ToolCall{{ID: "id1", Function: ToolCallFunction{Name: "get_weather"}}}
	if _, ok := resolveSkillCalls(calls, skills); ok {
		t.Fatal("expected ok=false for a non-load_skill call")
	}
}

func TestResolveSkillCallsMixedGoesToCaller(t *testing.T) {
	// A turn that mixes a skill call with a caller tool call must be left
	// entirely to the caller, not partially resolved.
	skills := []Skill{{Name: "pdf-fill", Body: "fill instructions"}}
	calls := []ToolCall{
		{ID: "id1", Function: ToolCallFunction{Name: LoadSkillToolName, Arguments: `{"name":"pdf-fill"}`}},
		{ID: "id2", Function: ToolCallFunction{Name: "get_weather"}},
	}
	if _, ok := resolveSkillCalls(calls, skills); ok {
		t.Fatal("expected ok=false for a mixed skill/non-skill turn")
	}
}

func TestResolveSkillCallsEmptyFails(t *testing.T) {
	if _, ok := resolveSkillCalls(nil, []Skill{{Name: "a"}}); ok {
		t.Fatal("expected ok=false for no calls at all")
	}
}

func TestSumGenerationStats(t *testing.T) {
	a := GenerationStats{PromptTokens: 5, GeneratedTokens: 10}
	b := GenerationStats{PromptTokens: 3, GeneratedTokens: 7}
	sum := sumGenerationStats(a, b)
	if sum.PromptTokens != 8 || sum.GeneratedTokens != 17 {
		t.Fatalf("sum = %+v", sum)
	}
}

func TestRunAgenticChatNoSkillsIsPassthrough(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 4
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1

	var streamed string
	res, err := RunAgenticChat(r, []ChatMessage{UserMessage("hi")}, opts, nil, func(s string) bool {
		streamed += s
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	if streamed != res.Text {
		t.Fatalf("passthrough should stream incrementally: streamed=%q result=%q", streamed, res.Text)
	}
}

func TestRunAgenticChatCallerToolsWithoutSkillsBuffers(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 4
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	opts.Tools = []ToolDefinition{{Type: "function", Function: ToolFunctionDef{Name: "get_weather"}}}

	calls := 0
	res, err := RunAgenticChat(r, []ChatMessage{UserMessage("hi")}, opts, nil, func(s string) bool {
		calls++
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls > 1 {
		t.Fatalf("onToken called %d times, want at most 1 when tools are active (must not leak raw content mid-stream)", calls)
	}
	if calls == 1 && res.Text == "" {
		t.Fatal("onToken fired but result text is empty")
	}
}

func TestRunAgenticChatToolChoiceNoneSkipsSkills(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 4
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	opts.ToolChoice = "none"

	skills := []Skill{{Name: "s", Description: "d", Body: "b"}}
	var streamed string
	res, err := RunAgenticChat(r, []ChatMessage{UserMessage("hi")}, opts, skills, func(s string) bool {
		streamed += s
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	if streamed != res.Text {
		t.Fatalf("tool_choice=none should bypass the agentic loop entirely: streamed=%q result=%q", streamed, res.Text)
	}
}
