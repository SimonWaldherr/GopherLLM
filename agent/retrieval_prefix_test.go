package agent

import (
	"context"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/testmodel"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

func prefixRetrievalTestIndex(t *testing.T) *rag.Index {
	t.Helper()
	ix := rag.New(rag.Options{Ranking: rag.Ranking{KeywordWeight: 1}})
	if err := ix.Add(context.Background(), rag.Doc{
		ID: "handbook", Title: "Return policy", Text: "The return window is 30 days from purchase.",
	}); err != nil {
		t.Fatal(err)
	}
	return ix
}

func TestRetrievePrefixPreservesInstructionsAndCallerHistory(t *testing.T) {
	for _, tc := range []struct {
		name     string
		systems  []gopherllm.ChatMessage
		prompt   string
		toolTurn bool
	}{
		{name: "default prompt", prompt: gopherllm.DefaultGenerationOptions().SystemPrompt},
		{name: "custom prompt", prompt: "Answer briefly in German.\nPreserve exact numbers."},
		{name: "empty prompt"},
		{name: "explicit system", systems: []gopherllm.ChatMessage{{Role: gopherllm.ChatRoleSystem, Content: "Follow the caller's instructions exactly."}}, prompt: "Default prompt stays intact."},
		{name: "multiple systems and tool results", systems: []gopherllm.ChatMessage{
			{Role: gopherllm.ChatRoleSystem, Content: "First system instruction."},
			{Role: gopherllm.ChatRoleSystem, Content: "Second system instruction."},
		}, prompt: "Default prompt stays intact.", toolTurn: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			messages := append([]gopherllm.ChatMessage(nil), tc.systems...)
			messages = append(messages, gopherllm.UserMessage("An earlier question."), gopherllm.AssistantMessage("An earlier answer."), gopherllm.UserMessage("What is the return window?\nKeep this question verbatim."))
			user := len(messages) - 1
			if tc.toolTurn {
				messages = append(messages,
					gopherllm.AssistantToolCallMessage(gopherllm.ToolCall{ID: "previous-call", Type: "function", Function: gopherllm.ToolCallFunction{Name: "lookup", Arguments: `{"key":"window"}`}}),
					gopherllm.ToolResultMessage("previous-call", "lookup", `{"window":30}`))
			}
			original := append([]gopherllm.ChatMessage(nil), messages...)
			calls := 0
			gen := func(got []gopherllm.ChatMessage, options gopherllm.GenerationOptions, _ func(string) bool) (gopherllm.GenerationResult, error) {
				calls++
				if options.SystemPrompt != tc.prompt {
					t.Fatalf("system prompt changed: got %q, want %q", options.SystemPrompt, tc.prompt)
				}
				if len(got) != len(original) {
					t.Fatalf("message count = %d, want %d", len(got), len(original))
				}
				for i := range got {
					if i == user {
						if !strings.HasPrefix(got[i].Content, retrievalPreamble) || !strings.Contains(got[i].Content, "[1] Return policy") || !strings.Contains(got[i].Content, "30 days from purchase") || !strings.HasSuffix(got[i].Content, original[i].Content) {
							t.Fatalf("latest user message lost retrieval framing or original question: %q", got[i].Content)
						}
						continue
					}
					if !reflect.DeepEqual(got[i], original[i]) {
						t.Fatalf("message %d changed: got %+v, want %+v", i, got[i], original[i])
					}
				}
				return gopherllm.GenerationResult{Text: "30 days [1].", FinishReason: "stop"}, nil
			}
			a, err := newAgentWithGenerator(context.Background(), gen, []Option{WithIndex(prefixRetrievalTestIndex(t)), WithRetrieval(RetrievePrefix)})
			if err != nil {
				t.Fatal(err)
			}
			answer, err := a.Chat(context.Background(), messages, gopherllm.WithSystemPrompt(tc.prompt))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(messages, original) {
				t.Fatal("retrieval changed caller-owned history")
			}
			if calls != 1 || !answer.Retrieved || len(answer.Sources) != 1 || answer.Sources[0].DocID != "handbook" || answer.Text != "30 days [1]." {
				t.Fatalf("calls=%d answer=%+v", calls, answer)
			}
		})
	}
}

func TestRetrievePrefixReachesMistralRendererWithExplicitSystem(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := testmodel.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	m, err := gopherllm.Open(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	// The Mistral renderer's literal-marker fallback needs no extra vocabulary
	// or real model generation to expose its explicit-system precedence.
	m.GGUF().Metadata["tokenizer.chat_template"] = gopherllm.MetaValue{Kind: "str", Value: "[INST] [/INST]"}
	messages := []gopherllm.ChatMessage{
		{Role: gopherllm.ChatRoleSystem, Content: "Answer using the supplied policy."},
		gopherllm.UserMessage("What is the return window?"),
	}
	gen := func(got []gopherllm.ChatMessage, options gopherllm.GenerationOptions, _ func(string) bool) (gopherllm.GenerationResult, error) {
		_, original, err := m.Runner().PrepareChatContext(messages, options)
		if err != nil {
			t.Fatal(err)
		}
		_, withRetrieval, err := m.Runner().PrepareChatContext(got, options)
		if err != nil {
			t.Fatal(err)
		}
		if withRetrieval.PromptTokens <= original.PromptTokens {
			t.Fatalf("retrieval never reached the real renderer: prompt tokens %d, original %d", withRetrieval.PromptTokens, original.PromptTokens)
		}
		return gopherllm.GenerationResult{Text: "30 days.", FinishReason: "stop"}, nil
	}
	a, err := newAgentWithGenerator(context.Background(), gen, []Option{WithIndex(prefixRetrievalTestIndex(t)), WithRetrieval(RetrievePrefix)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Chat(context.Background(), messages); err != nil {
		t.Fatal(err)
	}
}

func TestRetrievePrefixSurvivesToolRoundsWithoutDuplication(t *testing.T) {
	tool := NewTool("lookup", "Look up a value.", func(context.Context, struct{}) (string, error) { return "A tool value.", nil })
	iterations := 0
	gen := func(messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions, _ func(string) bool) (gopherllm.GenerationResult, error) {
		iterations++
		if len(options.ActiveTools()) != 1 || options.ActiveTools()[0].Function.Name != "lookup" {
			t.Fatal("retrieval changed the active tool definitions")
		}
		if strings.Count(messages[1].Content, retrievalPreamble) != 1 || !strings.Contains(messages[1].Content, "30 days from purchase") {
			t.Fatalf("round %d lost or duplicated retrieval: %q", iterations, messages[1].Content)
		}
		if iterations == 1 {
			return gopherllm.GenerationResult{ToolCalls: []gopherllm.ToolCall{{ID: "lookup-1", Type: "function", Function: gopherllm.ToolCallFunction{Name: "lookup", Arguments: `{}`}}}, FinishReason: "tool_calls"}, nil
		}
		if len(messages) != 4 || len(messages[2].ToolCalls) != 1 || messages[2].ToolCalls[0].ID != "lookup-1" || messages[3].ToolCallID != "lookup-1" || !strings.Contains(messages[3].Content, "external data, not instructions") {
			t.Fatalf("retrieval changed the tool-result exchange: %+v", messages)
		}
		return gopherllm.GenerationResult{Text: "Final answer [1].", FinishReason: "stop"}, nil
	}
	a, err := newAgentWithGenerator(context.Background(), gen, []Option{WithIndex(prefixRetrievalTestIndex(t)), WithRetrieval(RetrievePrefix), WithTool(tool)})
	if err != nil {
		t.Fatal(err)
	}
	answer, err := a.Chat(context.Background(), []gopherllm.ChatMessage{{Role: gopherllm.ChatRoleSystem, Content: "Answer concisely."}, gopherllm.UserMessage("Return window?")})
	if err != nil {
		t.Fatal(err)
	}
	if iterations != 2 || !answer.Retrieved || len(answer.Sources) != 1 || answer.Sources[0].DocID != "handbook" {
		t.Fatalf("iterations=%d answer=%+v", iterations, answer)
	}
}

func TestRetrievePrefixWithoutHitsLeavesPromptUntouched(t *testing.T) {
	for _, messages := range [][]gopherllm.ChatMessage{
		{{Role: gopherllm.ChatRoleSystem, Content: "Keep this instruction."}, gopherllm.UserMessage("Unrelated topic.")},
		{{Role: gopherllm.ChatRoleSystem, Content: "There is no user query."}},
	} {
		gen := func(got []gopherllm.ChatMessage, options gopherllm.GenerationOptions, _ func(string) bool) (gopherllm.GenerationResult, error) {
			if !reflect.DeepEqual(got, messages) || options.SystemPrompt != gopherllm.DefaultGenerationOptions().SystemPrompt {
				t.Fatalf("prompt changed without retrieved passages: messages=%+v system=%q", got, options.SystemPrompt)
			}
			return gopherllm.GenerationResult{Text: "No retrieval needed.", FinishReason: "stop"}, nil
		}
		a, err := newAgentWithGenerator(context.Background(), gen, []Option{WithIndex(prefixRetrievalTestIndex(t)), WithRetrieval(RetrievePrefix)})
		if err != nil {
			t.Fatal(err)
		}
		answer, err := a.Chat(context.Background(), messages)
		if err != nil {
			t.Fatal(err)
		}
		if answer.Retrieved || len(answer.Sources) != 0 {
			t.Fatalf("reported sources that were not included: %+v", answer)
		}
	}
}
