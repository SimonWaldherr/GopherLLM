package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/chatmemory"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

// byteContextPlanner is an exact, deterministic renderer for composition tests.
// Every byte and tool schema consumes a token, so a reference can never pass a
// budget check merely because the fake omitted its framing or active tools.
type byteContextPlanner struct{ limit int }

func contextPromptBytes(messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions) int {
	n := len(options.SystemPrompt)
	if tools := options.ActiveTools(); len(tools) > 0 {
		encoded, _ := json.Marshal(tools)
		n += len(encoded)
	}
	for _, message := range messages {
		n += 8 + len(message.Content) + len(message.Name) + len(message.ToolCallID)
		for _, call := range message.ToolCalls {
			n += len(call.ID) + len(call.Function.Name) + len(call.Function.Arguments)
		}
	}
	return n
}

func (p *byteContextPlanner) PrepareChatContext(messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions) ([]gopherllm.ChatMessage, gopherllm.ContextWindowInfo, error) {
	info := gopherllm.ContextWindowInfo{Mode: options.ContextWindowMode, ContextLength: p.limit, InputMessages: len(messages)}
	if options.ContextWindowMode == "" || options.ContextWindowMode == gopherllm.ContextWindowFull {
		info.PromptBudget, info.PromptTokens, info.RetainedMessages = p.limit, contextPromptBytes(messages, options), len(messages)
		return messages, info, nil
	}
	info.PromptBudget = p.limit - max(1, options.MaxTokens) - 8
	pinned := 0
	for pinned < len(messages) && messages[pinned].Role == gopherllm.ChatRoleSystem {
		pinned++
	}
	latest := lastUserMessageIndex(messages)
	if latest < 0 {
		return nil, info, fmt.Errorf("latest user missing")
	}
	var kept []gopherllm.ChatMessage
	for start := latest; start >= pinned; start-- {
		if messages[start].Role != gopherllm.ChatRoleUser {
			continue
		}
		candidate := append(append([]gopherllm.ChatMessage(nil), messages[:pinned]...), messages[start:]...)
		n := contextPromptBytes(candidate, options)
		if n > info.PromptBudget {
			if kept == nil {
				return nil, info, fmt.Errorf("latest user turn exceeds prompt budget")
			}
			break
		}
		kept, info.PromptTokens = candidate, n
	}
	info.RetainedMessages, info.DroppedMessages = len(kept), len(messages)-len(kept)
	return kept, info, nil
}

func recentRecallOptions(options *gopherllm.GenerationOptions) {
	options.ContextWindowMode = gopherllm.ContextWindowRecent
	options.MaxTokens = 1
	options.SystemPrompt = "Preserve the caller's policy."
}

func recallHistory() []gopherllm.ChatMessage {
	return []gopherllm.ChatMessage{
		{Role: gopherllm.ChatRoleSystem, Content: "Original system instruction."},
		gopherllm.UserMessage("nebula project"),
		gopherllm.AssistantMessage("nebula-secret is cerulean."),
		gopherllm.UserMessage("billing invoice audit charge"),
		gopherllm.AssistantMessage("billing-secret is crimson."),
		gopherllm.UserMessage(strings.Repeat("x", 48000)),
		gopherllm.AssistantMessage("Unrelated oversized turn."),
		gopherllm.UserMessage("nebula"),
	}
}

func TestContextRecallRequiresBoundedModeAndPlanner(t *testing.T) {
	for _, mode := range []gopherllm.ContextWindowMode{"", gopherllm.ContextWindowFull, gopherllm.ContextWindowRecent} {
		t.Run(string(mode), func(t *testing.T) {
			a, err := newAgentWithGenerator(context.Background(), func([]gopherllm.ChatMessage, gopherllm.GenerationOptions, func(string) bool) (gopherllm.GenerationResult, error) {
				t.Fatal("invalid recall configuration reached the generator")
				return gopherllm.GenerationResult{}, nil
			}, []Option{WithContextRecall(chatmemory.Options{})})
			if err != nil {
				t.Fatal(err)
			}
			_, err = a.Ask(context.Background(), "nebula", func(options *gopherllm.GenerationOptions) { options.ContextWindowMode = mode })
			want := "recent or autoCompress"
			if mode == gopherllm.ContextWindowRecent {
				want = "planner"
			}
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("error = %v, want %q", err, want)
			}
		})
	}
}

func TestContextRecallKeepsOriginalQueryAndIsolatesChatBranches(t *testing.T) {
	index := rag.New(rag.Options{Ranking: rag.Ranking{KeywordWeight: 1}})
	if err := index.Add(context.Background(), rag.Doc{ID: "guide", Text: "nebula documentation: billing invoice audit charge"}); err != nil {
		t.Fatal(err)
	}
	messages := recallHistory()
	original := append([]gopherllm.ChatMessage(nil), messages...)
	branch := 0
	gen := func(got []gopherllm.ChatMessage, options gopherllm.GenerationOptions, _ func(string) bool) (gopherllm.GenerationResult, error) {
		if options.ContextWindowMode != gopherllm.ContextWindowFull {
			t.Fatal("already budgeted recall prompt would be reduced again by the Runner")
		}
		if !reflect.DeepEqual(got[0], original[0]) {
			t.Fatal("conversation recall changed the system message")
		}
		content := got[lastUserMessageIndex(got)].Content
		if !strings.Contains(content, retrievalPreamble) || !strings.HasSuffix(content, "nebula") {
			t.Fatalf("document prefix or original request missing: %q", content)
		}
		if branch == 0 {
			if !strings.Contains(content, "nebula-secret") || strings.Contains(content, "billing-secret") {
				t.Fatalf("conversation query included configured or document-prefix terms: %q", content)
			}
		} else if strings.Contains(content, "nebula-secret") || strings.Contains(content, "billing-secret") {
			t.Fatal("a new branch reused the preceding Chat's recalled history")
		}
		return gopherllm.GenerationResult{Text: "Answer.", FinishReason: "stop"}, nil
	}
	a, err := newAgentWithGenerator(context.Background(), gen, []Option{
		WithIndex(index), WithRetrieval(RetrievePrefix),
		WithContextRecall(chatmemory.Options{MaxTurns: 2, MaxTokens: 1500, Query: "billing"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	a.contextPlanner = &byteContextPlanner{limit: 12000}
	answer, err := a.Chat(context.Background(), messages, recentRecallOptions)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(messages, original) {
		t.Fatal("Chat mutated caller-owned history")
	}
	if answer.ContextWindow == nil || answer.ContextWindow.Recall == nil || len(answer.ContextWindow.Recall.Sources) != 1 {
		t.Fatalf("recall metadata = %+v", answer.ContextWindow)
	}
	source := answer.ContextWindow.Recall.Sources[0]
	if source.StartMessage != 1 || source.EndMessage != 3 || source.Hash == "" || source.ID == "" {
		t.Fatalf("wrong historical source: %+v", source)
	}
	if !answer.Retrieved || len(answer.Sources) != 1 || answer.Sources[0].DocID != "guide" {
		t.Fatalf("conversation recall changed document-source reporting: %+v", answer)
	}
	branch++
	answer, err = a.Chat(context.Background(), []gopherllm.ChatMessage{messages[0], gopherllm.UserMessage("nebula")}, recentRecallOptions)
	if err != nil {
		t.Fatal(err)
	}
	if answer.ContextWindow == nil || answer.ContextWindow.Recall != nil {
		t.Fatalf("new branch reported stale recall: %+v", answer.ContextWindow)
	}
	if a.contextRecall.Query != "billing" {
		t.Fatal("request query overwrote the Agent's stored options")
	}
}

func TestContextRecallReplansToolRoundsAndReportsOnlyFinalSources(t *testing.T) {
	messages := recallHistory()
	original := append([]gopherllm.ChatMessage(nil), messages...)
	raw := strings.Repeat("opaque-tool-data\n", 500)
	tool := NewTool("lookup", "Lookup facts", func(context.Context, struct{}) (string, error) { return raw, nil })
	tool.Trusted = true
	tool.MaxResultBytes = len(raw)
	call := gopherllm.ToolCall{ID: "lookup-call", Type: "function", Function: gopherllm.ToolCallFunction{Name: "lookup", Arguments: "{}"}}
	options := gopherllm.DefaultGenerationOptions()
	recentRecallOptions(&options)
	effective, _ := gopherllm.AgenticOptionsForTools(options, nil, []gopherllm.AgenticTool{gopherllm.AgenticTool(tool)})
	effective.Tools, effective.ToolChoice = nil, "none"
	// The final prompt fits its complete latest tool turn exactly. Recall fit
	// before the tool ran, but no source can be injected into the final call.
	final := []gopherllm.ChatMessage{messages[0], messages[len(messages)-1], gopherllm.AssistantToolCallMessage(call), gopherllm.ToolResultMessage(call.ID, "lookup", raw)}
	planner := &byteContextPlanner{limit: contextPromptBytes(final, effective) + options.MaxTokens + 8}
	calls := 0
	gen := func(got []gopherllm.ChatMessage, roundOptions gopherllm.GenerationOptions, _ func(string) bool) (gopherllm.GenerationResult, error) {
		calls++
		if roundOptions.ContextWindowMode != gopherllm.ContextWindowFull {
			t.Fatal("tool round would repeat bounded planning after recall injection")
		}
		if !reflect.DeepEqual(got[0], messages[0]) || !strings.HasSuffix(got[lastUserMessageIndex(got)].Content, "nebula") {
			t.Fatal("recall changed protected system or user contents")
		}
		if calls == 1 {
			if !strings.Contains(got[lastUserMessageIndex(got)].Content, "nebula-secret") || len(roundOptions.ActiveTools()) != 1 {
				t.Fatal("first tool round did not receive recalled history and its tool schema")
			}
			return gopherllm.GenerationResult{ToolCalls: []gopherllm.ToolCall{call}, FinishReason: "tool_calls"}, nil
		}
		if !reflect.DeepEqual(got, final) || len(roundOptions.ActiveTools()) != 0 {
			t.Fatalf("final round split or changed the latest tool turn: messages=%d, want %d; active tools=%d", len(got), len(final), len(roundOptions.ActiveTools()))
		}
		return gopherllm.GenerationResult{Text: "Final answer.", FinishReason: "stop"}, nil
	}
	a, err := newAgentWithGenerator(context.Background(), gen, []Option{WithTool(tool), WithMaxToolRounds(1), WithContextRecall(chatmemory.Options{})})
	if err != nil {
		t.Fatal(err)
	}
	a.contextPlanner = planner
	answer, err := a.Chat(context.Background(), messages, recentRecallOptions)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || answer.ContextWindow == nil || answer.ContextWindow.Recall != nil {
		t.Fatalf("calls=%d final context=%+v", calls, answer.ContextWindow)
	}
	if answer.Retrieved || len(answer.Sources) != 0 || !reflect.DeepEqual(messages, original) {
		t.Fatal("conversation recall leaked into document reporting or caller history")
	}
}

func TestContextRecallMetadataJSON(t *testing.T) {
	encoded, err := json.Marshal(gopherllm.ContextWindowInfo{})
	if err != nil || strings.Contains(string(encoded), "recall") {
		t.Fatalf("ordinary context metadata gained recall: %s (%v)", encoded, err)
	}
	info := gopherllm.ContextWindowInfo{Recall: &gopherllm.ContextRecallInfo{Mode: "bm25", IndexedTurns: 2, TokenBudget: 64, AddedTokens: 32, Sources: []gopherllm.ContextRecallSource{{ID: "history:1", Turn: 1, StartMessage: 2, EndMessage: 4, Hash: "abc", Score: 0.5}}}}
	encoded, err = json.Marshal(info)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatal(err)
	}
	recall := got["recall"].(map[string]any)
	for _, key := range []string{"mode", "indexed_turns", "token_budget", "added_tokens", "sources"} {
		if _, ok := recall[key]; !ok {
			t.Fatalf("missing JSON field %q: %s", key, encoded)
		}
	}
	source := recall["sources"].([]any)[0].(map[string]any)
	for _, key := range []string{"id", "turn", "start_message", "end_message", "hash", "score"} {
		if _, ok := source[key]; !ok {
			t.Fatalf("missing source JSON field %q: %s", key, encoded)
		}
	}
}
