package chatmemory_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/chatmemory"
	"github.com/SimonWaldherr/GopherLLM/internal/testmodel"
)

// bytePlanner deliberately counts every byte of the rendered request. This
// makes the memory budget independent of the index's rune/chunk estimates and
// gives the tests an exact, deterministic budget without running inference.
type bytePlanner struct {
	limit  int
	before func([]gopherllm.ChatMessage, gopherllm.GenerationOptions)
}

func (p *bytePlanner) tokens(messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions) int {
	n := len(options.SystemPrompt)
	tools, _ := json.Marshal(options.ActiveTools())
	n += len(tools)
	for _, m := range messages {
		n += 4 + len(m.Content) + len(m.ToolCallID) + len(m.Name)
		for _, call := range m.ToolCalls {
			n += len(call.ID) + len(call.Type) + len(call.Function.Name) + len(call.Function.Arguments)
		}
	}
	return n
}

func (p *bytePlanner) PrepareChatContext(messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions) ([]gopherllm.ChatMessage, gopherllm.ContextWindowInfo, error) {
	if p.before != nil {
		p.before(messages, options)
	}
	mode := options.ContextWindowMode
	if mode == "" {
		mode = gopherllm.ContextWindowFull
	}
	info := gopherllm.ContextWindowInfo{Mode: mode, ContextLength: p.limit, InputMessages: len(messages), PromptBudget: p.limit}
	kept := messages
	if mode != gopherllm.ContextWindowFull {
		info.PromptBudget -= max(1, options.MaxTokens) + 8
		if info.PromptBudget < 1 {
			return nil, info, errors.New("completion does not fit")
		}
		if p.tokens(messages, options) > info.PromptBudget {
			leading := 0
			for leading < len(messages) && messages[leading].Role == gopherllm.ChatRoleSystem {
				leading++
			}
			kept = nil
			for i := len(messages) - 1; i >= leading; i-- {
				if messages[i].Role != gopherllm.ChatRoleUser {
					continue
				}
				candidate := append(append([]gopherllm.ChatMessage(nil), messages[:leading]...), messages[i:]...)
				if p.tokens(candidate, options) > info.PromptBudget {
					break
				}
				kept = candidate
			}
			if kept == nil {
				return nil, info, errors.New("latest turn does not fit")
			}
		}
	}
	info.PromptTokens = p.tokens(kept, options)
	info.RetainedMessages, info.DroppedMessages = len(kept), len(messages)-len(kept)
	return kept, info, nil
}

func memoryOptions() gopherllm.GenerationOptions {
	return gopherllm.GenerationOptions{MaxTokens: 16, ContextWindowMode: gopherllm.ContextWindowRecent, SystemPrompt: "Keep caller instructions."}
}

func memoryHistory(secret string) []gopherllm.ChatMessage {
	return []gopherllm.ChatMessage{
		{Role: gopherllm.ChatRoleSystem, Content: "Answer using verified conversation facts."},
		gopherllm.UserMessage("Remember account GX-1180."),
		gopherllm.AssistantMessage("GX-1180 has code " + secret + "."),
		gopherllm.UserMessage(strings.Repeat("unrelated padding ", 1100)),
		gopherllm.AssistantMessage("Padding acknowledged."),
		gopherllm.UserMessage("Which code belongs to GX-1180?\nKeep this exact question."),
	}
}

func newMemory(t *testing.T, ctx context.Context, planner *bytePlanner, options chatmemory.Options) *chatmemory.Session {
	t.Helper()
	s, err := chatmemory.New(ctx, planner, options)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func lastUser(messages []gopherllm.ChatMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == gopherllm.ChatRoleUser {
			return messages[i].Content
		}
	}
	return ""
}

func copyMessages(messages []gopherllm.ChatMessage) []gopherllm.ChatMessage {
	out := append([]gopherllm.ChatMessage(nil), messages...)
	for i := range out {
		out[i].ToolCalls = append([]gopherllm.ToolCall(nil), messages[i].ToolCalls...)
		if messages[i].Images != nil {
			out[i].Images = append([]gopherllm.ImageContent(nil), messages[i].Images...)
			for j := range out[i].Images {
				out[i].Images[j].Bytes = append([]byte(nil), messages[i].Images[j].Bytes...)
			}
		}
	}
	return out
}

func assertRecallBudget(t *testing.T, p *bytePlanner, messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions, info gopherllm.ContextWindowInfo) {
	t.Helper()
	if info.PromptTokens != p.tokens(messages, options) || info.PromptTokens > info.PromptBudget {
		t.Fatalf("inexact or overflowing final budget: %+v, actual=%d", info, p.tokens(messages, options))
	}
	if info.Recall == nil {
		t.Fatal("missing recall metadata")
	}
	if info.Recall.AddedTokens < 0 || info.Recall.AddedTokens > info.Recall.TokenBudget || info.Recall.TokenBudget > info.PromptBudget/8 {
		t.Fatalf("recall exceeded its separate budget: %+v", info.Recall)
	}
}

func TestRecallUsesWholeDroppedTurnAndPreservesInstructions(t *testing.T) {
	p := &bytePlanner{limit: 16384}
	s := newMemory(t, context.Background(), p, chatmemory.Options{})
	messages, options := memoryHistory("seven"), memoryOptions()
	original := copyMessages(messages)
	got, info, err := s.Prepare(messages, options)
	if err != nil {
		t.Fatal(err)
	}
	assertRecallBudget(t, p, got, options, info)
	if len(info.Recall.Sources) != 1 || !strings.Contains(lastUser(got), "code seven") || !strings.Contains(lastUser(got), "Remember account GX-1180") {
		t.Fatalf("complete old turn was not recalled: %+v\n%s", info.Recall, lastUser(got))
	}
	if !strings.HasSuffix(lastUser(got), lastUser(original)) {
		t.Fatal("current user question was altered")
	}
	if !reflect.DeepEqual(messages, original) || !reflect.DeepEqual(got[0], original[0]) {
		t.Fatal("caller history or explicit system instruction was changed")
	}
	source := info.Recall.Sources[0]
	if source.ID == "" || source.Hash == "" || source.StartMessage != 1 || source.EndMessage != 3 || source.Score <= 0 {
		t.Fatalf("missing or wrong original-turn provenance: %+v", source)
	}
	if info.Recall.TokenBudget > 512 {
		t.Fatalf("default recall budget exceeded: %d", info.Recall.TokenBudget)
	}
}

func TestFittingHistoryAndNoHitsDoNotSlimPrompt(t *testing.T) {
	for _, tc := range []struct {
		name     string
		messages []gopherllm.ChatMessage
	}{
		{"fits", []gopherllm.ChatMessage{gopherllm.UserMessage("GX-1180"), gopherllm.AssistantMessage(strings.Repeat("x", 15700)), gopherllm.UserMessage("GX-1180?")}},
		{"no matching dropped turn", []gopherllm.ChatMessage{gopherllm.UserMessage(strings.Repeat("older ", 1000)), gopherllm.AssistantMessage("Earlier answer."), gopherllm.UserMessage(strings.Repeat("recent ", 2260)), gopherllm.AssistantMessage("Recent answer."), gopherllm.UserMessage("GX-1180?")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, options := &bytePlanner{limit: 16384}, memoryOptions()
			base, baseline, err := p.PrepareChatContext(tc.messages, options)
			if err != nil {
				t.Fatal(err)
			}
			s := newMemory(t, context.Background(), p, chatmemory.Options{MaxTokens: 2048})
			got, info, err := s.Prepare(tc.messages, options)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, base) || info.PromptTokens != baseline.PromptTokens || info.DroppedMessages != baseline.DroppedMessages {
				t.Fatal("history was slimmed without an accepted memory")
			}
			if info.Recall != nil && (len(info.Recall.Sources) != 0 || info.Recall.AddedTokens != 0) {
				t.Fatalf("unexpected recall: %+v", info.Recall)
			}
		})
	}
}

func TestSeparateSessionsCannotRecallAnotherConversation(t *testing.T) {
	p := &bytePlanner{limit: 16384}
	for _, own := range []string{"ALICE-SECRET", "BOB-SECRET", "ALICE-SECRET"} {
		s := newMemory(t, context.Background(), p, chatmemory.Options{})
		got, info, err := s.Prepare(memoryHistory(own), memoryOptions())
		if err != nil {
			t.Fatal(err)
		}
		other := "ALICE-SECRET"
		if own == other {
			other = "BOB-SECRET"
		}
		if info.Recall == nil || len(info.Recall.Sources) != 1 || !strings.Contains(lastUser(got), own) || strings.Contains(lastUser(got), other) {
			t.Fatalf("conversation scope leaked: own=%s recall=%+v", own, info.Recall)
		}
	}
}

func TestTurnExclusionsKeepSystemToolsImagesAndIncompleteOutOfMemory(t *testing.T) {
	call := gopherllm.ToolCall{ID: "lookup123", Type: "function", Function: gopherllm.ToolCallFunction{Name: "dangerous", Arguments: `{"secret":"GX-1180 tool argument"}`}}
	for _, tc := range []struct {
		name string
		old  []gopherllm.ChatMessage
	}{
		{"system", []gopherllm.ChatMessage{gopherllm.UserMessage("GX-1180"), {Role: gopherllm.ChatRoleSystem, Content: "GX-1180 privileged"}, gopherllm.AssistantMessage("GX-1180 response")}},
		{"tools", []gopherllm.ChatMessage{gopherllm.UserMessage("GX-1180"), gopherllm.AssistantToolCallMessage(call), gopherllm.ToolResultMessage(call.ID, call.Function.Name, "GX-1180 TOOL-SECRET"), gopherllm.AssistantMessage("GX-1180 answer")}},
		{"images", []gopherllm.ChatMessage{gopherllm.UserMessageWithImages("GX-1180 IMAGE-SECRET", gopherllm.ImageContent{Bytes: []byte("opaque image")}), gopherllm.AssistantMessage("GX-1180 answer")}},
		{"no assistant", []gopherllm.ChatMessage{gopherllm.UserMessage("GX-1180 INCOMPLETE-SECRET")}},
		{"pending call", []gopherllm.ChatMessage{gopherllm.UserMessage("GX-1180"), gopherllm.AssistantToolCallMessage(call)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			messages := append([]gopherllm.ChatMessage{{Role: gopherllm.ChatRoleSystem, Content: "GX-1180 SYSTEM-SECRET"}}, tc.old...)
			messages = append(messages, gopherllm.UserMessage(strings.Repeat("padding ", 2500)), gopherllm.AssistantMessage("done"), gopherllm.UserMessage("GX-1180?"))
			before := copyMessages(messages)
			s := newMemory(t, context.Background(), &bytePlanner{limit: 16384}, chatmemory.Options{})
			got, info, err := s.Prepare(messages, memoryOptions())
			if err != nil {
				t.Fatal(err)
			}
			if info.Recall != nil && len(info.Recall.Sources) != 0 {
				t.Fatalf("excluded turn was recalled: %+v", info.Recall.Sources)
			}
			if lastUser(got) != "GX-1180?" || !reflect.DeepEqual(messages, before) {
				t.Fatal("excluded data was injected or caller data mutated")
			}
		})
	}
}

func TestOversizedMatchingTurnDoesNotHideSmallerWholeTurn(t *testing.T) {
	messages := []gopherllm.ChatMessage{
		gopherllm.UserMessage(strings.Repeat("GX-1180 ", 700)), gopherllm.AssistantMessage("LARGE-TURN-END"),
		gopherllm.UserMessage("GX-1180?"), gopherllm.AssistantMessage("SMALL-COMPLETE-ANSWER"),
		gopherllm.UserMessage(strings.Repeat("padding ", 2500)), gopherllm.AssistantMessage("done"), gopherllm.UserMessage("GX-1180?"),
	}
	p := &bytePlanner{limit: 16384}
	s := newMemory(t, context.Background(), p, chatmemory.Options{MaxTokens: 1024})
	got, info, err := s.Prepare(messages, memoryOptions())
	if err != nil {
		t.Fatal(err)
	}
	assertRecallBudget(t, p, got, memoryOptions(), info)
	if len(info.Recall.Sources) != 1 || info.Recall.Sources[0].StartMessage != 2 || info.Recall.Sources[0].EndMessage != 4 || !strings.Contains(lastUser(got), "SMALL-COMPLETE-ANSWER") || strings.Contains(lastUser(got), "LARGE-TURN-END") {
		t.Fatalf("oversized hit displaced a fitting complete turn: %+v", info.Recall)
	}
}

func TestRecallIndexAndSourceLimits(t *testing.T) {
	for _, tc := range []struct {
		name    string
		options chatmemory.Options
		wantMax int
		wantNil bool
	}{
		{"turn count", chatmemory.Options{MaxTokens: 2048, MaxTurns: 1, MaxIndexedTurns: 2}, 2, false},
		{"turn bytes", chatmemory.Options{MaxTurnBytes: 8}, 0, true},
		{"index bytes", chatmemory.Options{MaxIndexBytes: 8}, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			messages := []gopherllm.ChatMessage{}
			for i := range 5 {
				messages = append(messages, gopherllm.UserMessage(fmt.Sprintf("GX-1180 fact %d?", i)), gopherllm.AssistantMessage(fmt.Sprintf("GX-1180 answer %d", i)))
			}
			messages = append(messages, gopherllm.UserMessage(strings.Repeat("padding ", 2500)), gopherllm.AssistantMessage("done"), gopherllm.UserMessage("GX-1180?"))
			p := &bytePlanner{limit: 16384}
			s := newMemory(t, context.Background(), p, tc.options)
			got, info, err := s.Prepare(messages, memoryOptions())
			if err != nil {
				t.Fatal(err)
			}
			if info.Recall != nil && info.Recall.IndexedTurns > tc.wantMax {
				t.Fatalf("indexed too many turns: %+v", info.Recall)
			}
			if tc.wantNil {
				if lastUser(got) != "GX-1180?" || info.Recall != nil && len(info.Recall.Sources) != 0 {
					t.Fatal("byte limits allowed a partial or oversized turn")
				}
			} else if info.Recall == nil || len(info.Recall.Sources) != 1 {
				t.Fatalf("source limit or fixture failed: %+v", info.Recall)
			}
		})
	}
}

func TestQueryOverridePreservesOriginalQuestion(t *testing.T) {
	messages := memoryHistory("seven")
	messages[len(messages)-1].Content = "An unrelated current question.\nVERBATIM."
	s := newMemory(t, context.Background(), &bytePlanner{limit: 16384}, chatmemory.Options{Query: "GX-1180"})
	got, info, err := s.Prepare(messages, memoryOptions())
	if err != nil {
		t.Fatal(err)
	}
	if info.Recall == nil || len(info.Recall.Sources) != 1 || !strings.Contains(lastUser(got), "code seven") || !strings.HasSuffix(lastUser(got), lastUser(messages)) {
		t.Fatalf("query override did not use original request independently: %+v", info.Recall)
	}
}

func TestSessionAllowsAppendedToolRoundsButRejectsEditedPrefix(t *testing.T) {
	messages, options := memoryHistory("seven"), memoryOptions()
	s := newMemory(t, context.Background(), &bytePlanner{limit: 16384}, chatmemory.Options{})
	_, first, err := s.Prepare(messages, options)
	if err != nil || first.Recall == nil || len(first.Recall.Sources) != 1 {
		t.Fatalf("initial recall: info=%+v err=%v", first, err)
	}
	call := gopherllm.ToolCall{ID: "lookup123", Type: "function", Function: gopherllm.ToolCallFunction{Name: "lookup", Arguments: `{"key":"other"}`}}
	withTools := append(copyMessages(messages), gopherllm.AssistantToolCallMessage(call), gopherllm.ToolResultMessage(call.ID, call.Function.Name, "A current tool result."))
	before := copyMessages(withTools)
	got, second, err := s.Prepare(withTools, options)
	if err != nil {
		t.Fatal(err)
	}
	if second.Recall == nil || len(second.Recall.Sources) != 1 || second.Recall.Sources[0].ID != first.Recall.Sources[0].ID || second.Recall.Sources[0].Hash != first.Recall.Sources[0].Hash {
		t.Fatal("tool round changed the frozen recall snapshot")
	}
	if !reflect.DeepEqual(got[len(got)-2:], before[len(before)-2:]) || !reflect.DeepEqual(withTools, before) {
		t.Fatal("current tool exchange or caller-owned messages were altered")
	}
	if strings.Count(lastUser(got), "code seven") != 1 {
		t.Fatal("recall was duplicated across tool rounds")
	}
	edited := copyMessages(messages)
	edited[2].Content = "GX-1180 now has a different code."
	if _, _, err := s.Prepare(edited, options); err == nil {
		t.Fatal("session silently accepted an edited original snapshot")
	}
	if _, _, err := s.Prepare(messages[:3], options); err == nil {
		t.Fatal("session silently accepted a deleted original snapshot")
	}
}

func TestContextCancellationDoesNotReturnRecall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := chatmemory.New(ctx, &bytePlanner{limit: 16384}, chatmemory.Options{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("New cancellation: %v", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	s := newMemory(t, ctx, &bytePlanner{limit: 16384}, chatmemory.Options{})
	cancel()
	if _, info, err := s.Prepare(memoryHistory("seven"), memoryOptions()); !errors.Is(err, context.Canceled) || info.Recall != nil && len(info.Recall.Sources) != 0 {
		t.Fatalf("Prepare cancellation: info=%+v err=%v", info, err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	p := &bytePlanner{limit: 16384}
	s = newMemory(t, ctx, p, chatmemory.Options{})
	p.before = func(_ []gopherllm.ChatMessage, options gopherllm.GenerationOptions) {
		if options.ContextWindowMode == gopherllm.ContextWindowFull {
			cancel()
		}
	}
	if _, info, err := s.Prepare(memoryHistory("seven"), memoryOptions()); !errors.Is(err, context.Canceled) || info.Recall != nil && len(info.Recall.Sources) != 0 {
		t.Fatalf("cancellation during token probe: info=%+v err=%v", info, err)
	}
}

func TestWrapForwardsGenerationAndPublishesOnlyAcceptedSources(t *testing.T) {
	p := &bytePlanner{limit: 16384}
	s := newMemory(t, context.Background(), p, chatmemory.Options{})
	messages, options := memoryHistory("seven"), memoryOptions()
	called, streamed := 0, 0
	wrapped := s.Wrap(func(got []gopherllm.ChatMessage, opts gopherllm.GenerationOptions, onToken func(string) bool) (gopherllm.GenerationResult, error) {
		called++
		if !strings.Contains(lastUser(got), "code seven") || opts.SystemPrompt != options.SystemPrompt || opts.MaxTokens != options.MaxTokens {
			t.Fatal("wrapped generation lost recall or changed generation settings")
		}
		if opts.ContextWindowMode != gopherllm.ContextWindowFull {
			t.Fatal("wrapped generator could re-plan the already measured prompt")
		}
		if onToken("answer") {
			t.Fatal("stream cancellation was not forwarded")
		}
		return gopherllm.GenerationResult{Text: "answer", FinishReason: "stop"}, nil
	})
	result, err := wrapped(messages, options, func(text string) bool {
		streamed++
		return false
	})
	if err != nil {
		t.Fatal(err)
	}
	if called != 1 || streamed != 1 || result.Text != "answer" || result.ContextWindow == nil || result.ContextWindow.Recall == nil || len(result.ContextWindow.Recall.Sources) != 1 {
		t.Fatalf("wrapped result lost metadata or generator output: calls=%d streamed=%d result=%+v", called, streamed, result)
	}
	if options.ContextWindowMode != gopherllm.ContextWindowRecent {
		t.Fatal("wrapping changed caller-owned generation settings")
	}
}

func TestActualTokenizerRecallHasExactFinalPromptBudget(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := testmodel.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	gguf, err := gopherllm.ParseGGUF(data)
	if err != nil {
		t.Fatal(err)
	}
	// Keep the actual fixture tokenizer/template and tiny weights, but allow
	// enough context for the provenance envelope's separate recall budget.
	gguf.Metadata["llama.context_length"] = gopherllm.MetaValue{Kind: "u32", Value: uint32(8192)}
	gguf.Metadata["tokenizer.chat_template"] = gopherllm.MetaValue{Kind: "str", Value: "[INST] [/INST]"}
	planned := make([]gopherllm.PlannedTensor, len(gguf.Tensors))
	for i, tensor := range gguf.Tensors {
		planned[i] = gopherllm.PlannedTensor{Name: tensor.Name, Dims: tensor.Dims, DType: tensor.DType}
	}
	var rewritten bytes.Buffer
	writer, err := gopherllm.NewGGUFWriter(&rewritten, gguf.Metadata, planned, 32)
	if err != nil {
		t.Fatal(err)
	}
	for _, tensor := range gguf.Tensors {
		size, ok := tensor.DType.DataSize(tensor.Numel())
		if !ok {
			t.Fatal("test fixture has unsupported tensor encoding")
		}
		start := gguf.DataOffset + int(tensor.Offset)
		if err := writer.WriteTensor(data[start : start+size]); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	r, err := gopherllm.RunnerFromGGUFBytes(rewritten.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	for _, mode := range []gopherllm.ContextWindowMode{gopherllm.ContextWindowRecent, gopherllm.ContextWindowAutoCompress} {
		t.Run(string(mode), func(t *testing.T) {
			s, err := chatmemory.New(context.Background(), r, chatmemory.Options{})
			if err != nil {
				t.Fatal(err)
			}
			options := memoryOptions()
			options.ContextWindowMode = mode
			messages := memoryHistory("seven; for example")
			got, info, err := s.Prepare(messages, options)
			if err != nil {
				t.Fatal(err)
			}
			if info.Recall == nil || len(info.Recall.Sources) != 1 || !strings.Contains(lastUser(got), "code seven; for example") {
				t.Fatalf("real tokenizer did not recall the original uncompressed turn: %+v", info)
			}
			options.ContextWindowMode = gopherllm.ContextWindowFull
			_, exact, err := r.PrepareChatContext(got, options)
			if err != nil {
				t.Fatal(err)
			}
			if exact.PromptTokens != info.PromptTokens || info.PromptTokens > info.PromptBudget || info.Recall.AddedTokens > info.Recall.TokenBudget {
				t.Fatalf("real renderer disagrees with recall budget: recall=%+v exact=%+v", info, exact)
			}
		})
	}
}
