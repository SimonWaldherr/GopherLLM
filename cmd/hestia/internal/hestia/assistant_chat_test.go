package hestia

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type testChatModel struct {
	chat func(context.Context, []gopherllm.ChatMessage, ...gopherllm.GenOption) (gopherllm.GenerationResult, error)
}

func (f testChatModel) Chat(ctx context.Context, messages []gopherllm.ChatMessage, opts ...gopherllm.GenOption) (gopherllm.GenerationResult, error) {
	return f.chat(ctx, messages, opts...)
}

func (testChatModel) Close() error { return nil }

func TestLocalAssistantChatSendsOnlyThisQuestionWithBoundedOptions(t *testing.T) {
	questions := []string{"Was ist ein Quasar?", "Was ist ein Komet?"}
	call := 0
	c := &LocalAssistantChat{model: testChatModel{chat: func(ctx context.Context, msgs []gopherllm.ChatMessage, gen ...gopherllm.GenOption) (gopherllm.GenerationResult, error) {
		if len(msgs) != 1 || msgs[0].Role != gopherllm.ChatRoleUser || msgs[0].Content != questions[call] {
			t.Fatalf("messages = %+v, expected only question %d", msgs, call)
		}
		opts := gopherllm.ApplyGenOptions(ctx, gen...)
		if opts.SystemPrompt != assistantSystemPrompt || opts.MaxTokens != assistantReplyTokens || opts.ToolChoice != "none" || len(opts.ActiveTools()) != 0 || opts.Sampler.Temperature != 0 {
			t.Fatalf("unexpected generation options: %+v", opts)
		}
		if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > assistantReplyTimeout {
			t.Fatal("chat generation deadline is missing or unbounded")
		}
		call++
		return gopherllm.GenerationResult{Text: "  Eine kurze Antwort.  ", ReasoningText: "private reasoning"}, nil
	}}}
	for _, question := range questions {
		reply, err := c.Answer(t.Context(), "  "+question)
		if err != nil || reply != "Eine kurze Antwort." {
			t.Fatalf("reply = %q, %v", reply, err)
		}
	}
	if call != len(questions) {
		t.Fatalf("chat entered %d times", call)
	}
}

func TestLocalAssistantChatRejectsUnusableQuestionsAndOutputs(t *testing.T) {
	var calls int
	c := &LocalAssistantChat{model: testChatModel{chat: func(context.Context, []gopherllm.ChatMessage, ...gopherllm.GenOption) (gopherllm.GenerationResult, error) {
		calls++
		return gopherllm.GenerationResult{Text: "Eine Aktion", ToolCalls: []gopherllm.ToolCall{{ID: "device"}}}, nil
	}}}
	for _, question := range []string{"   ", strings.Repeat("a", assistantQuestionBytes+1)} {
		if _, err := c.Answer(t.Context(), question); err == nil {
			t.Fatal("accepted an empty or oversized question")
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := c.Answer(ctx, "Was ist ein Quasar?"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled request = %v", err)
	}
	if calls != 0 {
		t.Fatal("started generation for rejected input")
	}
	if reply, err := c.Answer(t.Context(), "Was ist ein Quasar?"); reply != "" || err == nil {
		t.Fatalf("unexpected tool call propagated as %q, %v", reply, err)
	}
}
