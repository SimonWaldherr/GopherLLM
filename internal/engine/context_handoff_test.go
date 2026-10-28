package engine

import (
	"slices"
	"strings"
	"testing"
)

func TestBoundedGenerationTokenHandoffMatchesDirectPrompt(t *testing.T) {
	for _, mode := range []ContextWindowMode{ContextWindowRecent, ContextWindowAutoCompress} {
		t.Run(string(mode), func(t *testing.T) {
			r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			cold, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
			if err != nil {
				t.Fatal(err)
			}
			defer cold.Close()
			opts := recentContextWindowOptions()
			opts.ContextWindowMode = mode
			opts.MaxTokens, opts.Seed, opts.Sampler.Temperature = 3, 17, 0
			messages := []ChatMessage{
				{Role: ChatRoleSystem, Content: "c"},
				UserMessage(strings.Repeat("a b ", 100)),
				AssistantMessage("An application programming interface is useful for example."),
				UserMessage("d e f"),
			}
			r.config.MaxSeqLen = contextWindowLimitFor(r, []ChatMessage{messages[0], messages[len(messages)-1]}, opts)
			cold.config.MaxSeqLen = r.config.MaxSeqLen
			kept, info, err := r.PrepareChatContext(messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			if info.DroppedMessages == 0 {
				t.Fatal("fixture did not exercise bounded selection")
			}
			got, err := r.GenerateChat(messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			opts.ContextWindowMode = ContextWindowFull
			want, err := cold.GenerateChat(kept, opts)
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != want.Text || got.Stats.PromptTokens != want.Stats.PromptTokens ||
				!slices.Equal(r.prefixCache.promptLogits, cold.prefixCache.promptLogits) ||
				!slices.Equal(r.prefixCache.tokens, cold.prefixCache.tokens) {
				t.Fatal("prepared-token generation differs from directly rendered selected prompt")
			}
		})
	}
}

func TestBoundedTokenHandoffLeavesImageRenderingToGeneration(t *testing.T) {
	r := newContextWindowTestRunner(t)
	options := recentContextWindowOptions()
	messages := []ChatMessage{UserMessage("a b")}
	messages[0].Images = []ImageContent{{Bytes: []byte("requires the vision renderer")}}
	tokens := []uint32{123}
	if _, _, err := r.prepareChatContext(messages, options, &tokens); err != nil {
		t.Fatal(err)
	}
	if tokens != nil {
		t.Fatal("text token handoff bypassed the generation renderer for an image prompt")
	}
}

// Compare the caller's historical plan-then-render sequence with generation's
// direct handoff of the already-rendered final prompt. Both use the same
// planner and exact tokenizer; model inference is deliberately excluded.
func BenchmarkContextPlanningTokenHandoff(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	r.config.MaxSeqLen = 1 << 20
	messages := []ChatMessage{{Role: ChatRoleSystem, Content: "c"}}
	for range 64 {
		messages = append(messages, UserMessage(strings.Repeat("a b ", 16)), AssistantMessage(strings.Repeat("d e ", 16)))
	}
	opts := recentContextWindowOptions()
	b.Run("plan_then_render", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			kept, _, err := r.PrepareChatContext(messages, opts)
			if err != nil {
				b.Fatal(err)
			}
			_ = r.renderMessages(kept, opts.SystemPrompt, opts.ActiveTools())
		}
	})
	b.Run("token_handoff", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			var tokens []uint32
			if _, _, err := r.prepareChatContext(messages, opts, &tokens); err != nil || len(tokens) == 0 {
				b.Fatalf("handoff returned %d tokens, error=%v", len(tokens), err)
			}
		}
	})
}
