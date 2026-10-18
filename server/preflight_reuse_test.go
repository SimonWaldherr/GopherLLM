package server

import (
	"context"
	"fmt"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/chatmemory"
)

// A real tiny Runner with native Mistral framing makes its existing prefix
// render-cache counters an exact render-call oracle, without production hooks.
func preflightReuseRunner(t testing.TB) *gopherllm.Runner {
	t.Helper()
	r, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUFWithContext(4096))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	vocab := []string{"<unk>", "<s>", "</s>", "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", "▁", "a", "b", "c", "d", "e", "f"}
	ids := make(map[string]uint32, len(vocab))
	for i, token := range vocab {
		ids[token] = uint32(i)
	}
	*r.Tokenizer() = gopherllm.Tokenizer{Vocab: vocab, Scores: make([]float32, len(vocab)), TokenToID: ids, Mode: gopherllm.TokenizerSentencePiece, AddBOS: true, BOSID: 1, EOSID: 2}
	r.GGUF().Metadata["tokenizer.chat_template"] = gopherllm.MetaValue{Kind: "str", Value: "[INST] [/INST]"}
	r.EnableMistralPromptPrefixCache(0)
	return r
}

func TestStreamPreflightDoesNotPlanTheInitialPromptTwice(t *testing.T) {
	for _, recall := range []bool{false, true} {
		t.Run(fmt.Sprintf("recall=%t", recall), func(t *testing.T) {
			r := preflightReuseRunner(t)
			defaults := gopherllm.DefaultGenerationOptions()
			defaults.SystemPrompt, defaults.MaxTokens, defaults.Sampler.Temperature = "a", 1, 0
			h := NewHandler(r, HandlerOptions{Defaults: defaults})
			t.Cleanup(func() { h.Close() })
			body := map[string]any{
				"messages":               []map[string]string{{"role": "user", "content": "b"}},
				"gopherllm_context_mode": "recent", "gopherllm_context_rag": recall,
				"stream": true, "max_tokens": 1,
			}
			info := contextRAGMetadata(t, postContextRAG(t, h, body), true)
			if info == nil || info["input_messages"] != float64(1) {
				t.Fatalf("prepared SSE lost final context metadata: %+v", info)
			}
			stats := r.MistralPromptPrefixCacheStats()
			// Plain planning hands its final tokens directly to generation.
			// Recall's generic Planner interface measures once and the Runner
			// renders the final selected text once for the owned token handoff.
			wantHits := uint64(0)
			if recall {
				wantHits = 1
			}
			if stats.Misses != 1 || stats.Hits != wantHits {
				t.Fatalf("initial stream repeated its context planning: %+v, want hits=%d misses=1", stats, wantHits)
			}
		})
	}
}

type countedPreflightPlanner struct {
	runner *gopherllm.Runner
	calls  int
}

func (p *countedPreflightPlanner) PrepareChatContext(messages []gopherllm.ChatMessage, options gopherllm.GenerationOptions) ([]gopherllm.ChatMessage, gopherllm.ContextWindowInfo, error) {
	p.calls++
	return p.runner.PrepareChatContext(messages, options)
}

func TestRecallPreflightConsumedOnceAndToolRoundReplanned(t *testing.T) {
	r := preflightReuseRunner(t)
	planner := &countedPreflightPlanner{runner: r}
	memory, err := chatmemory.New(context.Background(), planner, chatmemory.Options{})
	if err != nil {
		t.Fatal(err)
	}
	opts := gopherllm.DefaultGenerationOptions()
	opts.ContextWindowMode, opts.SystemPrompt, opts.MaxTokens, opts.Sampler.Temperature = gopherllm.ContextWindowRecent, "a", 1, 0
	messages := []gopherllm.ChatMessage{gopherllm.UserMessage("b")}
	plan, err := r.PrepareChatGeneration(messages, opts, memory.Prepare)
	if err != nil {
		t.Fatal(err)
	}
	wrapped := plan.Wrap(memory.Wrap(r.GenerateChatStreamUntil))
	if _, err := wrapped(messages, opts, nil); err != nil || planner.calls != 1 {
		t.Fatalf("initial recall plan repeated: calls=%d err=%v", planner.calls, err)
	}
	call := gopherllm.ToolCall{ID: "lookup", Type: "function", Function: gopherllm.ToolCallFunction{Name: "lookup", Arguments: "{}"}}
	messages = append(messages, gopherllm.AssistantToolCallMessage(call), gopherllm.ToolResultMessage(call.ID, "lookup", "c"))
	result, err := wrapped(messages, opts, nil)
	if err != nil || planner.calls != 2 || result.ContextWindow == nil || result.ContextWindow.InputMessages != len(messages) {
		t.Fatalf("tool round reused the initial recall plan: calls=%d result=%+v err=%v", planner.calls, result, err)
	}
}

func preflightReuseBenchmarkMessages() []gopherllm.ChatMessage {
	messages := []gopherllm.ChatMessage{{Role: gopherllm.ChatRoleSystem, Content: "b"}}
	for i := range 63 {
		text := strings.Repeat("f", 20)
		if i == 0 {
			text = "cedar"
		} else if i == 62 {
			text = strings.Repeat("f", 5000)
		}
		messages = append(messages, gopherllm.UserMessage(text), gopherllm.AssistantMessage("b"))
	}
	return append(messages, gopherllm.UserMessage("cedar"))
}

// Keep both paths in one binary: the baseline discards preflight then invokes
// the ordinary generator, while reuse consumes exactly that same preflight.
// Tiny weights keep the comparison focused on 128-message planning overhead.
func BenchmarkStreamingContextPreflight128(b *testing.B) {
	for _, recall := range []bool{false, true} {
		for _, reuse := range []bool{false, true} {
			b.Run(fmt.Sprintf("recall=%t/reuse=%t", recall, reuse), func(b *testing.B) {
				r, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUFWithContext(4096))
				if err != nil {
					b.Fatal(err)
				}
				b.Cleanup(func() { r.Close() })
				r.SetConversationCacheLimit(0)
				opts := gopherllm.DefaultGenerationOptions()
				opts.SystemPrompt, opts.ContextWindowMode, opts.MaxTokens, opts.Sampler.Temperature = "", gopherllm.ContextWindowRecent, 1, 0
				messages := preflightReuseBenchmarkMessages()
				if len(messages) != 128 {
					b.Fatal("benchmark must contain exactly 128 messages")
				}
				// Verify that recall-on really searches and injects a dropped turn.
				if recall {
					memory, err := chatmemory.New(context.Background(), r, chatmemory.Options{})
					if err != nil {
						b.Fatal(err)
					}
					if _, info, err := memory.Prepare(messages, opts); err != nil || info.Recall == nil {
						b.Fatalf("recall fixture is inactive: info=%+v err=%v", info, err)
					}
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					generate := gopherllm.ChatGenerator(r.GenerateChatStreamUntil)
					prepare := gopherllm.ChatContextPreparer(r.PrepareChatContext)
					if recall {
						memory, err := chatmemory.New(context.Background(), r, chatmemory.Options{})
						if err != nil {
							b.Fatal(err)
						}
						generate, prepare = memory.Wrap(generate), memory.Prepare
					}
					if reuse {
						customPrepare := prepare
						if !recall {
							customPrepare = nil // Runner's direct owned token handoff.
						}
						plan, err := r.PrepareChatGeneration(messages, opts, customPrepare)
						if err != nil {
							b.Fatal(err)
						}
						generate = plan.Wrap(generate)
					} else if _, _, err := prepare(messages, opts); err != nil {
						b.Fatal(err)
					}
					if _, err := generate(messages, opts, nil); err != nil {
						b.Fatal(err)
					}
					r.ClearConversationCache()
				}
			})
		}
	}
}
