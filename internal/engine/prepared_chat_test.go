package engine

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func preparedChatTestRunner(t *testing.T) (*Runner, GenerationOptions) {
	t.Helper()
	r := newTinyMistralKVPrefixRunner(t)
	t.Cleanup(func() { r.Close() })
	r.config.MaxSeqLen = 4096
	opts := tinyMistralKVPrefixOptions()
	opts.ContextWindowMode, opts.SystemPrompt = ContextWindowRecent, "a"
	return r, opts
}

func TestPreparedChatConsumesOneExactPlanAndOwnedTokens(t *testing.T) {
	r, opts := preparedChatTestRunner(t)
	r.EnableMistralPromptPrefixCache(0)
	messages := []ChatMessage{UserMessage("b"), AssistantMessage("c"), UserMessage("d")}
	plan, err := r.PrepareChatGeneration(messages, opts, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The whole fitting history needs one render, and the exact first call
	// must neither plan nor render it again.
	if stats := r.MistralPromptPrefixCacheStats(); stats.Misses != 1 || stats.Hits != 0 {
		t.Fatalf("preflight render count: %+v", stats)
	}
	fallbackCalls := 0
	wrapped := plan.Wrap(func([]ChatMessage, GenerationOptions, func(string) bool) (GenerationResult, error) {
		fallbackCalls++
		return GenerationResult{Text: "fallback"}, nil
	})
	got, err := wrapped(messages, opts, nil)
	if err != nil || fallbackCalls != 0 || got.ContextWindow == nil {
		t.Fatalf("first call: result=%+v calls=%d err=%v", got, fallbackCalls, err)
	}
	if stats := r.MistralPromptPrefixCacheStats(); stats.Hits != 0 {
		t.Fatalf("generation rendered the prepared text again: %+v", stats)
	}
	logits, resident := slices.Clone(r.prefixCache.promptLogits), slices.Clone(r.prefixCache.tokens)
	r.ClearConversationCache()
	want, err := r.GenerateChat(messages, opts)
	if err != nil || got.Text != want.Text || !slices.Equal(logits, r.prefixCache.promptLogits) || !slices.Equal(resident, r.prefixCache.tokens) {
		t.Fatalf("prepared generation differs from cold direct generation: got=%+v want=%+v err=%v", got, want, err)
	}
	if got, err = wrapped(messages, opts, nil); err != nil || got.Text != "fallback" || fallbackCalls != 1 {
		t.Fatalf("second call reused a consumed plan: result=%+v calls=%d err=%v", got, fallbackCalls, err)
	}
	if plan.data != nil {
		t.Fatal("consumed plan retained messages and tokens")
	}
}

func TestPreparedChatRejectsEveryChangedInputAndNestedOption(t *testing.T) {
	r, base := preparedChatTestRunner(t)
	base.ToolChoice = "none"
	base.StopSequences = []string{"stop"}
	base.Tools = []ToolDefinition{{Type: "function", Function: ToolFunctionDef{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}}}
	for _, tc := range []struct {
		name   string
		mutate func([]ChatMessage, *GenerationOptions)
	}{
		{"content", func(m []ChatMessage, _ *GenerationOptions) { m[0].Content = "c" }},
		{"role", func(m []ChatMessage, _ *GenerationOptions) { m[0].Role = ChatRoleAssistant }},
		{"message name", func(m []ChatMessage, _ *GenerationOptions) { m[0].Name = "changed" }},
		{"result ID", func(m []ChatMessage, _ *GenerationOptions) { m[0].ToolCallID = "changed" }},
		{"call ID", func(m []ChatMessage, _ *GenerationOptions) { m[0].ToolCalls[0].ID = "changed" }},
		{"call name", func(m []ChatMessage, _ *GenerationOptions) { m[0].ToolCalls[0].Function.Name = "changed" }},
		{"call arguments", func(m []ChatMessage, _ *GenerationOptions) { m[0].ToolCalls[0].Function.Arguments = "changed" }},
		{"tools", func(_ []ChatMessage, o *GenerationOptions) { o.Tools[0].Function.Parameters[2] = 'X' }},
		{"tool choice", func(_ []ChatMessage, o *GenerationOptions) { o.ToolChoice = "auto" }},
		{"stops", func(_ []ChatMessage, o *GenerationOptions) { o.StopSequences[0] = "changed" }},
		{"system", func(_ []ChatMessage, o *GenerationOptions) { o.SystemPrompt = "c" }},
		{"reserve", func(_ []ChatMessage, o *GenerationOptions) { o.MaxTokens++ }},
		{"mode", func(_ []ChatMessage, o *GenerationOptions) { o.ContextWindowMode = ContextWindowAutoCompress }},
		{"sampler", func(_ []ChatMessage, o *GenerationOptions) { o.Sampler.RepeatPenalty++ }},
		{"seed", func(_ []ChatMessage, o *GenerationOptions) { o.Seed++ }},
		{"rounds", func(_ []ChatMessage, o *GenerationOptions) { o.MaxToolRounds++ }},
		{"MTP", func(_ []ChatMessage, o *GenerationOptions) { o.MTPDraftTokens++ }},
		{"prompt lookup", func(_ []ChatMessage, o *GenerationOptions) { o.PromptLookupTokens++ }},
		{"FIM", func(_ []ChatMessage, o *GenerationOptions) { o.FIMSuffix = "c" }},
		{"context", func(_ []ChatMessage, o *GenerationOptions) {
			*o = o.WithContext(context.WithValue(context.Background(), struct{}{}, true))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := clonePreparedChatOptions(base)
			messages := []ChatMessage{UserMessage("b")}
			messages[0].ToolCalls = []ToolCall{{ID: "call", Type: "function", Function: ToolCallFunction{Name: "lookup", Arguments: "{}"}}}
			plan, err := r.PrepareChatGeneration(messages, opts, nil)
			if err != nil {
				t.Fatal(err)
			}
			originalMessages, originalOptions := clonePreparedChatMessages(messages), clonePreparedChatOptions(opts)
			tc.mutate(messages, &opts)
			calls := 0
			wrapped := plan.Wrap(func(got []ChatMessage, options GenerationOptions, _ func(string) bool) (GenerationResult, error) {
				calls++
				if calls == 1 && (!reflect.DeepEqual(got, messages) || !reflect.DeepEqual(options, opts)) {
					t.Fatal("fallback lost the changed request")
				}
				return GenerationResult{}, nil
			})
			if _, err := wrapped(messages, opts, nil); err != nil {
				t.Fatal(err)
			}
			if _, err := wrapped(originalMessages, originalOptions, nil); err != nil || calls != 2 {
				t.Fatalf("mismatch did not consume the plan: calls=%d err=%v", calls, err)
			}
		})
	}
}

func TestPreparedChatToolRoundReplansAndKeepsFinalMetadata(t *testing.T) {
	r, opts := preparedChatTestRunner(t)
	messages := []ChatMessage{UserMessage("b")}
	preparations := 0
	prepare := func(m []ChatMessage, o GenerationOptions) ([]ChatMessage, ContextWindowInfo, error) {
		preparations++
		return r.PrepareChatContext(m, o)
	}
	plan, err := r.PrepareChatGeneration(messages, opts, prepare)
	if err != nil {
		t.Fatal(err)
	}
	wrapped := plan.Wrap(func(m []ChatMessage, o GenerationOptions, onToken func(string) bool) (GenerationResult, error) {
		if _, _, err := prepare(m, o); err != nil {
			return GenerationResult{}, err
		}
		return r.GenerateChatStreamUntil(m, o, onToken)
	})
	if _, err := wrapped(messages, opts, nil); err != nil || preparations != 1 {
		t.Fatalf("first call repeated preflight: prepares=%d err=%v", preparations, err)
	}
	call := ToolCall{ID: "lookup", Type: "function", Function: ToolCallFunction{Name: "lookup", Arguments: "{}"}}
	next := append(clonePreparedChatMessages(messages), AssistantToolCallMessage(call), ToolResultMessage(call.ID, "lookup", "c"))
	result, err := wrapped(next, opts, nil)
	if err != nil || preparations != 2 || result.ContextWindow == nil || result.ContextWindow.InputMessages != len(next) {
		t.Fatalf("tool round reused the initial plan: prepares=%d result=%+v err=%v", preparations, result, err)
	}
}

func TestPreparedChatConcurrentConsumeAndCancellation(t *testing.T) {
	r, opts := preparedChatTestRunner(t)
	messages := []ChatMessage{UserMessage("b")}
	plan, err := r.PrepareChatGeneration(messages, opts, nil)
	if err != nil {
		t.Fatal(err)
	}
	var fallbacks atomic.Int32
	wrapped := plan.Wrap(func([]ChatMessage, GenerationOptions, func(string) bool) (GenerationResult, error) {
		fallbacks.Add(1)
		return GenerationResult{}, nil
	})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := wrapped(messages, opts, nil); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if fallbacks.Load() != 7 {
		t.Fatalf("single-use fallback count=%d, want 7", fallbacks.Load())
	}
	ctx, cancel := context.WithCancel(context.Background())
	opts = opts.WithContext(ctx)
	plan, err = r.PrepareChatGeneration(messages, opts, nil)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := plan.Wrap(nil)(messages, opts, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("prepared generation ignored cancellation: %v", err)
	}
	if _, err := r.PrepareChatGeneration(messages, opts, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("preflight ignored cancellation: %v", err)
	}
}

type preparedChatSliceContext struct {
	context.Context
	values []int
}

func TestPreparedChatRequiresTheSameRequestContext(t *testing.T) {
	key := struct{}{}
	first := context.WithValue(context.Background(), key, "request")
	second := context.WithValue(context.Background(), key, "request")
	uncomparable := preparedChatSliceContext{Context: context.Background(), values: []int{1}}
	for _, tc := range []struct {
		name      string
		prepared  context.Context
		requested context.Context
		wantReuse bool
	}{
		{name: "unset", wantReuse: true},
		{name: "same context", prepared: first, requested: first, wantReuse: true},
		{name: "equal contents with different identity", prepared: first, requested: second},
		{name: "uncomparable context", prepared: uncomparable, requested: uncomparable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, opts := preparedChatTestRunner(t)
			messages := []ChatMessage{UserMessage("b")}
			plan, err := r.PrepareChatGeneration(messages, opts.WithContext(tc.prepared), nil)
			if err != nil {
				t.Fatal(err)
			}
			fallbackCalls := 0
			wrapped := plan.Wrap(func(_ []ChatMessage, got GenerationOptions, _ func(string) bool) (GenerationResult, error) {
				fallbackCalls++
				if !reflect.DeepEqual(got.ctx, tc.requested) {
					t.Fatal("fallback lost the requested context")
				}
				return GenerationResult{}, nil
			})
			if _, err := wrapped(messages, opts.WithContext(tc.requested), nil); err != nil {
				t.Fatal(err)
			}
			if (fallbackCalls == 0) != tc.wantReuse {
				t.Fatalf("fallback calls=%d, want reuse=%t", fallbackCalls, tc.wantReuse)
			}
		})
	}
}

func TestPreparedChatKeepsSnapshotFenceFromBeforePreflight(t *testing.T) {
	for _, during := range []bool{false, true} {
		for _, action := range []string{"clear", "disable-enable"} {
			t.Run(action+map[bool]string{false: "/after", true: "/during"}[during], func(t *testing.T) {
				r, opts := preparedChatTestRunner(t)
				r.EnableMistralKVPrefixCache(0)
				invalidate := func() {
					if action == "clear" {
						r.ClearMistralKVPrefixCache()
					} else {
						r.DisableMistralKVPrefixCache()
						r.EnableMistralKVPrefixCache(0)
					}
				}
				prepare := func(m []ChatMessage, o GenerationOptions) ([]ChatMessage, ContextWindowInfo, error) {
					if during {
						invalidate()
					}
					return r.PrepareChatContext(m, o)
				}
				messages := []ChatMessage{UserMessage("b")}
				plan, err := r.PrepareChatGeneration(messages, opts, prepare)
				if err != nil {
					t.Fatal(err)
				}
				if !during {
					invalidate()
				}
				if _, err := plan.Wrap(nil)(messages, opts, nil); err != nil {
					t.Fatal(err)
				}
				if stats := r.MistralKVPrefixCacheStats(); stats.Entries != 0 || stats.Bytes != 0 {
					t.Fatalf("preflight crossed the cache invalidation fence: %+v", stats)
				}
			})
		}
	}
}

func TestPreparedChatImagesAndFIMKeepGenerationFallback(t *testing.T) {
	r, opts := preparedChatTestRunner(t)
	images := []ChatMessage{UserMessageWithImages("b", ImageContent{Bytes: []byte("not an image")})}
	if plan, err := r.PrepareChatGeneration(images, opts, nil); err != nil || plan != nil {
		t.Fatalf("image preflight should retain the generation renderer: plan=%v err=%v", plan, err)
	}
	opts.ContextWindowMode, opts.FIMSuffix = ContextWindowFull, "c"
	plan, err := r.PrepareChatGeneration([]ChatMessage{UserMessage("b")}, opts, nil)
	if err != nil || plan != nil {
		t.Fatalf("FIM preflight should retain the generation renderer: plan=%v err=%v", plan, err)
	}
	calls := 0
	wrapped := plan.Wrap(func([]ChatMessage, GenerationOptions, func(string) bool) (GenerationResult, error) {
		calls++
		return GenerationResult{}, nil
	})
	if _, err := wrapped(images, opts, nil); err != nil || calls != 1 {
		t.Fatalf("nil plan did not preserve fallback: calls=%d err=%v", calls, err)
	}
	opts.ContextWindowMode = ContextWindowRecent
	if _, err := r.PrepareChatGeneration([]ChatMessage{UserMessage("b")}, opts, nil); err == nil || !strings.Contains(err.Error(), "FIM") {
		t.Fatalf("bounded FIM lost its preflight error: %v", err)
	}
}

func TestPreparedChatPreflightUsesInferenceLeaseAndRejectsClosedRunner(t *testing.T) {
	r, opts := preparedChatTestRunner(t)
	messages := []ChatMessage{UserMessage("b")}
	prepare := func(m []ChatMessage, o GenerationOptions) ([]ChatMessage, ContextWindowInfo, error) {
		if r.modelMu.TryLock() {
			r.modelMu.Unlock()
			t.Fatal("preflight did not protect the model from Close")
		}
		return r.PrepareChatContext(m, o)
	}
	plan, err := r.PrepareChatGeneration(messages, opts, prepare)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := plan.Wrap(nil)(messages, opts, nil); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("prepared generation used a closed Runner: %v", err)
	}
	if _, err := r.PrepareChatGeneration(messages, opts, prepare); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("preflight used a closed Runner: %v", err)
	}
}

func TestPreparedChatPreflightAdmissionIsCancelable(t *testing.T) {
	r, opts := preparedChatTestRunner(t)
	release, err := r.acquireInference(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started, done := make(chan struct{}), make(chan error, 1)
	go func() {
		close(started)
		_, err := r.PrepareChatGeneration([]ChatMessage{UserMessage("b")}, opts.WithContext(ctx), nil)
		done <- err
	}()
	<-started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("preflight did not honor canceled admission: %v", err)
	}
}
