package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"testing"
)

// Run with GOPHERLLM_QWEN_CACHE_BENCH=1 and -benchtime=1x to measure one
// request boundary without loading model weights or running projections.
func BenchmarkQwen35PrefixCheckpoint27B(b *testing.B) {
	if os.Getenv("GOPHERLLM_QWEN_CACHE_BENCH") != "1" {
		b.Skip("set GOPHERLLM_QWEN_CACHE_BENCH=1 to allocate the 27B recurrent cache")
	}
	cfg := Config{SSMInner: 6144, SSMHeads: 48, SSMGroups: 16, SSMState: 128, SSMConv: 4}
	for _, snapshot := range []bool{true, false} {
		b.Run(fmt.Sprintf("snapshot=%t", snapshot), func(b *testing.B) {
			state := newQwen35Cache(cfg, 48)
			if got, want := state.bytes(), int64(156893184); got != want {
				b.Fatalf("27B recurrent backing = %d bytes, want %d", got, want)
			}
			var prefix prefixCacheState
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if snapshot {
					prefix.qwen35 = state.snapshot()
				} else {
					prefix.qwen35 = state
				}
			}
			b.StopTimer()
			backing := state.bytes()
			if prefix.qwen35 != state {
				backing += prefix.qwen35.bytes()
			}
			b.ReportMetric(float64(backing), "backing-B")
			runtime.KeepAlive(state)
			runtime.KeepAlive(prefix)
		})
	}
}

func qwen35RequireLivePrefix(t *testing.T, r *Runner) *Qwen35Cache {
	t.Helper()
	if r.workspaceCache == nil || r.prefixCache.cache != r.workspaceCache ||
		r.prefixCache.qwen35 == nil || r.prefixCache.qwen35 != r.workspaceCache.Qwen35 {
		t.Fatal("chat retained a separate recurrent prefix allocation")
	}
	return r.workspaceCache.Qwen35
}

func TestQwen35LivePrefixGrowthAndFollowup(t *testing.T) {
	for _, mtp := range []bool{false, true} {
		for _, f16 := range []bool{false, true} {
			t.Run(fmt.Sprintf("mtp=%t/f16=%t", mtp, f16), func(t *testing.T) {
				previousF16, previousI8 := useF16KVCache.Swap(f16), useI8KVCache.Swap(false)
				t.Cleanup(func() { useF16KVCache.Store(previousF16); useI8KVCache.Store(previousI8) })
				r := qwen35DenseRuntimeRunner(t, mtp)
				opts := tinyMistralKVPrefixOptions()
				opts.SystemPrompt, opts.MaxTokens = "", 2
				if mtp {
					opts.MTPDraftTokens = 2
				}
				messages := []ChatMessage{UserMessage("abcd")}
				first, err := r.GenerateChat(messages, opts)
				if err != nil {
					t.Fatal(err)
				}
				state, oldKV := qwen35RequireLivePrefix(t, r), r.workspaceCache
				messages = append(messages, AssistantMessage(first.Text), UserMessage("efghijklmnop"))
				got, err := r.GenerateChat(messages, opts)
				if err != nil || got.PromptCache == nil || !got.PromptCache.Hit {
					t.Fatalf("follow-up did not resume the live prefix: %+v, %v", got.PromptCache, err)
				}
				if r.workspaceCache == oldKV || qwen35RequireLivePrefix(t, r) != state {
					t.Fatal("attention-cache growth allocated or copied the recurrence")
				}
				cold := qwen35DenseRuntimeRunner(t, mtp)
				want, err := cold.GenerateChat(messages, opts)
				if err != nil {
					t.Fatal(err)
				}
				speculativeRequireSameResult(t, got, want)
				speculativeRequireSameVisibleState(t, r, cold)
			})
		}
	}
}

func TestQwen35LivePrefixGrowthStorageFormats(t *testing.T) {
	cfg, weights := qwen35BatchFixture(true)
	for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
		t.Run(fmt.Sprintf("kv=%d", format), func(t *testing.T) {
			previousF16, previousI8 := useF16KVCache.Swap(format == kvF16), useI8KVCache.Swap(format == kvI8)
			t.Cleanup(func() { useF16KVCache.Store(previousF16); useI8KVCache.Store(previousI8) })
			r := &Runner{kind: loadedQwen35, config: cfg, qwen35: weights}
			cache, buf := r.generationWorkspace(4)
			prefix := []uint32{1, 3, 4}
			var logits []float32
			for pos, token := range prefix {
				r.forwardTokenInto(cache, buf, token, pos, &logits)
			}
			state := cache.Qwen35
			before := state.snapshot()
			r.prefixCache = prefixCacheState{cache: cache, tokens: prefix, promptTokens: len(prefix), qwen35: state}
			grown, buf := r.generationChatWorkspace(5, false)
			if grown == cache || grown.kvFormat() != format || grown.Qwen35 != state ||
				!slices.Equal(grown.Qwen35.State, before.State) || !slices.Equal(grown.Qwen35.Conv, before.Conv) {
				t.Fatal("KV growth did not preserve the unique recurrent allocation and its boundary")
			}
			tokens := append(slices.Clone(prefix), 7)
			if r.prefixReuse(grown, tokens) != len(prefix) {
				t.Fatal("migrated cache lost its exact token prefix")
			}
			r.forwardTokenInto(grown, buf, 7, len(prefix), &logits)
			cold, coldBuf := qwen35BatchCache(cfg, weights, format), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
			var want []float32
			for pos, token := range tokens {
				ForwardQwen35Into(cfg, weights, cold, coldBuf, token, pos, &want)
			}
			if !slices.Equal(logits, want) || !slices.Equal(grown.Qwen35.State, cold.Qwen35.State) || !slices.Equal(grown.Qwen35.Conv, cold.Qwen35.Conv) {
				t.Fatal("migrated prefix changed continuation logits or recurrence")
			}
		})
	}
}

func TestQwen35LivePrefixMissClearAndScratchIsolation(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, false)
	opts := tinyMistralKVPrefixOptions()
	opts.SystemPrompt, opts.MaxTokens = "", 2
	a, b := []ChatMessage{UserMessage("abcdef")}, []ChatMessage{UserMessage("ponmlk")}
	for _, messages := range [][]ChatMessage{a, b, a} {
		got, err := r.GenerateChat(messages, opts)
		if err != nil || got.PromptCache == nil || got.PromptCache.Hit {
			t.Fatalf("divergent conversation reused a live recurrence: %+v, %v", got.PromptCache, err)
		}
		qwen35RequireLivePrefix(t, r)
		cold := qwen35DenseRuntimeRunner(t, false)
		want, err := cold.GenerateChat(messages, opts)
		if err != nil {
			t.Fatal(err)
		}
		speculativeRequireSameResult(t, got, want)
		speculativeRequireSameVisibleState(t, r, cold)
	}
	for _, clear := range []func(){r.ClearConversationCache, func() {
		cache, _ := r.generationWorkspace(r.workspaceCache.MaxLen)
		for _, value := range cache.Qwen35.State {
			if value != 0 {
				t.Fatal("non-chat workspace did not reset its recurrence")
			}
		}
	}, func() {
		previous, state := r.workspaceCache, r.workspaceCache.Qwen35
		cache, _ := r.generationWorkspace(previous.MaxLen + 1)
		if cache == previous || cache.Qwen35 != state {
			t.Fatal("cold workspace growth allocated a second recurrent state")
		}
	}} {
		clear()
		if r.prefixCache.cache != nil {
			t.Fatal("invalidated live recurrence still has prefix metadata")
		}
		got, err := r.GenerateChat(a, opts)
		if err != nil || got.PromptCache == nil || got.PromptCache.Hit {
			t.Fatalf("cleared chat did not prefill cold: %+v, %v", got.PromptCache, err)
		}
		qwen35RequireLivePrefix(t, r)
	}
}

func TestQwen35LiveGreedyPrefixWithoutLogitsStartsStochasticRequestCold(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, false)
	opts := tinyMistralKVPrefixOptions()
	opts.SystemPrompt, opts.MaxTokens = "", 1
	messages := []ChatMessage{UserMessage("abcdef")}
	if _, err := r.GenerateChat(messages, opts); err != nil {
		t.Fatal(err)
	}
	qwen35RequireLivePrefix(t, r)
	// A Metal argmax can retain only its next token, with no vocabulary
	// readback. Model that valid boundary independently of the GPU backend.
	r.prefixCache.promptLogits = nil
	r.prefixCache.promptHasGreedyToken = true
	opts.Sampler.Temperature, opts.Sampler.TopK = .7, 8
	got, err := r.GenerateChat(messages, opts)
	if err != nil || got.PromptCache == nil || got.PromptCache.Hit {
		t.Fatalf("Qwen attempted to rewind a no-logits greedy boundary: %+v, %v", got.PromptCache, err)
	}
	cold := qwen35DenseRuntimeRunner(t, false)
	want, err := cold.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	speculativeRequireSameResult(t, got, want)
	speculativeRequireSameVisibleState(t, r, cold)
}

func TestQwen35LiveGreedyPrefixFallbackResetsMTPBoundary(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, true)
	opts := tinyMistralKVPrefixOptions()
	opts.SystemPrompt, opts.MaxTokens, opts.MTPDraftTokens = "", 1, 1
	messages := []ChatMessage{UserMessage("abcdef")}
	if _, err := r.GenerateChat(messages, opts); err != nil {
		t.Fatal(err)
	}
	r.prefixCache.promptLogits = nil
	r.prefixCache.promptHasGreedyToken = true
	t.Setenv("GOPHERLLM_NO_GREEDY_ARGMAX", "1")
	got, err := r.GenerateChat(messages, opts)
	if err != nil || got.PromptCache == nil || got.PromptCache.Hit {
		t.Fatalf("Qwen MTP did not cold-prefill missing logits: %+v, %v", got.PromptCache, err)
	}
	cold := qwen35DenseRuntimeRunner(t, true)
	want, err := cold.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	speculativeRequireSameResult(t, got, want)
	speculativeRequireSameVisibleState(t, r, cold)
	speculativeRequireSameCacheRows(t, r.workspaceCache.Qwen35MTP.KV, cold.workspaceCache.Qwen35MTP.KV, len(r.prefixCache.tokens))
}

type qwen35CancelAfterPrefixInvalidation struct {
	context.Context
	r      *Runner
	cancel context.CancelFunc
}

func (c *qwen35CancelAfterPrefixInvalidation) Err() error {
	if c.r.prefixCache.cache == nil {
		for _, value := range c.r.workspaceCache.Qwen35.State {
			if value != 0 {
				c.cancel()
				break
			}
		}
	}
	return c.Context.Err()
}

func TestQwen35LivePrefixCancellation(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, false)
	opts := tinyMistralKVPrefixOptions()
	opts.SystemPrompt, opts.MaxTokens = "", 2
	a, b := []ChatMessage{UserMessage("abcdef")}, []ChatMessage{UserMessage("ponmlk")}
	if _, err := r.GenerateChat(a, opts); err != nil {
		t.Fatal(err)
	}
	boundary := qwen35RequireLivePrefix(t, r)
	before := slices.Clone(boundary.State)
	preCanceled, preCancel := context.WithCancel(context.Background())
	preCancel()
	if _, err := r.GenerateChat(b, opts.WithContext(preCanceled)); !errors.Is(err, context.Canceled) ||
		qwen35RequireLivePrefix(t, r) != boundary || !slices.Equal(boundary.State, before) {
		t.Fatalf("pre-canceled request changed the resident boundary: %v", err)
	}
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &qwen35CancelAfterPrefixInvalidation{Context: base, r: r, cancel: cancel}
	if _, err := r.GenerateChat(b, opts.WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Fatalf("partial prefill did not cancel: %v", err)
	}
	if r.prefixCache.cache != nil {
		t.Fatal("partial cold prefill retained the previous live prefix")
	}
	got, err := r.GenerateChat(a, opts)
	if err != nil || got.PromptCache == nil || got.PromptCache.Hit {
		t.Fatalf("request after cancellation reused invalid recurrence: %+v, %v", got.PromptCache, err)
	}
	cold := qwen35DenseRuntimeRunner(t, false)
	want, err := cold.GenerateChat(a, opts)
	if err != nil {
		t.Fatal(err)
	}
	speculativeRequireSameResult(t, got, want)
	speculativeRequireSameVisibleState(t, r, cold)
}

func TestQwen35NoneligibleRequestPreservesRetainedPrefixTokens(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, false)
	opts := tinyMistralKVPrefixOptions()
	opts.SystemPrompt, opts.MaxTokens = "", 1
	a, b := []ChatMessage{UserMessage("abcdef")}, []ChatMessage{UserMessage("ponmlk")}
	if _, err := r.GenerateChat(a, opts); err != nil {
		t.Fatal(err)
	}
	retained := r.workspaceCache
	tokens := slices.Clone(r.prefixCache.tokens)
	state := retained.Qwen35.snapshot()
	otherTokens := r.renderMessages(b, opts.SystemPrompt, opts.ActiveTools())
	if len(otherTokens) > cap(r.prefixCache.tokens) || slices.Equal(otherTokens, tokens) {
		t.Fatal("fixture must expose a different prompt that fits the retained token backing")
	}
	// Exercise noneligible generation without physically reserving a cache
	// above 512 MiB. Only its logical capacity exceeds the budget; both short
	// prompts access positions backed by this fixture's real small row slabs.
	temporary := qwen35BatchCache(r.config, r.qwen35, retained.kvFormat())
	rowBytes := kvCacheBytes(temporary.layerCount(), temporary.PerPosKDim, temporary.PerPosVDim, 1, temporary.kvFormat())
	temporary.MaxLen = int(maxReusableKVCacheBytes/rowBytes) + 1
	r.workspaceCache = temporary
	got, err := r.GenerateChat(b, opts)
	if err != nil || got.PromptCache == nil || got.PromptCache.Mode != "disabled" {
		t.Fatalf("fixture did not run a noneligible request: %+v, %v", got.PromptCache, err)
	}
	if r.prefixCache.cache != retained || !slices.Equal(r.prefixCache.tokens, tokens) ||
		!slices.Equal(retained.Qwen35.State, state.State) || !slices.Equal(retained.Qwen35.Conv, state.Conv) {
		t.Fatal("temporary request changed a different retained prefix's token IDs or recurrence")
	}
	r.workspaceCache = retained
	if warm, err := r.GenerateChat(a, opts); err != nil || warm.PromptCache == nil || !warm.PromptCache.Hit {
		t.Fatalf("retained prefix was no longer reusable after temporary generation: %+v, %v", warm.PromptCache, err)
	}
	got, err = r.GenerateChat(b, opts)
	if err != nil || got.PromptCache == nil || got.PromptCache.Hit {
		t.Fatalf("temporary token IDs caused a false retained prefix hit: %+v, %v", got.PromptCache, err)
	}
	cold := qwen35DenseRuntimeRunner(t, false)
	want, err := cold.GenerateChat(b, opts)
	if err != nil {
		t.Fatal(err)
	}
	speculativeRequireSameResult(t, got, want)
	speculativeRequireSameVisibleState(t, r, cold)
}
