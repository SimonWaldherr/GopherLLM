package gopherllm

import (
	"strings"
	"sync"
	"testing"
)

// newTinyMistralKVPrefixRunner uses the tiny standard transformer weights but
// a compact Mistral control-token vocabulary. Every token ID stays inside the
// 16-row synthetic embedding table, so runtime tests exercise the real
// prefill/KV path rather than a mock cache.
func newTinyMistralKVPrefixRunner(t testing.TB) *Runner {
	t.Helper()
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	tokens := []string{
		"<unk>", "<s>", "</s>",
		"[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]",
		"[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]",
		"▁", "a", "b", "c", "d", "e", "f",
	}
	toID := make(map[string]uint32, len(tokens))
	for i, token := range tokens {
		toID[token] = uint32(i)
	}
	r.arch = "mistral3"
	r.tok = &Tokenizer{
		Vocab: tokens, Scores: make([]float32, len(tokens)), TokenToID: toID,
		Mode: TokenizerSentencePiece, AddBOS: true, BOSID: 1, EOSID: 2,
	}
	return r
}

func tinyMistralKVPrefixOptions() GenerationOptions {
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 1
	opts.Seed = 41
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	return opts
}

func TestMistralKVPrefixCacheReusesStaticPrefixAcrossBranches(t *testing.T) {
	r := newTinyMistralKVPrefixRunner(t)
	defer r.Close()
	r.EnableMistralKVPrefixCache(0)
	opts := tinyMistralKVPrefixOptions()

	first := []ChatMessage{{Role: ChatRoleSystem, Content: "a"}, UserMessage("b")}
	if _, err := r.GenerateChat(first, opts); err != nil {
		t.Fatal(err)
	}
	stats := r.MistralKVPrefixCacheStats()
	if !stats.Enabled || stats.Entries != 1 || stats.Bytes <= 0 {
		t.Fatalf("first snapshot cache = %+v", stats)
	}

	// Replace the normal one-workspace prefix with a different system branch.
	// The next request cannot get the original static rows from prefixCache;
	// its hit must come from the opt-in immutable snapshot above.
	other := []ChatMessage{{Role: ChatRoleSystem, Content: "d"}, UserMessage("b")}
	if _, err := r.GenerateChat(other, opts); err != nil {
		t.Fatal(err)
	}

	branched := []ChatMessage{{Role: ChatRoleSystem, Content: "a"}, UserMessage("c")}
	got, err := r.GenerateChat(branched, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.PromptCache == nil || !got.PromptCache.Hit {
		t.Fatalf("branched prompt cache = %+v, want snapshot hit", got.PromptCache)
	}
	static := r.mistralStaticPromptPrefix(branched, opts.SystemPrompt, opts.ActiveTools(), r.renderMessages(branched, opts.SystemPrompt, opts.ActiveTools()))
	if len(static) <= 1 || got.PromptCache.ReusedTokens < len(static) {
		t.Fatalf("reused %d tokens, want at least static prefix %d", got.PromptCache.ReusedTokens, len(static))
	}
	stats = r.MistralKVPrefixCacheStats()
	if stats.Hits == 0 || stats.Entries < 2 {
		t.Fatalf("snapshot cache after branch = %+v, want hit and both branches", stats)
	}

	// Reuse changes only work, never the model result.
	cold := newTinyMistralKVPrefixRunner(t)
	defer cold.Close()
	want, err := cold.GenerateChat(branched, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != want.Text || got.FinishReason != want.FinishReason {
		t.Fatalf("cached generation = (%q, %q), cold = (%q, %q)", got.Text, got.FinishReason, want.Text, want.FinishReason)
	}
}

func TestMistralKVPrefixCacheSnapshotIsImmutableAndBounded(t *testing.T) {
	r := &Runner{}
	r.EnableMistralKVPrefixCache(0)
	epoch, ok := r.mistralKVPrefixCacheEpoch()
	if !ok || epoch == 0 {
		t.Fatalf("enabled cache epoch = %d, ok=%v", epoch, ok)
	}

	src := NewKVCache(2, 3, 2, 4)
	for layer := range src.K {
		for i := range src.K[layer] {
			src.K[layer][i] = float32(100*layer + i)
		}
		for i := range src.V[layer] {
			src.V[layer][i] = float32(1000 + 100*layer + i)
		}
	}
	prefix := []uint32{1, 5, 10}
	r.putMistralKVPrefixSnapshot(src, prefix, epoch)
	stats := r.MistralKVPrefixCacheStats()
	if stats.Entries != 1 || stats.Bytes <= 0 || stats.Bytes > stats.MaxBytes {
		t.Fatalf("stored snapshot stats = %+v", stats)
	}

	// Mutating the live workspace after insertion must not affect retained rows.
	for layer := range src.K {
		for i := range src.K[layer] {
			src.K[layer][i] = -1
		}
		for i := range src.V[layer] {
			src.V[layer][i] = -2
		}
	}
	dst := NewKVCache(2, 3, 2, 4)
	if copied, hit := r.mistralKVPrefixCacheReuse(dst, prefix); !hit || copied != len(prefix) {
		t.Fatalf("snapshot reuse = (%d, %v), want (%d, true)", copied, hit, len(prefix))
	}
	for layer := range dst.K {
		for i := 0; i < len(prefix)*dst.PerPosKDim; i++ {
			if got, want := dst.K[layer][i], float32(100*layer+i); got != want {
				t.Fatalf("K layer %d index %d = %v, want %v", layer, i, got, want)
			}
		}
		for i := 0; i < len(prefix)*dst.PerPosVDim; i++ {
			if got, want := dst.V[layer][i], float32(1000+100*layer+i); got != want {
				t.Fatalf("V layer %d index %d = %v, want %v", layer, i, got, want)
			}
		}
	}

	// A cap smaller than one snapshot preserves the strict bound and leaves
	// generation correct by behaving as a normal cold prefill.
	r.EnableMistralKVPrefixCache(1)
	epoch, ok = r.mistralKVPrefixCacheEpoch()
	if !ok {
		t.Fatal("small cache unexpectedly disabled")
	}
	r.putMistralKVPrefixSnapshot(dst, prefix, epoch)
	stats = r.MistralKVPrefixCacheStats()
	if stats.Entries != 0 || stats.Bytes != 0 || stats.MaxBytes != 1 {
		t.Fatalf("oversized snapshot bypassed bound: %+v", stats)
	}

	// Clear is an explicit invalidation boundary even for a pre-existing
	// generation which retained an old epoch before it started cloning rows.
	r.EnableMistralKVPrefixCache(0)
	epoch, _ = r.mistralKVPrefixCacheEpoch()
	r.ClearMistralKVPrefixCache()
	r.putMistralKVPrefixSnapshot(dst, prefix, epoch)
	stats = r.MistralKVPrefixCacheStats()
	if stats.Entries != 0 || stats.Bytes != 0 {
		t.Fatalf("stale epoch repopulated cleared cache: %+v", stats)
	}
}

// Cache-control methods are intentionally usable while a generation owns the
// Runner's serialized workspace. This exercises the epoch fence around an
// in-flight snapshot clone as well as the cache mutex under the race detector.
func TestMistralKVPrefixCacheConcurrentControlAndGeneration(t *testing.T) {
	r := newTinyMistralKVPrefixRunner(t)
	defer r.Close()
	r.EnableMistralKVPrefixCache(0)
	opts := tinyMistralKVPrefixOptions()
	messages := []ChatMessage{{Role: ChatRoleSystem, Content: "a"}, UserMessage("b")}

	var controls sync.WaitGroup
	controls.Add(1)
	go func() {
		defer controls.Done()
		for range 32 {
			r.ClearMistralKVPrefixCache()
			r.EnableMistralKVPrefixCache(DefaultMistralKVPrefixCacheBytes)
			_ = r.MistralKVPrefixCacheStats()
		}
	}()
	for range 16 {
		if _, err := r.GenerateChat(messages, opts); err != nil {
			t.Fatalf("GenerateChat while controlling KV cache: %v", err)
		}
	}
	controls.Wait()
}

func TestMistralStaticPromptPrefixIsConservative(t *testing.T) {
	// This legacy test tokenizer has no native system markers, so Mistral folds
	// system content into the final user turn. A static KV snapshot would be
	// wrong; the helper must decline it even though a BOS token is shared.
	r := &Runner{tok: newMistralToolTokenizer(), arch: "ministral"}
	full, _, ok, err := r.renderMistralInstMessages([]ChatMessage{UserMessage("question")}, "system", nil)
	if err != nil || !ok {
		t.Fatalf("render = ok=%v err=%v", ok, err)
	}
	if prefix := r.mistralStaticPromptPrefix([]ChatMessage{UserMessage("question")}, "system", nil, full); prefix != nil {
		t.Fatalf("legacy folded prefix = %v, want nil", prefix)
	}
}

func TestMistralKVPrefixTokenKeyIsExact(t *testing.T) {
	left := []uint32{1, 23, 456}
	right := append([]uint32(nil), left...)
	if mistralKVPrefixTokenKey(left) != mistralKVPrefixTokenKey(right) {
		t.Fatal("identical token IDs produced different cache keys")
	}
	right[2]++
	if mistralKVPrefixTokenKey(left) == mistralKVPrefixTokenKey(right) {
		t.Fatal("distinct token IDs produced the same cache key")
	}
	if got := mistralKVPrefixTokenKey(nil); got != "" {
		t.Fatalf("empty token key = %q, want empty", got)
	}
}

func BenchmarkMistralKVPrefixCache(b *testing.B) {
	// Alternate branches so the normal one-workspace cache cannot retain the
	// next prompt's static rows. The cached benchmark therefore measures actual
	// K/V-prefix reuse rather than repeated full-prompt logits reuse.
	makeMessages := func(system, user string) []ChatMessage {
		return []ChatMessage{{Role: ChatRoleSystem, Content: system}, UserMessage(user)}
	}
	systemA := strings.TrimSpace(strings.Repeat("a ", 96))
	systemB := strings.TrimSpace(strings.Repeat("b ", 96))
	prompts := [][]ChatMessage{makeMessages(systemA, "c"), makeMessages(systemB, "d")}
	opts := tinyMistralKVPrefixOptions()

	b.Run("disabled", func(b *testing.B) {
		r := newTinyMistralKVPrefixRunner(b)
		defer r.Close()
		if _, err := r.GenerateChat(prompts[0], opts); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := r.GenerateChat(prompts[i&1], opts); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("enabled", func(b *testing.B) {
		r := newTinyMistralKVPrefixRunner(b)
		defer r.Close()
		r.EnableMistralKVPrefixCache(0)
		for _, prompt := range prompts {
			if _, err := r.GenerateChat(prompt, opts); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := r.GenerateChat(prompts[i&1], opts); err != nil {
				b.Fatal(err)
			}
		}
	})
}
