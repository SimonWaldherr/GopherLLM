package gopherllm

import (
	"math"
	"strings"
	"testing"
)

func TestMeanPoolScalesBySampleCount(t *testing.T) {
	v := []float32{6, -3, 9}
	meanPoolInPlace(v, 3)
	want := []float32{2, -1, 3}
	for i := range want {
		if v[i] != want[i] {
			t.Fatalf("v[%d] = %v, want %v", i, v[i], want[i])
		}
	}
}

func TestL2NormalizeProducesUnitVector(t *testing.T) {
	v := []float32{3, 4}
	l2NormalizeInPlace(v)
	norm := math.Sqrt(float64(v[0]*v[0] + v[1]*v[1]))
	if math.Abs(norm-1) > 1e-6 {
		t.Fatalf("norm = %v, want 1", norm)
	}
}

func sequentialDecoderEmbeddingForTest(r *Runner, tokens []uint32) []float32 {
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	sum := make([]float32, r.config.Dim)
	for pos, token := range tokens {
		h := r.forwardHiddenToken(cache, buf, token, pos)
		addInPlace(sum, h)
	}
	meanPoolInPlace(sum, len(tokens))
	l2NormalizeInPlace(sum)
	return sum
}

func TestEmbedBatchedMatchesSequentialMultiChunk(t *testing.T) {
	oldChunk := prefillChunkOverrideValue()
	SetPrefillChunk(7)
	defer SetPrefillChunk(oldChunk)

	prompt := strings.Repeat("a b c ", 30)
	for _, arch := range []string{"llama", "stablelm"} {
		t.Run(arch, func(t *testing.T) {
			r, err := RunnerFromGGUFBytes(buildTinyStandardGGUF(arch))
			if err != nil {
				t.Fatal(err)
			}
			if !r.canBatchPrefill() {
				t.Fatal("expected batched embedding prefill")
			}
			tokens := r.tok.Encode(prompt)
			if len(tokens) <= 7 {
				t.Fatalf("need multiple chunks, got %d tokens", len(tokens))
			}
			want := sequentialDecoderEmbeddingForTest(r, tokens)
			got, err := r.Embed(prompt)
			if err != nil {
				t.Fatal(err)
			}
			if got.TokenCount != len(tokens) || len(got.Embedding) != len(want) {
				t.Fatalf("embedding shape got=%d/%d want=%d/%d", got.TokenCount, len(got.Embedding), len(tokens), len(want))
			}
			for i := range want {
				if d := math.Abs(float64(got.Embedding[i] - want[i])); d > 1e-3*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("value %d: batched=%v sequential=%v", i, got.Embedding[i], want[i])
				}
			}
		})
	}
}

func TestCompatibleDenseDecoderArchitecturesLoadAndGenerate(t *testing.T) {
	for _, arch := range []string{"phi3", "granite", "exaone", "internlm2", "stablelm"} {
		t.Run(arch, func(t *testing.T) {
			r, err := RunnerFromGGUFBytes(buildTinyStandardGGUF(arch))
			if err != nil {
				t.Fatal(err)
			}
			if arch == "stablelm" && (!r.config.UseLayerNorm || r.config.ParallelResidual) {
				t.Fatalf("stablelm config = %+v, want tensor-selected sequential LayerNorm residual", r.config)
			}
			opts := DefaultGenerationOptions()
			opts.MaxTokens = 1
			opts.Sampler.Temperature = 0
			if _, err := r.Generate("hello", opts); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCosineSimilarityRejectsInvalidInputs(t *testing.T) {
	if _, err := CosineSimilarity(nil, nil); err == nil {
		t.Fatal("empty vectors should fail")
	}
	if _, err := CosineSimilarity([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("dimension mismatch should fail")
	}
	if _, err := CosineSimilarity([]float32{0, 0}, []float32{1, 2}); err == nil {
		t.Fatal("zero-norm vector should fail")
	}
}

func TestRopeInterleavedArchitectureSelection(t *testing.T) {
	for _, arch := range []string{"llama", "llama2", "llama3", "mistral", "mistral3", "mixtral", "ministral", "internlm2"} {
		if !ropeInterleaved(arch) {
			t.Fatalf("ropeInterleaved(%q) = false, want true", arch)
		}
	}
	for _, arch := range []string{"qwen2", "phi3", "granite", "deepseek2"} {
		if ropeInterleaved(arch) {
			t.Fatalf("ropeInterleaved(%q) = true, want false", arch)
		}
	}
}

// TestInternLM2NeedsInterleavedRope guards a real bug: internlm2's GGUF
// conversion keeps the original interleaved (adjacent-pair) rotary layout
// rather than permuting weights for the split-half convention most
// HF-native architectures use. Defaulting to split-half left positional
// encoding subtly wrong -- early tokens looked fine, but generation drifted
// into wrong word choices and glued-together words as position grew
// (verified live: "42" for 6*7 and clean prose for the other two answers
// once interleaved rope was enabled, versus math gibberish and words like
// "theEiffy" beforehand).
func TestInternLM2NeedsInterleavedRope(t *testing.T) {
	if !ropeInterleaved("internlm2") {
		t.Fatal("internlm2 must use interleaved RoPE")
	}
}

func TestGenerationOptionsRejectsInvalidFloatsAndTopK(t *testing.T) {
	opts := DefaultGenerationOptions()
	opts.Sampler.Temperature = float32(math.Inf(1))
	if err := opts.Validate(); err == nil {
		t.Fatal("infinite temperature should fail")
	}

	opts = DefaultGenerationOptions()
	opts.Sampler.TopP = float32(math.Inf(1))
	if err := opts.Validate(); err == nil {
		t.Fatal("infinite top_p should fail")
	}

	opts = DefaultGenerationOptions()
	opts.Sampler.RepeatPenalty = float32(math.Inf(1))
	if err := opts.Validate(); err == nil {
		t.Fatal("infinite repeat_penalty should fail")
	}

	opts = DefaultGenerationOptions()
	opts.Sampler.TopK = -1
	if err := opts.Validate(); err == nil {
		t.Fatal("negative top_k should fail")
	}
}

func TestGreedyOutputFastPathAllowsRepeatPenalty(t *testing.T) {
	r := &Runner{}
	opts := DefaultGenerationOptions()
	opts.Sampler.Temperature = 0
	opts.Sampler.RepeatPenalty = 1.1
	if !r.canGreedyOutputFastPath(opts) {
		t.Fatal("deterministic decoding with repeat penalty should use the penalized argmax path")
	}
	opts.Sampler.Temperature = 0.7
	opts.Sampler.TopK = 40
	if r.canGreedyOutputFastPath(opts) {
		t.Fatal("sampling configuration should not use the argmax path")
	}
}

func TestPrefillChunkSizeEnvOverride(t *testing.T) {
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "")
	if got := prefillChunkSize(Config{Dim: 4096, HiddenDim: 14336}); got != 32 {
		t.Fatalf("large-model default chunk = %d, want 32", got)
	}
	if got := prefillChunkSize(Config{Dim: 3072, HiddenDim: 9216}); got != 128 {
		t.Fatalf("small-model default chunk = %d, want 128", got)
	}
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "64")
	if got := prefillChunkSize(Config{Dim: 3072, HiddenDim: 9216}); got != 64 {
		t.Fatalf("override chunk = %d, want 64", got)
	}
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "999")
	if got := prefillChunkSize(Config{}); got != 256 {
		t.Fatalf("clamped chunk = %d, want 256", got)
	}
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "nope")
	if got := prefillChunkSize(Config{}); got != 32 {
		t.Fatalf("invalid chunk = %d, want 32", got)
	}
}

func TestGenerationWorkspaceReusesAndGrowsBuffers(t *testing.T) {
	r := &Runner{config: Config{
		Dim: 8, HiddenDim: 16, NLayers: 2, NHeads: 2, NKVHeads: 1,
		HeadDim: 4, ValueDim: 4, KVDim: 4, MaxSeqLen: 128,
	}}
	cache1, buf1 := r.generationWorkspace(16)
	cache2, buf2 := r.generationWorkspace(8)
	if cache2 != cache1 || buf2 != buf1 {
		t.Fatal("workspace was not reused for a smaller request")
	}
	cache3, buf3 := r.generationWorkspace(32)
	if cache3 == cache1 {
		t.Fatal("KV cache did not grow for a larger request")
	}
	if cache3.MaxLen < 32 || cache3.MaxLen > r.config.MaxSeqLen || buf3 != buf1 {
		t.Fatalf("grown workspace: cache len=%d buffer reused=%v", cache3.MaxLen, buf3 == buf1)
	}
}

func TestCopyKVPrefixCopiesF32AndF16Rows(t *testing.T) {
	for _, f16 := range []bool{false, true} {
		t.Run(map[bool]string{false: "f32", true: "f16"}[f16], func(t *testing.T) {
			var src, dst *KVCache
			if f16 {
				src = NewKVCacheF16(2, 3, 2, 5)
				dst = NewKVCacheF16(2, 3, 2, 5)
				for layer := range src.K16 {
					for i := range src.K16[layer] {
						src.K16[layer][i] = uint16(i + 1 + layer*100)
					}
					for i := range src.V16[layer] {
						src.V16[layer][i] = uint16(i + 1 + layer*100)
					}
				}
			} else {
				src = NewKVCache(2, 3, 2, 5)
				dst = NewKVCache(2, 3, 2, 5)
				for layer := range src.K {
					for i := range src.K[layer] {
						src.K[layer][i] = float32(i + 1 + layer*100)
					}
					for i := range src.V[layer] {
						src.V[layer][i] = float32(i + 1 + layer*100)
					}
				}
			}
			if copied := copyKVPrefix(dst, src, 3); copied != 3 {
				t.Fatalf("copied positions = %d, want 3", copied)
			}
			for layer := 0; layer < 2; layer++ {
				if f16 {
					for i := 0; i < 3*src.PerPosKDim; i++ {
						if dst.K16[layer][i] != src.K16[layer][i] {
							t.Fatalf("layer %d K16[%d] = %d, want %d", layer, i, dst.K16[layer][i], src.K16[layer][i])
						}
					}
					if dst.K16[layer][3*src.PerPosKDim] != 0 {
						t.Fatalf("layer %d copied a K16 suffix", layer)
					}
					for i := 0; i < 3*src.PerPosVDim; i++ {
						if dst.V16[layer][i] != src.V16[layer][i] {
							t.Fatalf("layer %d V16[%d] = %d, want %d", layer, i, dst.V16[layer][i], src.V16[layer][i])
						}
					}
					continue
				}
				for i := 0; i < 3*src.PerPosKDim; i++ {
					if dst.K[layer][i] != src.K[layer][i] {
						t.Fatalf("layer %d K[%d] = %v, want %v", layer, i, dst.K[layer][i], src.K[layer][i])
					}
				}
				if dst.K[layer][3*src.PerPosKDim] != 0 {
					t.Fatalf("layer %d copied a K suffix", layer)
				}
				for i := 0; i < 3*src.PerPosVDim; i++ {
					if dst.V[layer][i] != src.V[layer][i] {
						t.Fatalf("layer %d V[%d] = %v, want %v", layer, i, dst.V[layer][i], src.V[layer][i])
					}
				}
			}
		})
	}
}

func TestGenerateChatReusesKVPrefixForFollowup(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.SystemPrompt = ""
	opts.MaxTokens = 1
	opts.Seed = 7
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1

	initial := []ChatMessage{UserMessage("a b c")}
	first, err := r.GenerateChat(initial, opts)
	if err != nil {
		t.Fatal(err)
	}
	if first.PromptCache == nil || first.PromptCache.Mode != "prefix" || first.PromptCache.Hit {
		t.Fatalf("first prompt cache = %+v, want a cold prefix cache", first.PromptCache)
	}

	followup := []ChatMessage{initial[0], AssistantMessage(first.Text), UserMessage("d e f")}
	cached, err := r.GenerateChat(followup, opts)
	if err != nil {
		t.Fatal(err)
	}
	if cached.PromptCache == nil || !cached.PromptCache.Hit || cached.PromptCache.ReusedTokens <= 0 {
		t.Fatalf("follow-up prompt cache = %+v, want reused prefix", cached.PromptCache)
	}
	if cached.PromptCache.ReusedTokens > cached.Stats.PromptTokens {
		t.Fatalf("reused %d of %d prompt tokens", cached.PromptCache.ReusedTokens, cached.Stats.PromptTokens)
	}

	cold, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	want, err := cold.GenerateChat(followup, opts)
	if err != nil {
		t.Fatal(err)
	}
	if cached.Text != want.Text {
		t.Fatalf("cached follow-up = %q, cold = %q", cached.Text, want.Text)
	}
}

func TestGenerateChatPrefixCacheKeepsSamplingDeterministic(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.SystemPrompt = ""
	opts.MaxTokens = 4
	opts.Seed = 37
	// Keep the normal repeat penalty and stochastic sampler enabled: sampling
	// mutates logits, so a prefix cache must never retain that altered vector.
	opts.Sampler.Temperature = 0.8
	opts.Sampler.TopK = 8
	messages := []ChatMessage{UserMessage("a b c")}
	if _, err := r.GenerateChat(messages, opts); err != nil {
		t.Fatal(err)
	}
	cached, err := r.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	if cached.PromptCache == nil || !cached.PromptCache.Hit || cached.PromptCache.ReusedTokens != cached.Stats.PromptTokens {
		t.Fatalf("same-prompt cache = %+v for %d tokens, want full prompt reuse", cached.PromptCache, cached.Stats.PromptTokens)
	}
	cold, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	want, err := cold.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	if cached.Text != want.Text {
		t.Fatalf("cached same-prompt = %q, cold = %q", cached.Text, want.Text)
	}
}

func TestEmbeddingInvalidatesKVPrefix(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.SystemPrompt = ""
	opts.MaxTokens = 1
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	if _, err := r.Generate("a b c", opts); err != nil {
		t.Fatal(err)
	}
	if r.prefixCache.cache == nil {
		t.Fatal("generation did not warm the prefix cache")
	}
	if _, err := r.Embed("a b c"); err != nil {
		t.Fatal(err)
	}
	if r.prefixCache.cache != nil {
		t.Fatal("embedding retained stale chat KV metadata")
	}
	result, err := r.Generate("a b c", opts)
	if err != nil {
		t.Fatal(err)
	}
	if result.PromptCache == nil || result.PromptCache.Hit {
		t.Fatalf("prompt cache after embedding = %+v, want cold", result.PromptCache)
	}
}

func TestGenerationCacheLenCapsBeforeMaxTokensCanOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if got := generationCacheLen(1024, 128, maxInt); got != 1024 {
		t.Fatalf("overflow-safe cache len = %d, want 1024", got)
	}
	if got := generationCacheLen(1024, 128, 5); got != 134 {
		t.Fatalf("ordinary cache len = %d, want 134", got)
	}
}

func TestRecentTokenWindowKeepsOnlyTrailingTokens(t *testing.T) {
	tokens := make([]uint32, repeatPenaltyWindow+8)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	recent := recentTokenWindowInto(nil, tokens)
	if len(recent) != repeatPenaltyWindow {
		t.Fatalf("window length = %d, want %d", len(recent), repeatPenaltyWindow)
	}
	if recent[0] != 8 || recent[len(recent)-1] != uint32(len(tokens)-1) {
		t.Fatalf("window = %v...%v, want 8...%d", recent[0], recent[len(recent)-1], len(tokens)-1)
	}
	tokens[len(tokens)-1] = 999
	if recent[len(recent)-1] == 999 {
		t.Fatal("recent window aliases the full prompt token slice")
	}
}

func TestValidUTF8PrefixLenKeepsIncompleteRuneBuffered(t *testing.T) {
	b := []byte{'H', 'i', ' ', 0xe2, 0x82}
	if got := validUTF8PrefixLen(b); got != 3 {
		t.Fatalf("prefix length = %d, want 3", got)
	}
}

func TestValidUTF8PrefixLenAcceptsCompletedRune(t *testing.T) {
	b := []byte{'H', 'i', ' ', 0xe2, 0x82, 0xac}
	if got := validUTF8PrefixLen(b); got != len(b) {
		t.Fatalf("prefix length = %d, want %d", got, len(b))
	}
}
