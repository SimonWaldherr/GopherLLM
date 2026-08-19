package gopherllm

import (
	"math/rand"
	"testing"
)

func BenchmarkOnlineAttention_ctx512(b *testing.B) {
	const headDim, ctx = 128, 512
	q := benchFloatSlice(headDim)
	keys := benchFloatSlice(ctx * headDim)
	values := benchFloatSlice(ctx * headDim)
	out := make([]float32, headDim)
	scale := float32(0.08838)
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * headDim * 4))
	for b.Loop() {
		clear(out)
		onlineAttention(q, keys, values, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
	}
}

func BenchmarkOnlineAttentionF16_ctx512(b *testing.B) {
	const headDim, ctx = 128, 512
	rng := rand.New(rand.NewSource(9))
	q := benchFloatSlice(headDim)
	keys := make([]uint16, ctx*headDim)
	values := make([]uint16, ctx*headDim)
	for i := range keys {
		keys[i] = F32ToF16(rng.Float32() - 0.5)
		values[i] = F32ToF16(rng.Float32() - 0.5)
	}
	out := make([]float32, headDim)
	scale := float32(0.08838)
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * headDim * 2))
	for b.Loop() {
		clear(out)
		onlineAttentionF16(q, keys, values, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
	}
}

// The ctx4096 pair exceeds L2, exposing the memory-bandwidth halving that is
// the point of the f16 cache (ctx512's 512 KiB working set is cache-resident).
func BenchmarkOnlineAttention_ctx4096(b *testing.B) {
	const headDim, ctx = 128, 4096
	q := benchFloatSlice(headDim)
	keys := benchFloatSlice(ctx * headDim)
	values := benchFloatSlice(ctx * headDim)
	out := make([]float32, headDim)
	scale := float32(0.08838)
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * headDim * 4))
	for b.Loop() {
		clear(out)
		onlineAttention(q, keys, values, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
	}
}

func BenchmarkOnlineAttentionF16_ctx4096(b *testing.B) {
	const headDim, ctx = 128, 4096
	rng := rand.New(rand.NewSource(9))
	q := benchFloatSlice(headDim)
	keys := make([]uint16, ctx*headDim)
	values := make([]uint16, ctx*headDim)
	for i := range keys {
		keys[i] = F32ToF16(rng.Float32() - 0.5)
		values[i] = F32ToF16(rng.Float32() - 0.5)
	}
	out := make([]float32, headDim)
	scale := float32(0.08838)
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * headDim * 2))
	for b.Loop() {
		clear(out)
		onlineAttentionF16(q, keys, values, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
	}
}

func benchmarkGQAAttention(b *testing.B, ctx int, grouped bool) {
	const queryHeads, headDim = 4, 128
	queries := benchFloatSlice(queryHeads * headDim)
	keys := benchFloatSlice(ctx * headDim)
	values := benchFloatSlice(ctx * headDim)
	out := make([]float32, len(queries))
	scale := float32(0.08838)
	b.ReportAllocs()
	// Report unique K/V bytes. The separate implementation logically reads the
	// same cache four times; the grouped implementation is designed to keep each
	// row hot while evaluating all query heads that share it.
	b.SetBytes(int64(2 * ctx * headDim * 4))
	for b.Loop() {
		clear(out)
		if grouped {
			onlineAttentionGroup(queries, keys, values, queryHeads,
				headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
			continue
		}
		for h := 0; h < queryHeads; h++ {
			off := h * headDim
			onlineAttention(queries[off:off+headDim], keys, values,
				headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0,
				out[off:off+headDim])
		}
	}
}

func BenchmarkGQAAttention_ctx4096(b *testing.B) {
	b.Run("separate", func(b *testing.B) { benchmarkGQAAttention(b, 4096, false) })
	b.Run("grouped", func(b *testing.B) { benchmarkGQAAttention(b, 4096, true) })
}

func benchmarkGQAAttentionF16(b *testing.B, ctx int, mode string) {
	const queryHeads, headDim = 4, 128
	queries := benchFloatSlice(queryHeads * headDim)
	keys := make([]uint16, ctx*headDim)
	values := make([]uint16, ctx*headDim)
	for i := range keys {
		keys[i] = F32ToF16(float32(i%17-8) / 16)
		values[i] = F32ToF16(float32(i%23-11) / 16)
	}
	out := make([]float32, len(queries))
	scale := float32(0.08838)
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * headDim * 2))
	for b.Loop() {
		clear(out)
		switch mode {
		case "grouped":
			onlineAttentionGroupF16(queries, keys, values, queryHeads,
				headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
		case "generic":
			// The former grouped implementation: keep it as a directly
			// comparable baseline for the shared-row NEON f16 primitives.
			onlineAttentionGroupEither(queries, nil, keys, nil, nil, values, nil, queryHeads,
				headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
		default:
			for h := 0; h < queryHeads; h++ {
				off := h * headDim
				onlineAttentionF16(queries[off:off+headDim], keys, values,
					headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0,
					out[off:off+headDim])
			}
		}
	}
}

func BenchmarkGQAAttentionF16_ctx4096(b *testing.B) {
	b.Run("separate", func(b *testing.B) { benchmarkGQAAttentionF16(b, 4096, "separate") })
	b.Run("generic", func(b *testing.B) { benchmarkGQAAttentionF16(b, 4096, "generic") })
	b.Run("grouped", func(b *testing.B) { benchmarkGQAAttentionF16(b, 4096, "grouped") })
}

func BenchmarkGQAAttentionF16_ctx32768(b *testing.B) {
	b.Run("generic", func(b *testing.B) { benchmarkGQAAttentionF16(b, 32768, "generic") })
	b.Run("grouped", func(b *testing.B) { benchmarkGQAAttentionF16(b, 32768, "grouped") })
}

// benchmarkQwen35GQAAttentionF16 covers Qwen3.8-27B's actual full-attention
// layout: 24 query heads, four KV heads, six queries per shared KV stream, and
// 256-wide heads. The old hybrid path ran all 24 heads separately; the grouped
// path is the long-context dispatch used by qwen35AttentionForward.
func benchmarkQwen35GQAAttentionF16(b *testing.B, ctx int, grouped bool) {
	const nHeads, nKVHeads, headDim = 24, 4, 256
	config := Config{NHeads: nHeads, NKVHeads: nKVHeads, HeadDim: headDim, ValueDim: headDim}
	cache := NewKVCacheF16(1, nKVHeads*headDim, nKVHeads*headDim, ctx)
	for i := range cache.K16[0] {
		cache.K16[0][i] = F32ToF16(float32(i%29-14) / 16)
		cache.V16[0][i] = F32ToF16(float32(i%31-15) / 16)
	}
	buf := &DecodeBuffer{
		Q:       benchFloatSlice(nHeads * headDim),
		AttnOut: make([]float32, nHeads*headDim),
	}
	scale := float32(1 / 16) // 1/sqrt(256), Qwen3.8's default attention scale.
	b.ReportAllocs()
	// Unique K/V bytes. The scalar baseline must logically reread this cache
	// six times per KV head; grouping keeps it hot while all six queries run.
	b.SetBytes(int64(2 * ctx * nKVHeads * headDim * 2))
	for b.Loop() {
		clear(buf.AttnOut)
		if grouped {
			qwen35ParallelAttendHeadGroups(config, cache, buf, 0, ctx-1, scale, nHeads/nKVHeads)
			continue
		}
		for h := 0; h < nHeads; h++ {
			qOff := h * headDim
			cache.attendHead(0, h/(nHeads/nKVHeads), buf.Q[qOff:qOff+headDim], headDim, headDim,
				0, ctx-1, scale, 0, buf.AttnOut[qOff:qOff+headDim])
		}
	}
}

func BenchmarkQwen35GQAAttentionF16_ctx4096(b *testing.B) {
	b.Run("separate", func(b *testing.B) { benchmarkQwen35GQAAttentionF16(b, 4096, false) })
	b.Run("grouped", func(b *testing.B) { benchmarkQwen35GQAAttentionF16(b, 4096, true) })
}

func benchmarkQwen35GQAAttention(b *testing.B, ctx int, grouped bool) {
	const nHeads, nKVHeads, headDim = 24, 4, 256
	config := Config{NHeads: nHeads, NKVHeads: nKVHeads, HeadDim: headDim, ValueDim: headDim}
	cache := NewKVCache(1, nKVHeads*headDim, nKVHeads*headDim, ctx)
	cache.K[0] = benchFloatSlice(len(cache.K[0]))
	cache.V[0] = benchFloatSlice(len(cache.V[0]))
	buf := &DecodeBuffer{
		Q:       benchFloatSlice(nHeads * headDim),
		AttnOut: make([]float32, nHeads*headDim),
	}
	scale := float32(1 / 16) // 1/sqrt(256), Qwen3.8's default attention scale.
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * nKVHeads * headDim * 4))
	for b.Loop() {
		clear(buf.AttnOut)
		if grouped {
			qwen35ParallelAttendHeadGroups(config, cache, buf, 0, ctx-1, scale, nHeads/nKVHeads)
			continue
		}
		for h := 0; h < nHeads; h++ {
			qOff := h * headDim
			cache.attendHead(0, h/(nHeads/nKVHeads), buf.Q[qOff:qOff+headDim], headDim, headDim,
				0, ctx-1, scale, 0, buf.AttnOut[qOff:qOff+headDim])
		}
	}
}

func BenchmarkQwen35GQAAttention_ctx4096(b *testing.B) {
	b.Run("separate", func(b *testing.B) { benchmarkQwen35GQAAttention(b, 4096, false) })
	b.Run("grouped", func(b *testing.B) { benchmarkQwen35GQAAttention(b, 4096, true) })
}

func BenchmarkGQAAttention_ctx32768(b *testing.B) {
	b.Run("separate", func(b *testing.B) { benchmarkGQAAttention(b, 32768, false) })
	b.Run("grouped", func(b *testing.B) { benchmarkGQAAttention(b, 32768, true) })
}

func BenchmarkParallelMinistralAttention_ctx1667(b *testing.B) {
	const nHeads, nKVHeads, headDim, ctx = 32, 8, 128, 1667
	config := Config{NHeads: nHeads, NKVHeads: nKVHeads, HeadDim: headDim, ValueDim: headDim}
	cache := NewKVCache(1, nKVHeads*headDim, nKVHeads*headDim, ctx)
	cache.K[0] = benchFloatSlice(len(cache.K[0]))
	cache.V[0] = benchFloatSlice(len(cache.V[0]))
	buf := &DecodeBuffer{Q: benchFloatSlice(nHeads * headDim), AttnOut: make([]float32, nHeads*headDim)}
	scale := float32(0.0883883476)
	layer := LayerWeights{}
	b.Run("separate", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			clear(buf.AttnOut)
			parallelAttendHeads(config, layer, cache, buf, 0, ctx-1, 0, scale, nHeads/nKVHeads)
		}
	})
	b.Run("grouped", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			clear(buf.AttnOut)
			parallelAttendHeadGroups(config, cache, buf, 0, ctx-1, 0, scale, nHeads/nKVHeads)
		}
	})
}

// The ctx32768 pair is larger than Apple Silicon's shared performance-core
// L2, so it measures the compact cache with a DRAM-sized working set instead
// of measuring only conversion throughput on cache-resident data.
func BenchmarkOnlineAttention_ctx32768(b *testing.B) {
	const headDim, ctx = 128, 32768
	q := benchFloatSlice(headDim)
	keys := benchFloatSlice(ctx * headDim)
	values := benchFloatSlice(ctx * headDim)
	out := make([]float32, headDim)
	scale := float32(0.08838)
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * headDim * 4))
	for b.Loop() {
		clear(out)
		onlineAttention(q, keys, values, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
	}
}

func BenchmarkOnlineAttentionF16_ctx32768(b *testing.B) {
	const headDim, ctx = 128, 32768
	rng := rand.New(rand.NewSource(9))
	q := benchFloatSlice(headDim)
	keys := make([]uint16, ctx*headDim)
	values := make([]uint16, ctx*headDim)
	for i := range keys {
		keys[i] = F32ToF16(rng.Float32() - 0.5)
		values[i] = F32ToF16(rng.Float32() - 0.5)
	}
	out := make([]float32, headDim)
	scale := float32(0.08838)
	b.ReportAllocs()
	b.SetBytes(int64(2 * ctx * headDim * 2))
	for b.Loop() {
		clear(out)
		onlineAttentionF16(q, keys, values, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, out)
	}
}

func BenchmarkRoPEApply_128x32(b *testing.B) {
	const headDim, nHeads = 128, 32
	cfg := Config{RopeTheta: 10000, RopeDimensionCount: headDim, MaxSeqLen: 4096}
	inv, mscale := buildRopeInvFreq(cfg, headDim)
	vec := benchFloatSlice(nHeads * headDim)
	var sin, cos []float32
	b.ReportAllocs()
	for b.Loop() {
		half, n := prepareRopeScratch(37, headDim, headDim, inv, mscale, &sin, &cos)
		applyPreparedRope(vec, headDim, nHeads, half, n, sin, cos, true)
	}
}

func BenchmarkDotQ8_0_4096(b *testing.B) {
	row := benchBytes((4096 / 32) * 34)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = DotQ8_0F32(row, x, 4096)
	}
}

func BenchmarkDotQ4_0_4096(b *testing.B) {
	row := benchBytes((4096 / 32) * 18)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = DotQ4_0F32(row, x, 4096)
	}
}

func BenchmarkDotMXFP4_4096(b *testing.B) {
	row := benchBytes((4096 / 32) * 17)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = DotMXFP4F32(row, x, 4096)
	}
}

const benchParagraph = "The quick brown fox jumps over 12 lazy dogs. iPhone models cost $999, e.g. version 3.14!\n\nMixedCASE words, numbers 2024, and punctuation... all get pre-tokenized differently."

func BenchmarkPretokenizeTekken(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = pretokenizeTekken(benchParagraph)
	}
}

func BenchmarkPretokenizeGPT2(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = pretokenizeGPT2(benchParagraph)
	}
}

func BenchmarkEncodeSentencePiece(b *testing.B) {
	tok := newInstTestTokenizer()
	text := "hello world this is a benchmark of the tokenizer merge loop"
	b.ReportAllocs()
	for b.Loop() {
		_ = tok.EncodeWithoutBOS(text)
	}
}

func BenchmarkMatvecBatchQ4K_3072x3072_P32(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	const rows, cols, P = 3072, 3072, 32
	data := make([]byte, 0, rows*(cols/256)*144)
	for r := 0; r < rows; r++ {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	w := Weight{Raw: data, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols}
	xs := make([][]float32, P)
	outs := make([][]float32, P)
	for p := range xs {
		xs[p] = randomVec(rng, cols)
		outs[p] = make([]float32, rows)
	}
	b.ReportAllocs()
	for b.Loop() {
		matvecBatch(w, xs, outs)
	}
}

func BenchmarkTinyModelForward(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		b.Fatal(err)
	}
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 16)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	b.ReportAllocs()
	for b.Loop() {
		r.forwardTokenInto(cache, buf, 3, 0, &logits)
	}
}

func BenchmarkTinyModelBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := r.tok.Encode("a b c a b c a b")
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyModelCachedPrompt(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		b.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.SystemPrompt = ""
	opts.MaxTokens = 1
	opts.Seed = 7
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	if _, err := r.Generate("a b c a b c", opts); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := r.Generate("a b c a b c", opts); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTinyStableLMBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyStandardGGUF("stablelm"))
	if err != nil {
		b.Fatal(err)
	}
	tokens := r.tok.Encode("a b c a b c a b")
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyQwen3BatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyQwen3GGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := r.tok.Encode("a b c a b c a b")
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyExaone4BatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyExaone4GGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := r.tok.Encode("a b c a b c a b")
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyOlmo2BatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyOlmo2GGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := r.tok.Encode("abcdefghijklm")
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyPhi2BatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyPhi2GGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := r.tok.Encode("abcdefghijklm")
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

// The 14 architectures below (GPT-2 through GraniteMoE) were added in
// acc7c42 with correctness coverage in model_families_extra_test.go but no
// performance benchmarks, so a regression on any of them could ship
// unnoticed. Each reuses that file's tiny-GGUF builder verbatim and follows
// the same "abcdefgh" + fallback-token-ids pattern as runExtraArchSmoke,
// since these builders' gpt2-BPE tokenizer.ggml.model with a plain a-z vocab
// (no merges table) can encode to zero tokens.

func extraBenchTokens(r *Runner) []uint32 {
	tokens := r.tok.Encode("abcdefgh")
	if len(tokens) == 0 {
		tokens = []uint32{2, 3, 4, 5}
	}
	return tokens
}

func BenchmarkTinyGPT2BatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyGPT2GGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyGPTNeoXBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyGPTNeoXGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyGPTJBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyGPTJGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyBLOOMBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyBLOOMGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyMPTBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyMPTGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

// buildTinyFalconGGUF (the 40B/180B "new decoder architecture" variant with a
// separate attn_norm_2) is the more structurally demanding of the two Falcon
// GGUF shapes model_families_extra_test.go exercises, so it is the one
// reused here.
func BenchmarkTinyFalconBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyFalconGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyStarCoderBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyStarCoderGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyStarCoder2BatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyStarCoder2GGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyChatGLMBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyChatGLMGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyGLM4BatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyGLM4GGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyCommandRBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyCommandRGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyMiniCPMBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyMiniCPMGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

func BenchmarkTinyGraniteBatchedPrefillReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyGraniteGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ForwardBatchInto(r.config, r.standard, cache, buf, tokens, 0, true, &logits)
	}
}

// GraniteMoE has sparse experts, so canBatchPrefill() is false for it and
// ForwardBatchInto (which has no MoE routing) is not a code path real
// traffic ever takes for this architecture. Benchmark the per-token decode
// path instead, mirroring runExtraArchSmoke's MoE fallback in
// model_families_extra_test.go.
func BenchmarkTinyGraniteMoEDecodeReuse(b *testing.B) {
	r, err := RunnerFromGGUFBytes(buildTinyGraniteMoEGGUF())
	if err != nil {
		b.Fatal(err)
	}
	tokens := extraBenchTokens(r)
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := []float32{}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		for pos, tok := range tokens {
			logits = Forward(r.config, r.standard, cache, buf, tok, pos)
		}
	}
	_ = logits
}
