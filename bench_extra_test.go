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
