package main

import "testing"

func BenchmarkOnlineAttention_ctx512(b *testing.B) {
	const headDim, ctx = 128, 512
	q := benchFloatSlice(headDim)
	keys := benchFloatSlice(ctx * headDim)
	values := benchFloatSlice(ctx * headDim)
	out := make([]float32, headDim)
	scale := float32(0.08838)
	b.ReportAllocs()
	for b.Loop() {
		clear(out)
		onlineAttention(q, keys, values, headDim, headDim, headDim, headDim, 0, ctx-1, scale, out)
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
