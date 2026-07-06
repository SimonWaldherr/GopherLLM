package gopherllm

import (
	"context"
	"math"
	"math/rand"
	"strings"
	"testing"
)

func TestBatchedPrefillMatchesPerToken(t *testing.T) {
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "32")
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	if !r.canBatchPrefill() {
		t.Fatal("tiny llama should support batched prefill")
	}
	// Repeated well past the prefill chunk size so this exercises multi-chunk
	// stitching (KV cache continuity across chunk boundaries), not just a
	// single chunk's worth of tokens.
	tokens := r.tok.Encode(strings.Repeat("abcdefghij", 10))
	if len(tokens) < 80 {
		t.Fatalf("need a prompt spanning multiple prefill chunks, got %d tokens", len(tokens))
	}

	kDim, vDim, mh, mk, mv := r.cacheDims()
	newRun := func() (*KVCache, *DecodeBuffer) {
		return NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1), NewDecodeBuffer(r.config, mh, mk, mv)
	}

	// Per-token reference.
	c1, b1 := newRun()
	ref := []float32{}
	for pos, tok := range tokens {
		if pos == len(tokens)-1 {
			r.forwardTokenInto(c1, b1, tok, pos, &ref)
		} else {
			r.forwardPrefillToken(c1, b1, tok, pos)
		}
	}

	// Batched.
	c2, b2 := newRun()
	got := []float32{}
	_ = r.prefillBatched(context.Background(), c2, b2, tokens, &got)

	if len(got) != len(ref) {
		t.Fatalf("logit len %d vs %d", len(got), len(ref))
	}
	for i := range ref {
		if d := math.Abs(float64(got[i] - ref[i])); d > 1e-3*math.Max(1, math.Abs(float64(ref[i]))) {
			t.Fatalf("logit %d: batched=%v per-token=%v", i, got[i], ref[i])
		}
	}
	// KV caches must match too.
	for l := range c1.K {
		for i := range c1.K[l] {
			if d := math.Abs(float64(c2.K[l][i] - c1.K[l][i])); d > 1e-3 {
				t.Fatalf("layer %d K[%d]: batched=%v per-token=%v", l, i, c2.K[l][i], c1.K[l][i])
			}
		}
	}
}

func TestBatchedPrefillSupportsFusedQKVAndGateUp(t *testing.T) {
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "32")
	r, err := RunnerFromGGUFBytes(buildTinyFusedLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	if !r.standard.Layers[0].HasQKV || !r.standard.Layers[0].HasGateUp {
		t.Fatal("test model did not load fused tensors")
	}
	if !r.canBatchPrefill() {
		t.Fatal("fused llama-style model should support batched prefill")
	}
	tokens := r.tok.Encode(strings.Repeat("abcdefghij", 10))
	if len(tokens) < 80 {
		t.Fatalf("need a prompt spanning multiple prefill chunks, got %d tokens", len(tokens))
	}

	kDim, vDim, mh, mk, mv := r.cacheDims()
	newRun := func() (*KVCache, *DecodeBuffer) {
		return NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1), NewDecodeBuffer(r.config, mh, mk, mv)
	}

	c1, b1 := newRun()
	ref := []float32{}
	for pos, tok := range tokens {
		if pos == len(tokens)-1 {
			r.forwardTokenInto(c1, b1, tok, pos, &ref)
		} else {
			r.forwardPrefillToken(c1, b1, tok, pos)
		}
	}

	c2, b2 := newRun()
	got := []float32{}
	_ = r.prefillBatched(context.Background(), c2, b2, tokens, &got)

	if len(got) != len(ref) {
		t.Fatalf("logit len %d vs %d", len(got), len(ref))
	}
	for i := range ref {
		if d := math.Abs(float64(got[i] - ref[i])); d > 1e-3*math.Max(1, math.Abs(float64(ref[i]))) {
			t.Fatalf("logit %d: batched=%v per-token=%v", i, got[i], ref[i])
		}
	}
	for l := range c1.K {
		for i := range c1.K[l] {
			if d := math.Abs(float64(c2.K[l][i] - c1.K[l][i])); d > 1e-3 {
				t.Fatalf("layer %d K[%d]: batched=%v per-token=%v", l, i, c2.K[l][i], c1.K[l][i])
			}
		}
	}
}

func TestMatvecBatchMatchesPerToken(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const rows, cols, P = 20, 512, 4

	xs := make([][]float32, P)
	for p := range xs {
		xs[p] = randomVec(rng, cols)
	}

	q4kData := make([]byte, 0, rows*(cols/256)*144)
	q6kData := make([]byte, 0, rows*(cols/256)*210)
	for r := 0; r < rows; r++ {
		q4kData = append(q4kData, randomQ4KRow(rng, cols)...)
		q6kData = append(q6kData, randomQ6KRow(rng, cols)...)
	}

	weights := map[string]Weight{
		"f32": {F32: randomVec(rng, rows*cols)},
		"q4k": {Raw: q4kData, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols},
		"q6k": {Raw: q6kData, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols},
	}

	for name, w := range weights {
		// Force the float kernels on both sides: this test checks the batched
		// dequantize-once bookkeeping against the per-token matvec exactly,
		// which the int8-activation default would blur with quantization error
		// (covered separately by TestMatvecBatchQ8CloseToFloat).
		withQ8Activations(false, func() {
			want := make([][]float32, P)
			for p := range want {
				want[p] = w.Matvec(xs[p])
			}
			got := make([][]float32, P)
			for p := range got {
				got[p] = make([]float32, rows)
			}
			matvecBatch(w, xs, got)
			for p := 0; p < P; p++ {
				for r := 0; r < rows; r++ {
					d := math.Abs(float64(got[p][r] - want[p][r]))
					if d > 1e-3*math.Max(1, math.Abs(float64(want[p][r]))) {
						t.Fatalf("%s token %d row %d: batch=%v per-token=%v", name, p, r, got[p][r], want[p][r])
					}
				}
			}
		})
	}
}

func buildTinyFusedLlamaGGUF() []byte {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads
		hidden = 16
		vocab  = 16
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{"<unk>", "<s>", "</s>"}
	for i := 0; i < vocab; i++ {
		if i < len(special) {
			toks[i] = special[i]
		} else {
			toks[i] = string(rune('a' + (i - len(special))))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "llama"},
		{"general.name", ggufStr, "tiny-fused"},
		{"llama.embedding_length", ggufU32, uint32(dim)},
		{"llama.block_count", ggufU32, uint32(1)},
		{"llama.attention.head_count", ggufU32, uint32(heads)},
		{"llama.attention.head_count_kv", ggufU32, uint32(kv)},
		{"llama.attention.key_length", ggufU32, uint32(hdim)},
		{"llama.attention.value_length", ggufU32, uint32(hdim)},
		{"llama.feed_forward_length", ggufU32, uint32(hidden)},
		{"llama.context_length", ggufU32, uint32(1024)},
		{"llama.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{"llama.rope.freq_base", ggufF32, float32(10000)},
		{"llama.rope.dimension_count", ggufU32, uint32(hdim)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	fusedF32t := func(name string, cols int, parts ...[]float32) ggufTensor {
		rows := 0
		for _, part := range parts {
			rows += len(part) / cols
		}
		data := make([]float32, 0, rows*cols)
		for _, part := range parts {
			data = append(data, part...)
		}
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(data)}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	q := smallWeights(heads*hdim*dim, 3)
	k := smallWeights(kv*hdim*dim, 4)
	v := smallWeights(kv*hdim*dim, 5)
	gate := smallWeights(hidden*dim, 7)
	up := smallWeights(hidden*dim, 8)
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim),
		fusedF32t("blk.0.attn_qkv.weight", dim, q, k, v),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		vec("blk.0.ffn_norm.weight", dim),
		fusedF32t("blk.0.ffn_up.weight", dim, gate, up),
		f32t("blk.0.ffn_down.weight", dim, hidden, 9),
	}
	return buildGGUF(3, kvs, tensors)
}
