package gopherllm

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

// withI8KVCache runs fn with the int8 KV cache forced on or off. Unlike
// withF16KVCache, this needs no per-platform variant: useI8KVCache has one
// definition everywhere (see kv_i8.go).
func withI8KVCache(enabled bool, fn func()) {
	saved := useI8KVCache.Load()
	useI8KVCache.Store(enabled)
	defer useI8KVCache.Store(saved)
	fn()
}

func TestQ8RowByteMathRoundTrips(t *testing.T) {
	for _, n := range []int{32, 64, 96, 224, 1024} {
		b := q8RowBytes(n)
		if want := (n / 32) * 34; b != want {
			t.Fatalf("q8RowBytes(%d) = %d, want %d", n, b, want)
		}
		if got := q8RowElems(b); got != n {
			t.Fatalf("q8RowElems(q8RowBytes(%d)) = %d, want %d", n, got, n)
		}
	}
	// Non-block-aligned input: q8RowBytes floors to whole blocks, matching
	// QuantizeRowQ8_0/DotQ8_0F32's own cols/32 derivation.
	if got := q8RowBytes(40); got != 34 {
		t.Fatalf("q8RowBytes(40) = %d, want 34 (one whole block)", got)
	}
}

func TestKvI8Eligible(t *testing.T) {
	cases := []struct {
		name                          string
		kDim, vDim, headDim, valueDim int
		want                          bool
	}{
		{"all 32-aligned", 1024, 1024, 128, 128, true},
		{"MLA-shaped real dims", 576, 512, 576, 512, true},
		{"headDim not aligned", 1024, 1024, 80, 80, false},
		{"kDim not aligned (deepseek2 tiny fixture)", 5, 3, 5, 3, false},
		{"valueDim not aligned", 128, 96, 32, 24, false},
		{"zero dims", 0, 0, 0, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := kvI8Eligible(tc.kDim, tc.vDim, tc.headDim, tc.valueDim); got != tc.want {
				t.Fatalf("kvI8Eligible(%d,%d,%d,%d) = %v, want %v", tc.kDim, tc.vDim, tc.headDim, tc.valueDim, got, tc.want)
			}
		})
	}
}

func TestAxpyQ8RowMatchesNaiveDequant(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	for _, n := range []int{32, 64, 96, 224} {
		row := randomVec(rng, n)
		q8 := QuantizeRowQ8_0(row, n)

		out1 := randomVec(rng, n)
		out2 := append([]float32(nil), out1...)

		axpyQ8Row(out1, 0.6, q8)
		AxpyF32(out2, 0.6, DequantRowQ8_0(q8, n))

		for i := range out1 {
			if diff := math.Abs(float64(out1[i] - out2[i])); diff > 1e-4*(1+math.Abs(float64(out2[i]))) {
				t.Fatalf("n=%d axpyQ8Row[%d] = %v, want %v (naive dequant+AxpyF32)", n, i, out1[i], out2[i])
			}
		}
	}
}

func TestDotQ8_0F32MatchesNaiveDequant(t *testing.T) {
	rng := rand.New(rand.NewSource(72))
	for _, n := range []int{32, 64, 96, 224} {
		row := randomVec(rng, n)
		q8 := QuantizeRowQ8_0(row, n)
		query := randomVec(rng, n)

		got := DotQ8_0F32(q8, query, n)
		want := DotF32(query, DequantRowQ8_0(q8, n))
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("n=%d DotQ8_0F32 = %v, want %v (naive dequant+DotF32)", n, got, want)
		}
	}
}

// TestOnlineAttentionI8MatchesF32 mirrors TestOnlineAttentionF16MatchesF32:
// drive both attention variants over the same K/V content (Q8_0-rounded in
// both so only the kernel paths differ). Q8_0's coarser, data-dependent
// quantization needs a looser tolerance than f16's fixed conversion.
func TestOnlineAttentionI8MatchesF32(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	const headDim, ctx = 64, 96
	q := randomVec(rng, headDim)
	kf := randomVec(rng, ctx*headDim)
	vf := randomVec(rng, ctx*headDim)
	k8 := make([]byte, 0, ctx*q8RowBytes(headDim))
	v8 := make([]byte, 0, ctx*q8RowBytes(headDim))
	kfQ := make([]float32, ctx*headDim)
	vfQ := make([]float32, ctx*headDim)
	for t := 0; t < ctx; t++ {
		kRow := QuantizeRowQ8_0(kf[t*headDim:(t+1)*headDim], headDim)
		vRow := QuantizeRowQ8_0(vf[t*headDim:(t+1)*headDim], headDim)
		k8 = append(k8, kRow...)
		v8 = append(v8, vRow...)
		copy(kfQ[t*headDim:], DequantRowQ8_0(kRow, headDim))
		copy(vfQ[t*headDim:], DequantRowQ8_0(vRow, headDim))
	}
	scale := float32(1 / math.Sqrt(headDim))
	outF32 := make([]float32, headDim)
	outI8 := make([]float32, headDim)
	onlineAttention(q, kfQ, vfQ, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, outF32)
	onlineAttentionI8WithSink(q, k8, v8, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, 0, 0, false, outI8)
	cos, err := CosineSimilarity(outF32, outI8)
	if err != nil {
		t.Fatal(err)
	}
	if cos < 0.999 {
		t.Fatalf("f32 vs int8 attention output cosine similarity %.5f < 0.999: f32=%v i8=%v", cos, outF32, outI8)
	}

	// Softcap branch too.
	clear(outF32)
	clear(outI8)
	onlineAttention(q, kfQ, vfQ, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 50, outF32)
	onlineAttentionI8WithSink(q, k8, v8, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 50, 0, 0, false, outI8)
	cos, err = CosineSimilarity(outF32, outI8)
	if err != nil {
		t.Fatal(err)
	}
	if cos < 0.999 {
		t.Fatalf("softcap: f32 vs int8 attention output cosine similarity %.5f < 0.999", cos)
	}
}

func TestCopyKVPrefixCopiesI8Rows(t *testing.T) {
	const kDim, vDim = 64, 64 // 32-aligned, unlike the f32/f16 test's kDim=3,vDim=2
	src := NewKVCacheI8(2, kDim, vDim, 5)
	dst := NewKVCacheI8(2, kDim, vDim, 5)
	rng := rand.New(rand.NewSource(74))
	for layer := range src.K8 {
		for pos := 0; pos < 5; pos++ {
			k := randomVec(rng, kDim)
			v := randomVec(rng, vDim)
			src.storeKV(layer, pos, k, v)
		}
	}
	if copied := copyKVPrefix(dst, src, 3); copied != 3 {
		t.Fatalf("copied positions = %d, want 3", copied)
	}
	kRowBytes := q8RowBytes(kDim)
	vRowBytes := q8RowBytes(vDim)
	for layer := 0; layer < 2; layer++ {
		if got, want := dst.K8[layer][:3*kRowBytes], src.K8[layer][:3*kRowBytes]; !bytesEqual(got, want) {
			t.Fatalf("layer %d K8 prefix mismatch", layer)
		}
		if got, want := dst.V8[layer][:3*vRowBytes], src.V8[layer][:3*vRowBytes]; !bytesEqual(got, want) {
			t.Fatalf("layer %d V8 prefix mismatch", layer)
		}
		for _, b := range dst.K8[layer][3*kRowBytes:] {
			if b != 0 {
				t.Fatalf("layer %d copied a K8 suffix", layer)
			}
		}
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// buildTiny32AlignedLlamaGGUF is buildTinyStandardGGUF with head_dim=32
// (dim=64, heads=2) instead of head_dim=4 — large enough for the int8 KV
// cache's kvI8Eligible gate to actually admit it, unlike every other tiny
// fixture in this test suite (all deliberately sized small-and-fast, with
// head dims nowhere near a multiple of 32).
func buildTiny32AlignedLlamaGGUF() []byte {
	const (
		arch   = "llama"
		dim    = 64
		heads  = 2
		kv     = 2
		hdim   = dim / heads // 32
		hidden = 64
		vocab  = 32
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
		{"general.architecture", ggufStr, arch},
		{"general.name", ggufStr, "tiny32"},
		{arch + ".embedding_length", ggufU32, uint32(dim)},
		{arch + ".block_count", ggufU32, uint32(1)},
		{arch + ".attention.head_count", ggufU32, uint32(heads)},
		{arch + ".attention.head_count_kv", ggufU32, uint32(kv)},
		{arch + ".attention.key_length", ggufU32, uint32(hdim)},
		{arch + ".attention.value_length", ggufU32, uint32(hdim)},
		{arch + ".feed_forward_length", ggufU32, uint32(hidden)},
		{arch + ".context_length", ggufU32, uint32(1024)},
		{arch + ".attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{arch + ".rope.freq_base", ggufF32, float32(10000)},
		{arch + ".rope.dimension_count", ggufU32, uint32(hdim)},
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
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 3),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 4),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 5),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		f32t("blk.0.ffn_gate.weight", hidden, dim, 7),
		f32t("blk.0.ffn_up.weight", hidden, dim, 8),
		f32t("blk.0.ffn_down.weight", dim, hidden, 9),
		vec("blk.0.ffn_norm.weight", dim),
	}
	return buildGGUF(3, kvs, tensors)
}

// TestGenerateWithI8KVCacheMatchesF32 mirrors TestGenerateWithF16KVCacheMatchesF32.
// int8's quantization is considerably coarser than f16's, so this was not
// assumed to be bit-identical going in — it was verified empirically on this
// fixture first (12 greedy steps, tiny random weights) before writing this as
// a hard equality assertion. That does not guarantee every model/prompt stays
// bit-identical under int8 KV rounding; TestOnlineAttentionI8MatchesF32's
// cosine-similarity check is the bound that's expected to generalize.
func TestGenerateWithI8KVCacheMatchesF32(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTiny32AlignedLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 12
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1

	run := func() string {
		res, err := m.Runner().Generate("a b c d", opts)
		if err != nil {
			t.Fatal(err)
		}
		return res.Text
	}
	var f32Text, i8Text string
	withI8KVCache(false, func() { f32Text = run() })
	withI8KVCache(true, func() { i8Text = run() })
	if f32Text != i8Text {
		t.Fatalf("greedy decode diverged: f32=%q i8=%q", f32Text, i8Text)
	}
}
