package gopherllm

import (
	"context"
	"math"
	"math/rand"
	"strings"
	"sync/atomic"
	"testing"
)

func TestReuseBatchViewsReusesBackingStorage(t *testing.T) {
	flat := []float32{}
	views := [][]float32{}
	first := reuseBatchViews(&flat, &views, 8, 16)
	first[0][0] = 42
	ptr := &flat[0]
	second := reuseBatchViews(&flat, &views, 4, 16)
	if &flat[0] != ptr {
		t.Fatal("batch backing array was reallocated for a smaller chunk")
	}
	if len(second) != 4 || len(second[0]) != 16 || second[0][0] != 42 {
		t.Fatalf("reused views have unexpected shape or backing: %dx%d first=%v", len(second), len(second[0]), second[0][0])
	}
}

func TestParallelRowsBatchedUsesOneRangePerWorker(t *testing.T) {
	threads := numThreads()
	rows := threads * 128
	var calls atomic.Int32
	var covered atomic.Int64
	parallelRowsBatched(rows, func(start, end int) {
		calls.Add(1)
		covered.Add(int64(end - start))
	})
	if got := int(calls.Load()); got != threads {
		t.Fatalf("batch dispatch ranges = %d, want one per worker (%d)", got, threads)
	}
	if got := int(covered.Load()); got != rows {
		t.Fatalf("batch dispatch covered %d rows, want %d", got, rows)
	}
}

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

func TestMistral3TemperatureScalingMatchesBatchedPrefill(t *testing.T) {
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "32")
	r, err := RunnerFromGGUFBytes(buildTinyStandardGGUF("mistral3"))
	if err != nil {
		t.Fatal(err)
	}
	// Keep the synthetic model short while crossing multiple temperature
	// floors; a real Ministral-3-14B uses 16,384 here.
	r.config.AttentionTemperatureScale = 0.1
	r.config.AttentionTemperatureFloor = 2
	tokens := r.tok.Encode(strings.Repeat("abcdefghij", 3))
	kDim, vDim, mh, mk, mv := r.cacheDims()
	newRun := func() (*KVCache, *DecodeBuffer) {
		return NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1), NewDecodeBuffer(r.config, mh, mk, mv)
	}

	perCache, perBuf := newRun()
	perToken := []float32{}
	for pos, tok := range tokens {
		if pos == len(tokens)-1 {
			r.forwardTokenInto(perCache, perBuf, tok, pos, &perToken)
		} else {
			r.forwardPrefillToken(perCache, perBuf, tok, pos)
		}
	}
	batchCache, batchBuf := newRun()
	batched := []float32{}
	if err := r.prefillBatched(context.Background(), batchCache, batchBuf, tokens, &batched); err != nil {
		t.Fatal(err)
	}
	if len(batched) != len(perToken) {
		t.Fatalf("logit len %d vs %d", len(batched), len(perToken))
	}
	for i := range perToken {
		if d := math.Abs(float64(batched[i] - perToken[i])); d > 1e-3*math.Max(1, math.Abs(float64(perToken[i]))) {
			t.Fatalf("logit %d: batched=%v per-token=%v", i, batched[i], perToken[i])
		}
	}

	// The schedule must have a real effect past its floor, so this test cannot
	// pass merely because both prefill implementations skipped it.
	saved := r.config.AttentionTemperatureScale
	r.config.AttentionTemperatureScale = 0
	plainCache, plainBuf := newRun()
	plain := []float32{}
	for pos, tok := range tokens {
		if pos == len(tokens)-1 {
			r.forwardTokenInto(plainCache, plainBuf, tok, pos, &plain)
		} else {
			r.forwardPrefillToken(plainCache, plainBuf, tok, pos)
		}
	}
	r.config.AttentionTemperatureScale = saved
	different := false
	for i := range plain {
		if math.Abs(float64(plain[i]-perToken[i])) > 1e-6 {
			different = true
			break
		}
	}
	if !different {
		t.Fatal("temperature schedule did not affect long-context logits")
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

func TestBatchedPrefillMatchesPerTokenForStableLM(t *testing.T) {
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "32")
	r, err := RunnerFromGGUFBytes(buildTinyStandardGGUF("stablelm"))
	if err != nil {
		t.Fatal(err)
	}
	if !r.config.UseLayerNorm || r.config.ParallelResidual || !r.canBatchPrefill() {
		t.Fatalf("Stable-Code-style StableLM should use batched sequential LayerNorm prefill: %+v", r.config)
	}
	tokens := r.tok.Encode(strings.Repeat("abcdefghij", 10))
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
	if err := r.prefillBatched(context.Background(), c2, b2, tokens, &got); err != nil {
		t.Fatal(err)
	}
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
		for i := range c1.V[l] {
			if d := math.Abs(float64(c2.V[l][i] - c1.V[l][i])); d > 1e-3 {
				t.Fatalf("layer %d V[%d]: batched=%v per-token=%v", l, i, c2.V[l][i], c1.V[l][i])
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
	q5kData := make([]byte, rows*(cols/256)*176)
	q6kData := make([]byte, 0, rows*(cols/256)*210)
	for r := 0; r < rows; r++ {
		q4kData = append(q4kData, randomQ4KRow(rng, cols)...)
		q6kData = append(q6kData, randomQ6KRow(rng, cols)...)
	}
	for block := 0; block < len(q5kData)/176; block++ {
		base := block * 176
		q5kData[base+1] = 0x3c // f16 1.0 scale
		for i := 4; i < 176; i++ {
			q5kData[base+i] = byte(rng.Intn(256))
		}
	}
	legacyData := func(blockBytes int) []byte {
		out := make([]byte, rows*(cols/32)*blockBytes)
		for block := 0; block < len(out)/blockBytes; block++ {
			base := block * blockBytes
			// f16 1.0 scale; arbitrary deterministic packed values thereafter.
			out[base+1] = 0x3c
			for i := 2; i < blockBytes; i++ {
				out[base+i] = byte(rng.Intn(256))
			}
		}
		return out
	}
	mxfp4Data := func() []byte {
		out := make([]byte, rows*(cols/32)*17)
		for block := 0; block < len(out)/17; block++ {
			base := block * 17
			for i := 0; i < 16; i++ {
				out[base+i] = byte(rng.Intn(256))
			}
			out[base+16] = 127 // scale 1; avoids overflow in differential checks.
		}
		return out
	}

	weights := map[string]Weight{
		"f32":   {F32: randomVec(rng, rows*cols)},
		"q4_0":  {Raw: legacyData(18), Type: GGMLTypeQ4_0, Rows: rows, Cols: cols},
		"q8_0":  {Raw: legacyData(34), Type: GGMLTypeQ8_0, Rows: rows, Cols: cols},
		"q4k":   {Raw: q4kData, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols},
		"q5k":   {Raw: q5kData, Type: GGMLTypeQ5_K, Rows: rows, Cols: cols},
		"q6k":   {Raw: q6kData, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols},
		"mxfp4": {Raw: mxfp4Data(), Type: GGMLTypeMXFP4, Rows: rows, Cols: cols},
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

// Q/K/V and gate/up prefill fusion must preserve the exact Q8-activation row
// results of issuing the same batch projections separately. The three weights
// intentionally use Q4_K, Q4_K, and Q6_K — the mixed layout used by real
// Ministral and Granite checkpoints.
func TestMatvecBatchQ8FusedProjectionsMatchSeparate(t *testing.T) {
	rng := rand.New(rand.NewSource(58))
	const (
		cols = 512
		p    = 32
	)
	makeWeight := func(typ GGMLType, rows int) Weight {
		var raw []byte
		for range rows {
			switch typ {
			case GGMLTypeQ4_K:
				raw = append(raw, randomQ4KRow(rng, cols)...)
			case GGMLTypeQ6_K:
				raw = append(raw, randomQ6KRow(rng, cols)...)
			default:
				t.Fatalf("unsupported fixture type %s", typ)
			}
		}
		return Weight{Raw: raw, Type: typ, Rows: rows, Cols: cols}
	}
	newOutputs := func(rows int) [][]float32 {
		out := make([][]float32, p)
		for i := range out {
			out[i] = make([]float32, rows)
		}
		return out
	}
	xs := make([][]float32, p)
	for i := range xs {
		xs[i] = randomVec(rng, cols)
	}
	a := makeWeight(GGMLTypeQ4_K, 72)
	b := makeWeight(GGMLTypeQ4_K, 48)
	c := makeWeight(GGMLTypeQ6_K, 40)
	wantA, wantB, wantC := newOutputs(a.Rows), newOutputs(b.Rows), newOutputs(c.Rows)
	gotA, gotB, gotC := newOutputs(a.Rows), newOutputs(b.Rows), newOutputs(c.Rows)

	withQ8Activations(true, func() {
		matvecBatch(a, xs, wantA)
		matvecBatch(b, xs, wantB)
		matvecBatch(c, xs, wantC)
		matvecBatch3(a, b, c, xs, gotA, gotB, gotC)
	})
	for label, pair := range map[string]struct{ got, want [][]float32 }{
		"q": {gotA, wantA},
		"k": {gotB, wantB},
		"v": {gotC, wantC},
	} {
		for token := range p {
			for row := range pair.want[token] {
				if got, want := pair.got[token][row], pair.want[token][row]; got != want {
					t.Fatalf("%s token %d row %d: fused=%v separate=%v", label, token, row, got, want)
				}
			}
		}
	}
}

func TestSameTypeQ4_0FusionMatchesSeparateMatvecs(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	const cols = 64
	makeWeight := func(rows int) Weight {
		data := make([]byte, rows*(cols/32)*18)
		for block := 0; block < len(data)/18; block++ {
			base := block * 18
			data[base+1] = 0x3c // f16 1.0
			for i := 2; i < 18; i++ {
				data[base+i] = byte(rng.Intn(256))
			}
		}
		return Weight{Raw: data, Type: GGMLTypeQ4_0, Rows: rows, Cols: cols}
	}
	a, b, c := makeWeight(7), makeWeight(5), makeWeight(3)
	x := randomVec(rng, cols)
	withQ8Activations(false, func() {
		wantA, wantB, wantC := a.Matvec(x), b.Matvec(x), c.Matvec(x)
		gotA, gotB, gotC := []float32{}, []float32{}, []float32{}
		if !tryMatvec3Into(a, b, c, x, &[]float32{}, &gotA, &gotB, &gotC) {
			t.Fatal("same-type Q4_0 fusion declined valid weights")
		}
		for name, got := range map[string][]float32{"a": gotA, "b": gotB, "c": gotC} {
			var want []float32
			switch name {
			case "a":
				want = wantA
			case "b":
				want = wantB
			default:
				want = wantC
			}
			for i := range want {
				if d := math.Abs(float64(got[i] - want[i])); d > 1e-5*math.Max(1, math.Abs(float64(want[i]))) {
					t.Fatalf("%s[%d]: fused=%v separate=%v", name, i, got[i], want[i])
				}
			}
		}
	})
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
