package gopherllm

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

func TestBatchedPrefillMatchesPerToken(t *testing.T) {
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
	r.prefillBatched(c2, b2, tokens, &got)

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
	}
}
