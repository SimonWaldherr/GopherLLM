package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func TestBatchViewsReleaseReplacedSlab(t *testing.T) {
	var flat []float32
	var rows [][]float32
	reuseBatchViews(&flat, &rows, 32, 256)
	old := &flat[0]
	// A shorter batch in a wider layer replaces the flat allocation but
	// reuses the row descriptors. Unused descriptors must release the old slab.
	reuseBatchViews(&flat, &rows, 17, 512)
	if &flat[0] == old {
		t.Fatal("fixture did not replace the slab")
	}
	for _, shape := range [][2]int{{17, 512}, {3, 128}, {0, 128}, {32, 128}} {
		p, width := shape[0], shape[1]
		reuseBatchViews(&flat, &rows, p, width)
		for i, row := range rows {
			if len(row) != width || cap(row) != width || &row[0] != &flat[i*width] {
				t.Fatalf("batch %d: row %d no longer views the current slab", p, i)
			}
		}
		for _, row := range rows[len(rows):cap(rows)] {
			if row != nil {
				t.Fatalf("batch %d: unused row retains activation storage", p)
			}
		}
	}
}

func TestBatchRowsInExplicitSlab(t *testing.T) {
	var flat []float32
	var rows [][]float32
	reuseBatchViews(&flat, &rows, 5, 7)
	got, ok := batchRowsInSlab(rows, 7, flat)
	if !ok || len(got) != 35 || &got[0] != &flat[0] {
		t.Fatal("clipped workspace views did not resolve to their explicit slab")
	}
	for _, row := range rows {
		if len(row) != 7 || cap(row) != 7 {
			t.Fatal("slab validation widened a row")
		}
	}
	for _, tc := range []struct {
		name  string
		rows  [][]float32
		width int
		slab  []float32
	}{
		{"empty", nil, 7, flat},
		{"zero width", rows, 0, flat},
		{"negative width", rows, -1, flat},
		{"overflow width", rows, int(^uint(0) >> 1), flat},
		{"missing slab", rows, 7, nil},
		{"short slab", rows, 7, flat[:34]},
		{"wrong allocation", rows, 7, make([]float32, 35)},
		{"wrong offset", rows[1:], 7, flat},
		{"reversed rows", [][]float32{rows[1], rows[0]}, 7, flat},
		{"short row", [][]float32{flat[:6]}, 7, flat},
		{"oversized row", [][]float32{flat[:8]}, 7, flat},
		{"empty row", [][]float32{nil}, 7, flat},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, ok := batchRowsInSlab(tc.rows, tc.width, tc.slab); ok {
				t.Fatal("invalid slab provenance accepted")
			}
		})
	}
	// A caller can provide a genuine sub-slab without exposing spare capacity.
	if got, ok := batchRowsInSlab(rows[1:4], 7, flat[7:28:28]); !ok || len(got) != 21 || &got[0] != &flat[7] {
		t.Fatal("valid sub-slab declined")
	}
}

func TestBatchSlabsKeepQuantizedFusion(t *testing.T) {
	withQ8Activations(true, func() {
		const tokens, cols = 33, 256
		rng := rand.New(rand.NewSource(18201))
		var inputs []float32
		var xs [][]float32
		reuseBatchViews(&inputs, &xs, tokens, cols)
		for i := range inputs {
			inputs[i] = rng.Float32()*2 - 1
		}
		weights := [3]Weight{
			{Type: GGMLTypeQ4_K, Rows: 11, Cols: cols},
			{Type: GGMLTypeQ6_K, Rows: 9, Cols: cols},
			{Type: GGMLTypeQ4_K, Rows: 7, Cols: cols},
		}
		var outputs, want [3][][]float32
		var flat [3][]float32
		for i := range weights {
			w := &weights[i]
			for range w.Rows {
				if w.Type == GGMLTypeQ4_K {
					w.Raw = append(w.Raw, randomQ4KRow(rng, cols)...)
				} else {
					w.Raw = append(w.Raw, randomQ6KRow(rng, cols)...)
				}
			}
			reuseBatchViews(&flat[i], &outputs[i], tokens, w.Rows)
			var wantFlat []float32
			reuseBatchViews(&wantFlat, &want[i], tokens, w.Rows)
		}
		matvecBatch3(weights[0], weights[1], weights[2], xs, want[0], want[1], want[2])
		matvecBatch3Slabs(weights[0], weights[1], weights[2], xs, outputs[0], outputs[1], outputs[2], inputs, flat[0], flat[1], flat[2])
		check := func(count int) {
			for projection := range count {
				for token := range tokens {
					for row, value := range want[projection][token] {
						if math.Float32bits(outputs[projection][token][row]) != math.Float32bits(value) {
							t.Fatalf("projection %d token %d row %d changed Q8 arithmetic", projection, token, row)
						}
					}
				}
			}
		}
		check(3)
		matvecBatch2Slabs(weights[0], weights[1], xs, outputs[0], outputs[1], inputs, flat[0], flat[1])
		check(2)
		matvecBatchSlabs(weights[0], xs, outputs[0], inputs, flat[0])
		check(1)
	})
}

func TestBatchedPrefillF32SlabsMatchesTokenGraph(t *testing.T) {
	// Unlike tiny GGUF fixtures, these dimensions cross the BLAS dispatch
	// threshold. A short first chunk checks reuse at a nonzero KV offset;
	// the second layer exercises fused QKV and gate/up slab handoffs.
	const dim, hidden, vocab, prefix, batch = 128, 256, 64, 7, 33
	config := Config{
		Dim: dim, HiddenDim: hidden, NLayers: 2, NHeads: 4, NKVHeads: 2, KVMul: 2,
		HeadDim: 32, ValueDim: 32, VocabSize: vocab, MaxSeqLen: prefix + batch,
		RMSNormEps: 1e-5, RopeTheta: 10000, RopeDimensionCount: 32,
		EmbeddingScale: 1, ResidualScale: 1, LogitScale: 1,
	}
	rng := rand.New(rand.NewSource(18202))
	weight := func(rows, cols int) Weight {
		w := Weight{F32: make([]float32, rows*cols), Rows: rows, Cols: cols}
		for i := range w.F32 {
			w.F32[i] = (rng.Float32()*2 - 1) * .05
		}
		return w
	}
	norm := func() []float32 {
		v := make([]float32, dim)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	weights := ModelWeights{TokenEmbd: weight(vocab, dim), Output: weight(vocab, dim), OutputNorm: norm(), Layers: make([]LayerWeights, 2)}
	for i := range weights.Layers {
		layer := &weights.Layers[i]
		layer.AttnNorm, layer.FFNNorm = norm(), norm()
		layer.WQ, layer.WK, layer.WV, layer.WO = weight(dim, dim), weight(dim/2, dim), weight(dim/2, dim), weight(dim, dim)
		layer.W1, layer.W3, layer.W2 = weight(hidden, dim), weight(hidden, dim), weight(dim, hidden)
	}
	weights.Layers[1].HasQKV, weights.Layers[1].HasGateUp = true, true
	weights.Layers[1].WQKV, weights.Layers[1].WGateUp = weight(dim*2, dim), weight(hidden*2, dim)
	tokens := make([]uint32, prefix+batch)
	for i := range tokens {
		tokens[i] = uint32(i % vocab)
	}
	newRun := func() (*KVCache, *DecodeBuffer) {
		return NewKVCache(2, dim/2, dim/2, len(tokens)), NewDecodeBuffer(config, 32, 2, 32)
	}
	refCache, refBuf := newRun()
	var want []float32
	for pos, token := range tokens {
		ForwardInto(config, weights, refCache, refBuf, token, pos, &want)
	}
	cache, buf := newRun()
	var got []float32
	ForwardBatchInto(config, weights, cache, buf, tokens[:prefix], 0, false, nil)
	ForwardBatchInto(config, weights, cache, buf, tokens[prefix:], prefix, true, &got)
	check := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s lengths differ: %d vs %d", label, len(got), len(want))
		}
		for i, value := range want {
			diff := math.Abs(float64(got[i]) - float64(value))
			if !finite32(got[i]) || diff > 1e-4*math.Max(1, math.Abs(float64(value))) {
				t.Fatalf("%s[%d] = %g, want %g", label, i, got[i], value)
			}
		}
	}
	check("logits", got, want)
	for layer := range cache.K {
		check("K", cache.K[layer], refCache.K[layer])
		check("V", cache.V[layer], refCache.V[layer])
	}
}
