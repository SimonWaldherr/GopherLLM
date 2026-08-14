package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// Synthetic vision tower for measuring EncodeImagePixtral. The weights are
// random F32 rather than a real checkpoint: the encoder's cost depends only on
// the shapes, and this keeps the benchmark runnable without a multi-gigabyte
// mmproj on disk.
func benchVisionConfig(layers int) PixtralVisionConfig {
	return PixtralVisionConfig{
		EmbeddingLength:   256,
		FeedForwardLength: 1024,
		HeadCount:         8,
		HeadDim:           32,
		BlockCount:        layers,
		PatchSize:         16,
		SpatialMergeSize:  2,
		ProjectionDim:     256,
		Epsilon:           1e-5,
		RopeTheta:         10000,
		UseSiLU:           true,
	}
}

func benchF32Weight(rng *rand.Rand, rows, cols int) Weight {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = rng.Float32()*0.1 - 0.05
	}
	return Weight{F32: data, Type: GGMLTypeF32, Rows: rows, Cols: cols}
}

func benchVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*0.1 + 0.95
	}
	return v
}

func benchVisionWeights(rng *rand.Rand, vc PixtralVisionConfig) PixtralVisionWeights {
	dim := vc.EmbeddingLength
	ff := vc.FeedForwardLength
	merge := vc.SpatialMergeSize
	w := PixtralVisionWeights{
		PatchEmbd:   benchF32Weight(rng, dim, 3*vc.PatchSize*vc.PatchSize),
		PreNorm:     benchVec(rng, dim),
		InputNorm:   benchVec(rng, dim),
		PatchMerger: benchF32Weight(rng, dim, dim*merge*merge),
		Proj1:       benchF32Weight(rng, vc.ProjectionDim, dim),
		Proj2:       benchF32Weight(rng, vc.ProjectionDim, vc.ProjectionDim),
	}
	for range vc.BlockCount {
		w.Layers = append(w.Layers, PixtralVisionLayerWeights{
			AttnNorm: benchVec(rng, dim),
			Q:        benchF32Weight(rng, dim, dim),
			K:        benchF32Weight(rng, dim, dim),
			V:        benchF32Weight(rng, dim, dim),
			Out:      benchF32Weight(rng, dim, dim),
			FFNNorm:  benchVec(rng, dim),
			FFNGate:  benchF32Weight(rng, ff, dim),
			FFNUp:    benchF32Weight(rng, ff, dim),
			FFNDown:  benchF32Weight(rng, dim, ff),
		})
	}
	return w
}

func benchPreprocessedImage(rng *rand.Rand, vc PixtralVisionConfig, rows, cols int) *PreprocessedImage {
	patchLen := 3 * vc.PatchSize * vc.PatchSize
	pixels := make([][]float32, rows*cols)
	for i := range pixels {
		p := make([]float32, patchLen)
		for j := range p {
			p[j] = rng.Float32()*2 - 1
		}
		pixels[i] = p
	}
	return &PreprocessedImage{Pixels: pixels, Rows: rows, Cols: cols}
}

// BenchmarkEncodeImagePixtral covers the whole tower. The grid sizes bracket
// what the dynamic preprocessor actually produces: attention is O(n^2) in the
// patch count, so the larger grid is where the encoder's cost concentrates.
func BenchmarkEncodeImagePixtral(b *testing.B) {
	for _, tc := range []struct {
		name       string
		rows, cols int
	}{
		{"grid16x16", 16, 16},
		{"grid32x32", 32, 32},
	} {
		b.Run(tc.name, func(b *testing.B) {
			rng := rand.New(rand.NewSource(7))
			vc := benchVisionConfig(8)
			w := benchVisionWeights(rng, vc)
			img := benchPreprocessedImage(rng, vc, tc.rows, tc.cols)
			b.ResetTimer()
			for b.Loop() {
				if _, _, _, err := EncodeImagePixtral(vc, w, img); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestEncodeImagePixtralBenchFixtureIsSane keeps the benchmark honest: if the
// synthetic tower produced NaNs or a wrong token count, the benchmark would
// still "pass" while measuring nonsense.
func TestEncodeImagePixtralBenchFixtureIsSane(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	vc := benchVisionConfig(2)
	w := benchVisionWeights(rng, vc)
	img := benchPreprocessedImage(rng, vc, 8, 8)

	embeds, mr, mc, err := EncodeImagePixtral(vc, w, img)
	if err != nil {
		t.Fatal(err)
	}
	if mr != 4 || mc != 4 {
		t.Fatalf("merged grid = %dx%d, want 4x4 for an 8x8 grid at merge=2", mr, mc)
	}
	if len(embeds) != 16 {
		t.Fatalf("embeds = %d, want 16", len(embeds))
	}
	for i, e := range embeds {
		if len(e) != vc.ProjectionDim {
			t.Fatalf("embed %d has width %d, want %d", i, len(e), vc.ProjectionDim)
		}
		for j, v := range e {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("embed %d component %d is %v", i, j, v)
			}
		}
	}
}

// TestEncodeImagePixtralGoldenValues pins the encoder's actual output for a
// fixed synthetic tower. The fixture-sanity test above only checks shapes and
// finiteness, which a head-mixing or offset bug would sail straight through —
// attention reads K and V through a head-major staging layout, and getting that
// transpose wrong yields perfectly finite, perfectly wrong embeddings.
//
// These values were captured after verifying the head-major rewrite against the
// previous per-token-slice implementation: max relative difference 2.0e-5, mean
// 9.5e-7, which is the expected signature of routing the softmax through
// fastExpF32 and the FFN through siluMulF32 rather than libm. The tolerance
// here is looser than that gap so a future fast-math swap of similar accuracy
// does not trip it, but far tighter than any real bug.
func TestEncodeImagePixtralGoldenValues(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	vc := benchVisionConfig(4)
	w := benchVisionWeights(rng, vc)
	img := benchPreprocessedImage(rng, vc, 8, 8)

	embeds, _, _, err := EncodeImagePixtral(vc, w, img)
	if err != nil {
		t.Fatal(err)
	}
	if len(embeds) != 16 {
		t.Fatalf("embeds = %d, want 16", len(embeds))
	}

	// First and last merged tokens, so a bug confined to one end of the grid
	// still shows up.
	for _, want := range []struct {
		token int
		vals  [4]float32
	}{
		{0, [4]float32{0.0116726458, -0.036083214, 0.189651042, 0.0944179446}},
		{15, [4]float32{-0.0642388165, -0.0541599095, -0.0499021187, -0.183261067}},
	} {
		for j, wv := range want.vals {
			got := embeds[want.token][j]
			diff := math.Abs(float64(got - wv))
			tol := 1e-4 * math.Max(1, math.Abs(float64(wv)))
			if diff > tol {
				t.Errorf("embeds[%d][%d] = %.9g, want %.9g (diff %.3g > tol %.3g)",
					want.token, j, got, wv, diff, tol)
			}
		}
	}
}
