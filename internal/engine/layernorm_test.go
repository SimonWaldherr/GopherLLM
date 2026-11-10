package engine

import (
	"math"
	"math/rand"
	"testing"
)

// layerNormReference is the straightforward float64 definition that
// layerNormInto must match: normalise to zero mean / unit variance, then apply
// the per-element weight (default 1) and bias (default 0).
func layerNormReference(x, weight, bias []float32, eps float32) []float64 {
	n := len(x)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(n)
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(n)
	scale := 1 / math.Sqrt(variance+float64(eps))
	out := make([]float64, n)
	for i, v := range x {
		w, b := 1.0, 0.0
		if i < len(weight) {
			w = float64(weight[i])
		}
		if i < len(bias) {
			b = float64(bias[i])
		}
		out[i] = (float64(v)-mean)*scale*w + b
	}
	return out
}

func TestLayerNormIntoMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 31, 33, 64, 127, 768, 1025, 4096} {
		x := randomVec(rng, n)
		for i := range x {
			x[i] = x[i]*3 + 0.75 // non-zero mean, non-unit variance
		}
		full := randomVec(rng, n)
		for _, tc := range []struct {
			name         string
			weight, bias []float32
		}{
			{"weight+bias", full, randomVec(rng, n)},
			{"weight only", full, nil},
			{"neither", nil, nil},
			{"bias only", nil, randomVec(rng, n)},
			{"short weight+bias", full[:n/2], full[:n/3]},
		} {
			t.Run(tc.name, func(t *testing.T) {
				want := layerNormReference(x, tc.weight, tc.bias, 1e-5)
				var got []float32
				layerNormInto(x, tc.weight, tc.bias, 1e-5, &got)
				if len(got) != n {
					t.Fatalf("n=%d: len(out)=%d", n, len(got))
				}
				for i := range got {
					if diff := math.Abs(float64(got[i]) - want[i]); diff > 1e-4*(1+math.Abs(want[i])) {
						t.Fatalf("n=%d out[%d]=%v want %v", n, i, got[i], want[i])
					}
				}

				// In-place use (out aliasing x) is how BERT and the
				// Parakeet encoder call it; it must give the same result.
				inPlace := append([]float32(nil), x...)
				layerNormInto(inPlace, tc.weight, tc.bias, 1e-5, &inPlace)
				for i := range got {
					if inPlace[i] != got[i] {
						t.Fatalf("n=%d in-place out[%d]=%v differs from out-of-place %v", n, i, inPlace[i], got[i])
					}
				}
			})
		}
	}
}

func TestLayerNormIntoEmpty(t *testing.T) {
	var out []float32
	layerNormInto(nil, nil, nil, 1e-5, &out)
	if len(out) != 0 {
		t.Fatalf("len(out)=%d for empty input", len(out))
	}
}
