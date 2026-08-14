//go:build !amd64

package gopherllm

import (
	"math"
	"testing"
)

// withQ8Activations runs fn with the int8-activation path forced on or off.
// Non-amd64 targets also expose this switch: Apple Silicon has an SDOT-backed
// implementation, while other targets use the portable kernel when enabled.
func withQ8Activations(enabled bool, fn func()) {
	saved := useQ8Activations.Load()
	useQ8Activations.Store(enabled)
	defer useQ8Activations.Store(saved)
	fn()
}

// requireCosine mirrors the amd64 helper of the same name (q4k_q8_amd64_test.go)
// so int8-vs-float matvec comparison tests can run on every target, including
// the ones (like Q3_K, which has no AVX2/NEON kernel yet) whose fast path is
// the portable scalar kernel everywhere.
func requireCosine(t *testing.T, name string, fout, qout []float32) {
	t.Helper()
	cos, err := CosineSimilarity(fout, qout)
	if err != nil {
		t.Fatal(err)
	}
	if cos < 0.999 {
		t.Fatalf("%s: int8 matvec cosine similarity %.5f < 0.999", name, cos)
	}
	var maxAbs float32
	for _, v := range fout {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	for i := range fout {
		if math.Abs(float64(fout[i])) > 0.5*float64(maxAbs) {
			rel := math.Abs(float64(qout[i]-fout[i])) / math.Abs(float64(fout[i]))
			if rel > 0.05 {
				t.Fatalf("%s row %d: int8 %v vs float %v (rel %.4f)", name, i, qout[i], fout[i], rel)
			}
		}
	}
}
