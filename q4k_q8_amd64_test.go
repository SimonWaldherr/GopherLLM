//go:build amd64

package main

import (
	"math"
	"math/rand"
	"testing"
)

// TestQ4KQ8MatvecCloseToFloat checks that the opt-in int8-activation matvec
// stays very close to the exact float matvec: the output direction (what the
// next layer sees) must be preserved. int8 introduces a small magnitude error,
// so we bound cosine similarity rather than requiring an exact match.
func TestQ4KQ8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(3))
	const rows, cols = 96, 1024
	rowBytes := (cols / 256) * 144
	data := make([]byte, 0, rows*rowBytes)
	for r := 0; r < rows; r++ {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)

	fout := []float32{}
	MatvecQ4KInto(data, x, rows, cols, &fout) // float reference (flag off in tests)

	scratch := []float32{}
	xs := fillQ4KXSums(x, cols, &scratch)
	var q8s []int8
	var scs []float32
	q8, xsc := quantizeQ8Into(x, cols, &q8s, &scs)
	qout := make([]float32, rows)
	dotQ4KRowsQ8(data, q8, xsc, xs, cols, rowBytes, 0, rows, qout)

	cos, err := CosineSimilarity(fout, qout)
	if err != nil {
		t.Fatal(err)
	}
	if cos < 0.999 {
		t.Fatalf("int8 matvec cosine similarity %.5f < 0.999", cos)
	}

	// Sanity: relative error on the largest-magnitude outputs stays small.
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
				t.Fatalf("row %d: int8 %v vs float %v (rel %.4f)", i, qout[i], fout[i], rel)
			}
		}
	}
}
