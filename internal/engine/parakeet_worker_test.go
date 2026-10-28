package engine

import (
	"math"
	"testing"
)

// The original batched graph keeps projection arithmetic identical while
// checking parallel activation and output-buffer reuse bit for bit.
func referenceParakeetFeedForwardBatch(xs [][]float32, normW, normB []float32, lin1, lin2 Weight, eps float32) [][]float32 {
	t := len(xs)
	result := make([][]float32, t)
	if t == 0 {
		return result
	}
	dModel := len(xs[0])
	normed := audioMatrixRows(t, dModel)
	for i := range xs {
		layerNormInto(xs[i], normW, normB, eps, &normed[i])
	}
	hiddenDim := weightRows(lin1, dModel)
	hidden := audioMatrixRows(t, hiddenDim)
	blasMatvecBatch(lin1, normed, hidden)
	for i := range hidden {
		row := hidden[i]
		for j, v := range row {
			row[j] = siluExact(v)
		}
	}
	outDim := weightRows(lin2, hiddenDim)
	out := audioMatrixRows(t, outDim)
	blasMatvecBatch(lin2, hidden, out)
	for i := range result {
		row := out[i][:len(xs[i])]
		oi := out[i]
		for j := range row {
			row[j] = xs[i][j] + 0.5*oi[j]
		}
		result[i] = row
	}
	return result
}

func TestParakeetFeedForwardParallelBits(t *testing.T) {
	fill := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32((i*17)%71-35) * .01
		}
		return out
	}
	for _, workers := range []int{1, 3, 8} {
		setPerformanceThreads(t, workers)
		for _, n := range []int{0, 1, 15, 16, 33, 88} {
			for _, hidden := range []int{7, 513} {
				const dim = 96
				x := audioMatrixRows(n, dim)
				for i := range x {
					copy(x[i], fill(dim))
					x[i][0] += float32(i) * .01
				}
				w1, w2 := Weight{F32: fill(hidden * dim)}, Weight{F32: fill(dim * hidden)}
				normW, normB := fill(dim), fill(dim)
				want := referenceParakeetFeedForwardBatch(x, normW, normB, w1, w2, 1e-5)
				got := parakeetFeedForwardBatch(x, normW, normB, w1, w2, 1e-5)
				if len(got) != len(want) {
					t.Fatal("output frame count changed")
				}
				for i := range want {
					for j, value := range want[i] {
						if math.Float32bits(value) != math.Float32bits(got[i][j]) {
							t.Fatalf("workers%d frames%d hidden%d [%d][%d]: %g, want %g", workers, n, hidden, i, j, got[i][j], value)
						}
					}
				}
			}
		}
	}
}
