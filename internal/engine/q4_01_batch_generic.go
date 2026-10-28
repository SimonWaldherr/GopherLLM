//go:build !arm64

package engine

func batchQ4_01Rows4(_ Weight, _ [][]float32, _ []int8, _, _ []float32, _, _ int) bool {
	return false
}
