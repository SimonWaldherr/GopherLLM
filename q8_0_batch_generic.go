//go:build !arm64

package gopherllm

func batchQ8_0Rows4(w Weight, outs [][]float32, q8 []int8, scales []float32, start, end int) bool {
	return false
}
