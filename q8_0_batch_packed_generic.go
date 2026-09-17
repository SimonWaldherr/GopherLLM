//go:build !arm64

package gopherllm

func prepareQ8_0PackedBatch(dst *[]int8, q8 []int8, cols, tokens int) bool {
	return false
}

func batchQ8_0PackedRows4(w Weight, outs [][]float32, q8, packed []int8, scales []float32, start, end int) bool {
	return false
}
