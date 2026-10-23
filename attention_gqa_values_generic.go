//go:build !arm64

package gopherllm

func gqa4WeightedValuesF32(out, values, scores []float32, dim, stride, span, startT, rows int) bool {
	return false
}
