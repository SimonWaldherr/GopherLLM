//go:build !amd64 && !(darwin && arm64)

package gopherllm

// Targets with no hand-written int8 SIMD: the Q8K row dots and the activation
// quantizer are the portable scalar kernels. See q8k_dot_darwin_arm64.go for
// the Apple Silicon counterpart, which routes the same three entry points
// through NEON SDOT.

// hasQ8KDotAsm reports whether this target has int8 SIMD behind the Q8K row
// dots. It also decides whether the int8-activation path is on by default (see
// defaultQ8Activations): without int8 SIMD the path is correct and selectable
// but not presumed faster than the vectorised float path, so it stays opt-in.
const hasQ8KDotAsm = false

func q8kQuantize(x []float32, q8 []int8, scales []float32, blocks int) {
	q8kQuantizePortable(x, q8, scales, blocks)
}

func q4kDotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	return q4kDotQ8KRowPortable(row, q8, xscales, xsums, blocks)
}

func q6kDotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	return q6kDotQ8KRowPortable(row, q8, xscales, xsums, blocks)
}
