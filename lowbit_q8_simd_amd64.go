//go:build amd64

package gopherllm

// lowBitQ8SIMDEnabled reports whether the Q2_K/Q3_K per-matrix Q8 path uses
// the AVX2/F16C kernels rather than the generic float-fused traversal.  Keep
// this separate from useQ8Activations: the latter is a runtime policy toggle,
// while this function is the hardware capability needed to make splitting a
// fused projection worthwhile.
func lowBitQ8SIMDEnabled() bool {
	return useQ8Activations.Load() && hasAVX2 && hasF16C
}
