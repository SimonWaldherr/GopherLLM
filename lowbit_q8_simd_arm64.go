//go:build arm64

package gopherllm

// lowBitQ8SIMDEnabled reports whether the Q2_K/Q3_K per-matrix Q8 path uses
// the SDOT kernels.  An arm64 CPU without FEAT_DotProd can still opt into the
// portable Q8 path, but its scalar dots do not justify discarding float
// multi-matrix fusion.
func lowBitQ8SIMDEnabled() bool {
	return useQ8Activations.Load() && hasQ8KDotAsm
}
