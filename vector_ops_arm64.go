//go:build arm64

package gopherllm

func axpyF32(out []float32, alpha float32, x []float32)
func scaleF32(out []float32, alpha float32)
func scaleAddF32(out []float32, alpha float32, x []float32)
func mulScaleF32(x []float32, weight []float32, scale float32, out []float32)

// siluMulF32 computes out[i] = silu(gate[i]) * up[i] (SwiGLU). Scalar on
// arm64 (the decode budget there is dominated by the NEON matvec kernels).
func siluMulF32(gate, up, out []float32) {
	siluMulF32Scalar(gate, up, out, 0, min(len(gate), len(up), len(out)))
}
