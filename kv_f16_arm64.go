//go:build arm64

package gopherllm

import "os"

// Apple Silicon implements the ARMv8.2 FP16 vector-conversion instructions.
// Keep the compact cache opt-in by default: the NEON kernels make it much
// faster than the portable path, but f32 is still quicker on some Apple CPUs.
// The autotuner may enable it when an end-to-end probe finds it beneficial.
var useF16KVCache = newAtomicBool(os.Getenv("GOPHERLLM_KV_F16") == "1")

//go:noescape
func dotF32F16NEON(a []float32, b []uint16) float32

//go:noescape
func axpyF16NEON(out []float32, alpha float32, x []uint16)

//go:noescape
func scaleAddF16NEON(out []float32, alpha float32, x []uint16)

//go:noescape
func f32ToF16RowNEON(dst []uint16, src []float32)

func dotF32F16(a []float32, b []uint16) float32 {
	n8 := min(len(a), len(b)) &^ 7
	return dotF32F16NEON(a[:n8], b[:n8]) + dotF32F16Scalar(a, b, n8)
}

func axpyF16(out []float32, alpha float32, x []uint16) {
	n8 := min(len(out), len(x)) &^ 7
	axpyF16NEON(out[:n8], alpha, x[:n8])
	axpyF16Scalar(out, alpha, x, n8)
}

func scaleAddF16(out []float32, alpha float32, x []uint16) {
	n8 := min(len(out), len(x)) &^ 7
	scaleAddF16NEON(out[:n8], alpha, x[:n8])
	scaleAddF16Scalar(out, alpha, x, n8)
}

func f32ToF16Row(dst []uint16, src []float32) {
	n8 := min(len(dst), len(src)) &^ 7
	f32ToF16RowNEON(dst[:n8], src[:n8])
	f32ToF16RowScalar(dst, src, n8)
}
