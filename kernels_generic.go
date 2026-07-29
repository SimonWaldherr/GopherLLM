//go:build !arm64 && !amd64

package gopherllm

func dotF32(a, b []float32) float32 {
	return dotF32Scalar(a, b)
}

const hasQuantSIMD = false
const hasPreparedQ4K = false

func q4kQDots8(q *byte, x *float32, qdots *float32) {
	panic("q4kQDots8 requires arm64 (NEON) or amd64 (AVX2)")
}

func q4kDotPrepared(q *byte, x *float32, scales *float32, mins *float32, xsums *float32, blocks int) float32 {
	panic("q4kDotPrepared is only available on arm64")
}

func q6kQDots16(ql *byte, qh *byte, x *float32, qdots *float32) {
	panic("q6kQDots16 requires arm64 (NEON) or amd64 (AVX2)")
}

func sumF32Groups32(x *float32, out *float32, groups int) {
	panic("sumF32Groups32 requires arm64 (NEON) or amd64 (AVX2)")
}

func sumF32Groups16(x *float32, out *float32, groups int) {
	panic("sumF32Groups16 requires arm64 (NEON) or amd64 (AVX2)")
}

func axpyF32(out []float32, alpha float32, x []float32) {
	axpyF32Scalar(out, alpha, x)
}

func scaleF32(out []float32, alpha float32) {
	scaleF32Scalar(out, alpha)
}

func scaleAddF32(out []float32, alpha float32, x []float32) {
	scaleAddF32Scalar(out, alpha, x)
}

func mulScaleF32(x []float32, weight []float32, scale float32, out []float32) {
	mulScaleF32Scalar(x, weight, scale, out)
}

func siluMulF32(gate, up, out []float32) {
	siluMulF32Scalar(gate, up, out, 0, min(len(gate), len(up), len(out)))
}
