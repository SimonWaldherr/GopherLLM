//go:build arm64

package gopherllm

func dotF32(a, b []float32) float32

const hasQuantSIMD = true
const hasPreparedQ4K = true

// q4kQDots8 computes the 8 per-sub-block dot products (unsigned 4-bit
// quants times activations) of one Q4_K block. q must point at the 128
// packed nibble bytes, x at 256 floats and qdots at 8 floats.
//
//go:noescape
func q4kQDots8(q *byte, x *float32, qdots *float32)

// q4kDotPrepared computes one prepared Q4_K row dot product. q must point at
// the first block's 128 packed nibble bytes, x at the row activations, and
// scales/mins/xsums at 8 floats per block.
//
//go:noescape
func q4kDotPrepared(q *byte, x *float32, scales *float32, mins *float32, xsums *float32, blocks int) float32

// q6kQDots16 computes the 16 per-sub-block dot products (unsigned 6-bit
// quants, before the -32 offset, times activations) of one Q6_K block.
// ql must point at 128 bytes, qh at 64 bytes, x at 256 floats and qdots at
// 16 floats.
//
//go:noescape
func q6kQDots16(ql *byte, qh *byte, x *float32, qdots *float32)

// sumF32Groups32 writes one sum per 32-float group from x into out.
//
//go:noescape
func sumF32Groups32(x *float32, out *float32, groups int)

// sumF32Groups16 writes one sum per 16-float group from x into out.
//
//go:noescape
func sumF32Groups16(x *float32, out *float32, groups int)

func axpyF32(out []float32, alpha float32, x []float32)
func scaleF32(out []float32, alpha float32)
func scaleAddF32(out []float32, alpha float32, x []float32)
func mulScaleF32(x []float32, weight []float32, scale float32, out []float32)

// siluMulF32 computes out[i] = silu(gate[i]) * up[i] (SwiGLU). Scalar on
// arm64 (the decode budget there is dominated by the NEON matvec kernels).
func siluMulF32(gate, up, out []float32) {
	siluMulF32Scalar(gate, up, out, 0, min(len(gate), len(up), len(out)))
}
