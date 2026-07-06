//go:build amd64

package gopherllm

// hasQuantSIMD gates the AVX2 quantized dot-product fast paths on amd64. It
// mirrors hasAVX2 so callers fall back to the portable scalar kernels when the
// CPU lacks AVX2 + FMA.
var hasQuantSIMD = hasAVX2

const hasPreparedQ4K = false

// q4kQDots8 computes the 8 per-sub-block dot products (unsigned 4-bit quants
// times activations) of one Q4_K block. q points at the 128 packed nibble
// bytes, x at 256 floats and qdots at 8 floats. Implemented in
// quant_simd_amd64.s (AVX2).
//
//go:noescape
func q4kQDots8(q *byte, x *float32, qdots *float32)

func q4kDotPrepared(q *byte, x *float32, scales *float32, mins *float32, xsums *float32, blocks int) float32 {
	panic("q4kDotPrepared is only available on arm64")
}

// q8kQuantize quantizes x to int8 per 256-element block (symmetric absmax,
// one float scale per block — llama.cpp's Q8_K convention), used by
// acquireQ8 for the default int8-activation matvec path.
//
//go:noescape
func q8kQuantize(x *float32, q8 *int8, scales *float32, blocks int)

// q4kDotQ8KRow computes one full Q4_K row dot product against Q8K-quantized
// activations via VPMADDUBSW, with in-register scale/min decode and a single
// horizontal reduction per row. xsums are the per-32-element float sums of
// the original activations (exact dmin term).
//
//go:noescape
func q4kDotQ8KRow(q *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

// q6kDotQ8KRow is the Q6_K full-row analogue. xsums are the per-16-element
// sums of the original activations pre-scaled by 32.
//
//go:noescape
func q6kDotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

// q6kQDots16 computes the 16 per-16-element dot products (unsigned 6-bit
// quants, before the -32 offset, times activations) of one Q6_K block. ql
// points at 128 bytes, qh at 64 bytes, x at 256 floats and qdots at 16 floats.
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
