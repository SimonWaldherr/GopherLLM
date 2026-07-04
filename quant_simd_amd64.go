//go:build amd64

package gopherllm

// hasQuantSIMD gates the AVX2 quantized dot-product fast paths on amd64. It
// mirrors hasAVX2 so callers fall back to the portable scalar kernels when the
// CPU lacks AVX2 + FMA.
var hasQuantSIMD = hasAVX2

// q4kQDots8 computes the 8 per-sub-block dot products (unsigned 4-bit quants
// times activations) of one Q4_K block. q points at the 128 packed nibble
// bytes, x at 256 floats and qdots at 8 floats. Implemented in
// quant_simd_amd64.s (AVX2).
//
//go:noescape
func q4kQDots8(q *byte, x *float32, qdots *float32)

// q4kQDots8Q8 computes the 8 Q4_K sub-block dot products against int8-quantized
// activations via VPMADDUBSW, writing 8 int32 results to intqdots. Used only by
// the opt-in int8 activation path (see q4k_q8_amd64.go).
//
//go:noescape
func q4kQDots8Q8(q *byte, q8 *int8, intqdots *int32)

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
