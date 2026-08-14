//go:build arm64

package gopherllm

import "unsafe"

// The f16 GQA primitives mirror dotF32x4/axpyF32x4, but convert the shared
// K/V row only once. Mistral-family GQA uses four query heads for each KV head,
// so this avoids three of the four FCVTL conversions and K/V loads in the
// compact-cache attention path.
//
// These need no feature probe. Unlike the SDOT kernels, the only extension
// instructions here are FCVTL/FCVTL2 (half-to-single conversion), which are
// baseline ARMv8.0 NEON — mandatory on every arm64 part. The assembly they bind
// to has always lived in attention_gqa_arm64.s, which is built for bare arm64;
// only these Go declarations were gated to darwin, so non-Apple arm64 was
// compiling the kernels and then not calling them.

//go:noescape
func dotF32F16x4NEON(q0, q1, q2, q3 *float32, x *uint16, n int) (s0, s1, s2, s3 float32)

func dotF32F16x4(q0, q1, q2, q3 *float32, x *uint16, n int) (s0, s1, s2, s3 float32) {
	if n <= 0 {
		return
	}
	n8 := n &^ 7
	if n8 > 0 {
		s0, s1, s2, s3 = dotF32F16x4NEON(q0, q1, q2, q3, x, n8)
	}
	q0s, q1s := unsafe.Slice(q0, n), unsafe.Slice(q1, n)
	q2s, q3s := unsafe.Slice(q2, n), unsafe.Slice(q3, n)
	xs := unsafe.Slice(x, n)
	for i := n8; i < n; i++ {
		xv := F16ToF32(xs[i])
		s0 += q0s[i] * xv
		s1 += q1s[i] * xv
		s2 += q2s[i] * xv
		s3 += q3s[i] * xv
	}
	return
}

//go:noescape
func axpyF32F16x4NEON(out0, out1, out2, out3 *float32, a0, a1, a2, a3 float32, x *uint16, n int)

func axpyF32F16x4(out0, out1, out2, out3 *float32, a0, a1, a2, a3 float32, x *uint16, n int) {
	if n <= 0 {
		return
	}
	n8 := n &^ 7
	if n8 > 0 {
		axpyF32F16x4NEON(out0, out1, out2, out3, a0, a1, a2, a3, x, n8)
	}
	out0s, out1s := unsafe.Slice(out0, n), unsafe.Slice(out1, n)
	out2s, out3s := unsafe.Slice(out2, n), unsafe.Slice(out3, n)
	xs := unsafe.Slice(x, n)
	for i := n8; i < n; i++ {
		xv := F16ToF32(xs[i])
		out0s[i] += a0 * xv
		out1s[i] += a1 * xv
		out2s[i] += a2 * xv
		out3s[i] += a3 * xv
	}
}
