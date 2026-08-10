//go:build !(darwin && arm64)

package gopherllm

import "unsafe"

// Keep the grouped f16 attention shape available on every target. Apple
// Silicon replaces these compositions with the NEON implementations above;
// other targets retain their existing per-head SIMD/scalar primitives.
func dotF32F16x4(q0, q1, q2, q3 *float32, x *uint16, n int) (s0, s1, s2, s3 float32) {
	if n <= 0 {
		return
	}
	xs := unsafe.Slice(x, n)
	s0 = dotF32F16(unsafe.Slice(q0, n), xs)
	s1 = dotF32F16(unsafe.Slice(q1, n), xs)
	s2 = dotF32F16(unsafe.Slice(q2, n), xs)
	s3 = dotF32F16(unsafe.Slice(q3, n), xs)
	return
}

func axpyF32F16x4(out0, out1, out2, out3 *float32, a0, a1, a2, a3 float32, x *uint16, n int) {
	if n <= 0 {
		return
	}
	xs := unsafe.Slice(x, n)
	axpyF16(unsafe.Slice(out0, n), a0, xs)
	axpyF16(unsafe.Slice(out1, n), a1, xs)
	axpyF16(unsafe.Slice(out2, n), a2, xs)
	axpyF16(unsafe.Slice(out3, n), a3, xs)
}
