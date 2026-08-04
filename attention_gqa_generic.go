//go:build !arm64

package gopherllm

import "unsafe"

const hasFastGQA4 = false

func dotF32x4(q0, q1, q2, q3, x *float32, n int) (s0, s1, s2, s3 float32) {
	if n <= 0 {
		return
	}
	xs := unsafe.Slice(x, n)
	s0 = DotF32(unsafe.Slice(q0, n), xs)
	s1 = DotF32(unsafe.Slice(q1, n), xs)
	s2 = DotF32(unsafe.Slice(q2, n), xs)
	s3 = DotF32(unsafe.Slice(q3, n), xs)
	return
}

func axpyF32x4(out0, out1, out2, out3 *float32, a0, a1, a2, a3 float32, x *float32, n int) {
	if n <= 0 {
		return
	}
	xs := unsafe.Slice(x, n)
	AxpyF32(unsafe.Slice(out0, n), a0, xs)
	AxpyF32(unsafe.Slice(out1, n), a1, xs)
	AxpyF32(unsafe.Slice(out2, n), a2, xs)
	AxpyF32(unsafe.Slice(out3, n), a3, xs)
}
