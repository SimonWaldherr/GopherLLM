//go:build !arm64

package gopherllm

import "unsafe"

// dotF32x4/axpyF32x4 compose the shared-row grouped-GQA algorithm out of the
// portable DotF32/AxpyF32 primitives instead of a hand-written 4-wide kernel
// (arm64 has one in attention_gqa_arm64.go/.s). On amd64 DotF32/AxpyF32 are
// themselves AVX2-accelerated, so this still gets full SIMD throughput per
// call; grouping's actual win — reading each shared K/V row from the KV cache
// once instead of once per query head — applies regardless of instruction
// set, so there is no hasFastGQA4 gate here: the call sites gate on kvMul==4
// alone.
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

// hasFastDotF32x4 is false here: dotF32x4 above is a composition, not a kernel.
// Measured on amd64, routing matvecBatch's token loop through it cost roughly
// 1.4x on the vision tower versus calling DotF32 per token directly — the
// shared-row reuse that makes it worthwhile on arm64 needs the row to stay in
// registers, which only the hand-written kernel achieves.
const hasFastDotF32x4 = false
