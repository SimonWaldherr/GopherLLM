//go:build !arm64 && !amd64

package gopherllm

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
