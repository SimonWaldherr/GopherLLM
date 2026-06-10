//go:build !arm64

package main

const hasQuantNEON = false

func q4kQDots8(q *byte, x *float32, qdots *float32) {
	panic("q4kQDots8 is only available on arm64")
}

func q6kQDots16(ql *byte, qh *byte, x *float32, qdots *float32) {
	panic("q6kQDots16 is only available on arm64")
}

func sumF32Groups32(x *float32, out *float32, groups int) {
	panic("sumF32Groups32 is only available on arm64")
}

func sumF32Groups16(x *float32, out *float32, groups int) {
	panic("sumF32Groups16 is only available on arm64")
}
