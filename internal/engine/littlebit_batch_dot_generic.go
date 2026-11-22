//go:build !arm64

package engine

const hasFastLittleBitDot4 = false

func littleBitDot4(packed []byte, x0, x1, x2, x3 []float32) (float32, float32, float32, float32) {
	return littleBitDot(packed, x0), littleBitDot(packed, x1), littleBitDot(packed, x2), littleBitDot(packed, x3)
}

const hasFastLittleBitDotRows4 = false

func littleBitDotRows4(packed0, packed1, packed2, packed3 []byte, x []float32) (float32, float32, float32, float32) {
	return littleBitDot(packed0, x), littleBitDot(packed1, x), littleBitDot(packed2, x), littleBitDot(packed3, x)
}
