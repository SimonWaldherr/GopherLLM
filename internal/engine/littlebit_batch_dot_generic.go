//go:build !arm64

package engine

const hasFastLittleBitDot4 = false

func littleBitDot4(packed []byte, x0, x1, x2, x3 []float32) (float32, float32, float32, float32) {
	return littleBitDot(packed, x0), littleBitDot(packed, x1), littleBitDot(packed, x2), littleBitDot(packed, x3)
}
