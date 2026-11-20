//go:build !arm64 && !amd64

package engine

func littleBitDot(packed []byte, x []float32) float32 {
	return littleBitDotScalar(packed, x)
}
