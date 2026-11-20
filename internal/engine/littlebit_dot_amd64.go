package engine

//go:noescape
func littleBitDotAVX2(packed *byte, x *float32, groups int) float32

func littleBitDot(packed []byte, x []float32) float32 {
	groups := len(x) / 8
	if groups == 0 || !hasAVX2 {
		return littleBitDotScalar(packed, x)
	}
	sum := littleBitDotAVX2(&packed[0], &x[0], groups)
	return sum + littleBitDotScalar(packed[groups:], x[groups*8:])
}
