package engine

//go:noescape
func dotF32BF16AVX2(x *float32, raw *byte, n int) float32

func bf16WeightSIMD() bool { return hasAVX2 }

func dotF32BF16(x []float32, raw []byte) float32 {
	if !hasAVX2 {
		return dotF32BF16Scalar(x, raw)
	}
	var sum float32
	packed := len(x) &^ 7
	if packed != 0 {
		sum = dotF32BF16AVX2(&x[0], &raw[0], packed)
	}
	return dotF32BF16Tail(sum, x[packed:], raw[packed*2:])
}
