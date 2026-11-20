package engine

//go:noescape
func dotF32BF16NEON(x *float32, raw *byte, blocks int) float32

func bf16WeightSIMD() bool { return true }

func dotF32BF16(x []float32, raw []byte) float32 {
	var sum float32
	packed := len(x) &^ 15
	if packed != 0 {
		sum = dotF32BF16NEON(&x[0], &raw[0], packed/16)
	}
	return dotF32BF16Tail(sum, x[packed:], raw[packed*2:])
}
