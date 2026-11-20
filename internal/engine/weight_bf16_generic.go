//go:build !amd64 && !arm64

package engine

func bf16WeightSIMD() bool { return false }

func dotF32BF16(x []float32, raw []byte) float32 { return dotF32BF16Scalar(x, raw) }
