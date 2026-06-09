//go:build !arm64

package main

func axpyF32(out []float32, alpha float32, x []float32) {
	axpyF32Scalar(out, alpha, x)
}

func scaleF32(out []float32, alpha float32) {
	scaleF32Scalar(out, alpha)
}

func scaleAddF32(out []float32, alpha float32, x []float32) {
	scaleAddF32Scalar(out, alpha, x)
}
