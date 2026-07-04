//go:build !arm64 && !amd64

package main

func dotF32(a, b []float32) float32 {
	return dotF32Scalar(a, b)
}
