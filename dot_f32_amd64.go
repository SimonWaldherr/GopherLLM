//go:build amd64

package main

// dotF32AVX2 computes the dot product of the overlapping prefix of a and b
// using AVX2 + FMA. Implemented in dot_f32_amd64.s.
func dotF32AVX2(a, b []float32) float32

func dotF32(a, b []float32) float32 {
	if hasAVX2 {
		return dotF32AVX2(a, b)
	}
	return dotF32Scalar(a, b)
}
