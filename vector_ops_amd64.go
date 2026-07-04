//go:build amd64

package gopherllm

//go:noescape
func axpyF32AVX2(out []float32, alpha float32, x []float32)

//go:noescape
func scaleF32AVX2(out []float32, alpha float32)

//go:noescape
func scaleAddF32AVX2(out []float32, alpha float32, x []float32)

//go:noescape
func mulScaleF32AVX2(x []float32, weight []float32, scale float32, out []float32)

func axpyF32(out []float32, alpha float32, x []float32) {
	if hasAVX2 {
		axpyF32AVX2(out, alpha, x)
		return
	}
	axpyF32Scalar(out, alpha, x)
}

func scaleF32(out []float32, alpha float32) {
	if hasAVX2 {
		scaleF32AVX2(out, alpha)
		return
	}
	scaleF32Scalar(out, alpha)
}

func scaleAddF32(out []float32, alpha float32, x []float32) {
	if hasAVX2 {
		scaleAddF32AVX2(out, alpha, x)
		return
	}
	scaleAddF32Scalar(out, alpha, x)
}

func mulScaleF32(x []float32, weight []float32, scale float32, out []float32) {
	if hasAVX2 {
		mulScaleF32AVX2(x, weight, scale, out)
		return
	}
	mulScaleF32Scalar(x, weight, scale, out)
}
