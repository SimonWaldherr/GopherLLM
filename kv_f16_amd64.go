//go:build amd64

package gopherllm

import "os"

// useF16KVCache stores KV-cache rows as f16 where AVX2+F16C makes the
// on-the-fly conversion effectively free. Halves attention bandwidth and
// cache footprint; set GOPHERLLM_KV_F16=0 to keep the exact f32 cache
// (A/B testing, accuracy debugging).
var useF16KVCache = hasAVX2 && hasF16C && os.Getenv("GOPHERLLM_KV_F16") != "0"

//go:noescape
func dotF32F16AVX2(a []float32, b []uint16) float32

//go:noescape
func axpyF16AVX2(out []float32, alpha float32, x []uint16)

//go:noescape
func scaleAddF16AVX2(out []float32, alpha float32, x []uint16)

//go:noescape
func f32ToF16RowAVX2(dst []uint16, src []float32)

func dotF32F16(a []float32, b []uint16) float32 {
	if hasAVX2 && hasF16C {
		n8 := min(len(a), len(b)) &^ 7
		return dotF32F16AVX2(a, b) + dotF32F16Scalar(a, b, n8)
	}
	return dotF32F16Scalar(a, b, 0)
}

func axpyF16(out []float32, alpha float32, x []uint16) {
	if hasAVX2 && hasF16C {
		axpyF16AVX2(out, alpha, x)
		axpyF16Scalar(out, alpha, x, min(len(out), len(x))&^7)
		return
	}
	axpyF16Scalar(out, alpha, x, 0)
}

func scaleAddF16(out []float32, alpha float32, x []uint16) {
	if hasAVX2 && hasF16C {
		scaleAddF16AVX2(out, alpha, x)
		scaleAddF16Scalar(out, alpha, x, min(len(out), len(x))&^7)
		return
	}
	scaleAddF16Scalar(out, alpha, x, 0)
}

func f32ToF16Row(dst []uint16, src []float32) {
	if hasAVX2 && hasF16C {
		f32ToF16RowAVX2(dst, src)
		f32ToF16RowScalar(dst, src, min(len(dst), len(src))&^7)
		return
	}
	f32ToF16RowScalar(dst, src, 0)
}
