//go:build !amd64

package gopherllm

// The f16 KV cache is currently enabled only on amd64 (F16C converts rows
// in-register); other platforms keep the exact f32 cache. The scalar
// implementations below keep the f16 code paths compilable and correct
// everywhere (they run only if a KVCache is explicitly built with F16).
const useF16KVCache = false

func dotF32F16(a []float32, b []uint16) float32 { return dotF32F16Scalar(a, b, 0) }

func axpyF16(out []float32, alpha float32, x []uint16) { axpyF16Scalar(out, alpha, x, 0) }

func scaleAddF16(out []float32, alpha float32, x []uint16) { scaleAddF16Scalar(out, alpha, x, 0) }

func f32ToF16Row(dst []uint16, src []float32) { f32ToF16RowScalar(dst, src, 0) }
