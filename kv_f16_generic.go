//go:build !amd64 && !(darwin && arm64)

package gopherllm

import "os"

// The f16 KV cache remains opt-in on platforms without the amd64 F16C or
// Apple Silicon NEON implementations. Scalar conversion is slower, but
// `GOPHERLLM_KV_F16=1` still halves KV-cache memory. The default stays exact
// f32 and the scalar implementations below keep every target correct.
var useF16KVCache = newAtomicBool(os.Getenv("GOPHERLLM_KV_F16") == "1")

func dotF32F16(a []float32, b []uint16) float32 { return dotF32F16Scalar(a, b, 0) }

func axpyF16(out []float32, alpha float32, x []uint16) { axpyF16Scalar(out, alpha, x, 0) }

func scaleAddF16(out []float32, alpha float32, x []uint16) { scaleAddF16Scalar(out, alpha, x, 0) }

func f32ToF16Row(dst []uint16, src []float32) { f32ToF16RowScalar(dst, src, 0) }
