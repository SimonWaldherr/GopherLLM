//go:build !amd64

package gopherllm

import "os"

// The f16 KV cache is enabled by default only on amd64 (F16C converts rows
// in-register). On other platforms it remains opt-in: scalar conversion is
// slower, but `GOPHERLLM_KV_F16=1` halves KV-cache memory, which is useful for
// long-context models on unified-memory Macs. The default stays exact f32.
// The scalar implementations below are correct everywhere.
var useF16KVCache = os.Getenv("GOPHERLLM_KV_F16") == "1"

func dotF32F16(a []float32, b []uint16) float32 { return dotF32F16Scalar(a, b, 0) }

func axpyF16(out []float32, alpha float32, x []uint16) { axpyF16Scalar(out, alpha, x, 0) }

func scaleAddF16(out []float32, alpha float32, x []uint16) { scaleAddF16Scalar(out, alpha, x, 0) }

func f32ToF16Row(dst []uint16, src []float32) { f32ToF16RowScalar(dst, src, 0) }
