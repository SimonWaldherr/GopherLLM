//go:build darwin && cgo && metal

package engine

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// BenchmarkQwen35MetalFFNConfigured isolates actual checkpoint FFNs. Loading
// retains every layer, but timed work selects only the first trunk FFN and
// the native MTP FFN; it is not a full-model generation benchmark.
func BenchmarkQwen35MetalFFNConfigured(b *testing.B) {
	path := os.Getenv("GOPHERLLM_SPEC_AB_RAW_MODEL")
	if path == "" {
		path = os.Getenv("GOPHERLLM_VALIDATE_MODEL")
	}
	if path == "" {
		b.Skip("set GOPHERLLM_SPEC_AB_RAW_MODEL to a native Qwen MTP checkpoint")
	}
	r, threads := specABLoad(b, path, true)
	if r.kind != loadedQwen35 || len(r.qwen35.Layers) == 0 || r.qwen35.MTP == nil {
		b.Fatal("a complete Qwen model with a native MTP head is required")
	}
	draft := qwen35MTPExecutionWeights(*r.qwen35.MTP)
	for _, target := range []struct {
		name string
		ffn  Qwen35FFNWeights
	}{
		{"trunk", r.qwen35.Layers[0].FFN}, {"mtp", draft.FFN},
	} {
		w := target.ffn
		if !metalQwen35FFNBatchReady(w.Gate.Metal, w.Up.Metal, w.Down.Metal) {
			b.Fatalf("%s FFN cannot execute the requested Metal batch", target.name)
		}
		bytes := int64(len(w.Gate.Raw) + len(w.Up.Raw) + len(w.Down.Raw))
		b.Logf("%s gate=%dx%d/%s up=%dx%d/%s down=%dx%d/%s packed_bytes=%d", target.name,
			w.Gate.Rows, w.Gate.Cols, w.Gate.Type, w.Up.Rows, w.Up.Cols, w.Up.Type, w.Down.Rows, w.Down.Cols, w.Down.Type, bytes)
		for _, n := range []int{2, 4} {
			input := make([]float32, n*w.Gate.Cols)
			for i := range input {
				input[i] = float32(math.Sin(float64(i)*.271) + .3*math.Cos(float64(i)*.037))
			}
			reference := make([]float32, n*w.Down.Rows)
			serial := func(out []float32) bool {
				for i := range n {
					row := out[i*w.Down.Rows : (i+1)*w.Down.Rows]
					if !matvecMetalQwen35SwiGLUInto(w.Gate.Metal, w.Up.Metal, w.Down.Metal, input[i*w.Gate.Cols:(i+1)*w.Gate.Cols], &row) {
						return false
					}
				}
				return true
			}
			if !serial(reference) {
				b.Fatal("serial Metal FFN declined")
			}
			for _, mode := range []struct {
				name, vector string
				serial       bool
			}{
				{"serial", "0", true}, {"batch", "0", false},
				{"serial_vector", "1", true}, {"batch_vector", "1", false},
			} {
				b.Run(fmt.Sprintf("%s/P%d/%s", target.name, n, mode.name), func(b *testing.B) {
					setPerformanceThreads(b, threads)
					b.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", mode.vector)
					out := make([]float32, len(reference))
					run := func() bool {
						if mode.serial {
							return serial(out)
						}
						return matvecMetalSwiGLUDecodeBatchInto(w.Gate.Metal, w.Up.Metal, w.Down.Metal, input, n, &out)
					}
					if !run() {
						b.Fatal("Metal FFN declined")
					}
					if relative := mtpHeadRelativeL2(b, out, reference); relative > 1e-5 {
						b.Fatalf("FFN float reduction error %g exceeds 1e-5", relative)
					}
					b.ReportAllocs()
					b.SetBytes(int64(n) * bytes)
					b.ResetTimer()
					for b.Loop() {
						if !run() {
							b.Fatal("Metal FFN failed")
						}
					}
					b.ReportMetric(float64(n), "tokens/op")
				})
			}
		}
	}
}
