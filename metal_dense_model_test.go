//go:build darwin && cgo && metal

package gopherllm

import (
	"io"
	"math"
	"os"
	"testing"
)

// Optional correctness check using a real checkpoint, independent of timing.
func TestMetalDenseModelParity(t *testing.T) {
	path := os.Getenv("GOPHERLLM_VALIDATE_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_VALIDATE_MODEL for checkpoint validation")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	forceExactMetalReference(t)
	old := metalDenseDecodeEnabled
	metalDenseDecodeEnabled = true
	defer func() { metalDenseDecodeEnabled = old }()
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	c := r.config
	cpu := r.standard
	cpu.Layers = append([]LayerWeights(nil), cpu.Layers...)
	cpu.Output.Metal = nil
	for i := range cpu.Layers {
		l := &cpu.Layers[i]
		l.WQ.Metal = nil
		l.WK.Metal = nil
		l.WV.Metal = nil
		l.WO.Metal = nil
		l.W1.Metal = nil
		l.W3.Metal = nil
		l.W2.Metal = nil
	}
	ck := NewKVCache(c.NLayers, c.KVDim, c.NKVHeads*c.ValueDim, 64)
	gk := NewKVCache(c.NLayers, c.KVDim, c.NKVHeads*c.ValueDim, 64)
	cb := NewDecodeBuffer(c, c.HeadDim, c.NKVHeads, c.ValueDim)
	gb := NewDecodeBuffer(c, c.HeadDim, c.NKVHeads, c.ValueDim)
	defer func() {
		if gb.metalDense != nil {
			gb.metalDense.close()
		}
	}()
	// Exercise real checkpoint prefill, rewind and CPU/GPU cache equivalence.
	tokens := make([]uint32, 32)
	for i := range tokens {
		tokens[i] = uint32(i+3) % uint32(c.VocabSize)
	}
	var batchWant, batchGot []float32
	ForwardBatchInto(c, cpu, ck, cb, tokens, 0, true, &batchWant)
	ForwardBatchInto(c, r.standard, gk, gb, tokens, 0, true, &batchGot)
	var batchError, batchNorm float64
	for i, v := range batchWant {
		d := float64(batchGot[i] - v)
		batchError += d * d
		batchNorm += float64(v) * float64(v)
	}
	relativeBatch := math.Sqrt(batchError / math.Max(batchNorm, 1e-30))
	t.Logf("prefill relative logit L2 error %.6g", relativeBatch)
	if relativeBatch > 0.002 {
		t.Fatalf("prefill mismatch: %g", relativeBatch)
	}

	var want, got []float32
	state := uint32(1)
	for pos := 0; pos < 5; pos++ {
		state = state*1664525 + 1013904223
		token := state % uint32(c.VocabSize)
		ForwardInto(c, cpu, ck, cb, token, pos, &want)
		if !tryMetalDenseDecodeOutput(c, r.standard, gk, gb, token, pos, &got) {
			t.Fatal("dense path inactive")
		}
		finishLogits(c, r.standard, &got)
		var error2, norm2 float64
		for i, v := range want {
			if !finite32(got[i]) {
				t.Fatal("nonfinite output")
			}
			delta := float64(got[i] - v)
			error2 += delta * delta
			norm2 += float64(v) * float64(v)
		}
		relative := math.Sqrt(error2 / math.Max(norm2, 1e-30))
		t.Logf("position %d relative logit L2 error %.6g", pos, relative)
		if relative > 0.002 {
			t.Fatalf("logit error exceeds 0.2%%: %g", relative)
		}
	}
}
