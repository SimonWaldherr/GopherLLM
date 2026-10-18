//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"testing"
)

// Optional checkpoint validation compares the complete submission with the
// established mixed CPU/GPU graph, then continues from its published Go KV.
func TestMetalWholePrefillModelParity(t *testing.T) {
	path := os.Getenv("GOPHERLLM_VALIDATE_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_VALIDATE_MODEL for checkpoint validation")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	forceExactMetalReference(t)
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	if r.kind != loadedStandard {
		t.Fatal("standard dense model required")
	}
	c := r.config
	if !metalDenseEligible(c, r.standard, NewKVCache(c.NLayers, c.KVDim, c.NKVHeads*c.ValueDim, 96)) {
		t.Fatal("checkpoint is not eligible")
	}
	ck, cb := metalPrefillBuffer(t, c, 96)
	gk, gb := metalPrefillBuffer(t, c, 96)
	var want, got []float32
	for _, chunk := range []struct{ start, n int }{{0, 32}, {32, 17}, {19, 17}} {
		tokens := make([]uint32, chunk.n)
		for i := range tokens {
			tokens[i] = uint32(i+chunk.start+3) % uint32(c.VocabSize)
		}
		t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "0")
		ForwardBatchInto(c, r.standard, ck, cb, tokens, chunk.start, true, &want)
		t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
		if err := forwardBatchIntoContext(context.Background(), c, r.standard, gk, gb, tokens, chunk.start, true, &got, nil); err != nil {
			t.Fatal(err)
		}
		if gb.metalDense == nil || gb.metalDense.decoder.PrefillScratchBytes() == 0 {
			t.Fatal("whole GPU path inactive")
		}
		var error2, norm2 float64
		for i, v := range want {
			if !finite32(got[i]) {
				t.Fatal("nonfinite logits")
			}
			d := float64(got[i] - v)
			error2 += d * d
			norm2 += float64(v) * float64(v)
		}
		relative := math.Sqrt(error2 / math.Max(norm2, 1e-30))
		t.Logf("start=%d n=%d relative logit L2=%g scratch=%d", chunk.start, chunk.n, relative, gb.metalDense.decoder.PrefillScratchBytes())
		if relative > .002 {
			t.Fatalf("logit mismatch %g", relative)
		}
		if argmaxFiniteToken(got) != argmaxFiniteToken(want) {
			t.Fatal("prefill greedy token differs")
		}
	}
	// The next CPU steps must see complete F32 cache rows from every GPU layer.
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
	token := argmaxFiniteToken(want)
	for pos := 36; pos < 40; pos++ {
		ForwardInto(c, cpu, ck, cb, token, pos, &want)
		ForwardInto(c, cpu, gk, gb, token, pos, &got)
		if argmaxFiniteToken(got) != argmaxFiniteToken(want) {
			t.Fatalf("CPU fallback greedy differs at %d", pos)
		}
		token = argmaxFiniteToken(want)
	}
}

func BenchmarkMetalWholePrefillModel(b *testing.B) {
	path := os.Getenv("GOPHERLLM_VALIDATE_MODEL")
	if path == "" {
		b.Skip("set GOPHERLLM_VALIDATE_MODEL")
	}
	if !MetalAvailable() {
		b.Fatal(MetalError())
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, LogWriter: io.Discard})
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	if r.kind != loadedStandard {
		b.Fatal("standard dense model required")
	}
	c := r.config
	for _, n := range []int{32, 128, 256} {
		for _, whole := range []string{"0", "1"} {
			b.Run(fmt.Sprintf("P%d/whole_%s", n, whole), func(b *testing.B) {
				b.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", whole)
				k, buf := metalPrefillBuffer(b, c, 512)
				tokens := make([]uint32, n)
				for i := range tokens {
					tokens[i] = uint32(i+3) % uint32(c.VocabSize)
				}
				ForwardBatchInto(c, r.standard, k, buf, tokens, 0, false, nil)
				if buf.metalDense == nil || (whole == "1" && buf.metalDense.decoder.PrefillScratchBytes() == 0) {
					b.Fatal("required GPU path inactive")
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					ForwardBatchInto(c, r.standard, k, buf, tokens, 0, false, nil)
				}
			})
		}
	}
}
