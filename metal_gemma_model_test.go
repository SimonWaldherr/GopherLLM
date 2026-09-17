//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"io"
	"math"
	"os"
	"testing"
)

func TestMetalGemmaModelParity(t *testing.T) {
	path := os.Getenv("GOPHERLLM_GEMMA_MODEL")
	if path == "" {
		t.Skip("explicit checkpoint required")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_GEMMA", "1")
	t.Setenv("GOPHERLLM_PREFILL_CHUNK", "16")
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if r.kind != loadedGemma4 {
		t.Fatal("Gemma checkpoint required")
	}
	c := r.config
	cpu := r.gemma4
	if cpu.Native {
		cpu.Layers = append([]Gemma4LayerWeights(nil), cpu.Layers...)
		for i := range cpu.Layers {
			l := &cpu.Layers[i]
			for _, w := range []*Weight{&l.AttnQ, &l.AttnK, &l.AttnV, &l.AttnOutput, &l.FFNGate, &l.FFNUp, &l.FFNDown, &l.PerLayerInputGate, &l.PerLayerProj} {
				w.Metal = nil
			}
		}
		if cpu.PerLayer != nil {
			p := *cpu.PerLayer
			p.ModelProj.Metal = nil
			cpu.PerLayer = &p
		}
		cpu.Output.Metal = nil
	} else {
		cpu.Standard.Layers = append([]LayerWeights(nil), cpu.Standard.Layers...)
		for i := range cpu.Standard.Layers {
			l := &cpu.Standard.Layers[i]
			for _, w := range []*Weight{&l.WQ, &l.WK, &l.WV, &l.WO, &l.W1, &l.W2, &l.W3} {
				w.Metal = nil
			}
		}
		cpu.Standard.Output.Metal = nil
	}
	kd, vd, hd, kv, val := r.cacheDims()
	ck := NewKVCache(r.kvCacheLayerCount(), kd, vd, 64)
	gk := NewKVCache(r.kvCacheLayerCount(), kd, vd, 64)
	cb := NewDecodeBuffer(c, hd, kv, val)
	gb := NewDecodeBuffer(c, hd, kv, val)
	tokens := make([]uint32, 37)
	state := uint32(1)
	for i := range tokens {
		state = state*1664525 + 1013904223
		tokens[i] = state % uint32(c.VocabSize)
	}
	check := func(label string, want, got []float32) {
		t.Helper()
		if len(want) != len(got) {
			t.Fatal("length mismatch")
		}
		var e, n float64
		for i, v := range want {
			if !finite32(got[i]) {
				t.Fatal("nonfinite output")
			}
			d := float64(got[i] - v)
			e += d * d
			n += float64(v) * float64(v)
		}
		rel := math.Sqrt(e / math.Max(n, 1e-30))
		t.Logf("%s relative L2=%g", label, rel)
		if rel > 1e-4 {
			t.Fatalf("%s error %g", label, rel)
		}
	}
	var want, got []float32
	for i := 0; i < 35; i++ {
		ForwardGemma4Into(c, cpu, ck, cb, tokens[i], i, &want)
	}
	if cpu.Native && !r.canBatchNativeGemma4() {
		t.Fatal("Metal prefill was not activated")
	}
	if err := r.prefillBatchedAt(context.Background(), gk, gb, tokens[:35], 0, &got); err != nil {
		t.Fatal(err)
	}
	check("prefill", want, got)
	for i := 35; i < 37; i++ {
		ForwardGemma4Into(c, cpu, ck, cb, tokens[i], i, &want)
		ForwardGemma4Into(c, r.gemma4, gk, gb, tokens[i], i, &got)
		check("decode", want, got)
	}
	for l := range ck.K {
		check("K", ck.K[l][:37*kd], gk.K[l][:37*kd])
		check("V", ck.V[l][:37*vd], gk.V[l][:37*vd])
	}
}
