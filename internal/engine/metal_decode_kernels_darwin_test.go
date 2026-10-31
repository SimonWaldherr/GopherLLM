//go:build darwin && cgo && metal

package engine

import (
	"fmt"
	"math"
	"testing"
)

// Split sums change rounding. Use the common GPU/CPU numerical bound for
// complete decoder states rather than requiring the previous FP32 additions.
func TestMetalDenseQ4SplitSums(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_DENSE_VECTOR", "1")
	t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_GATE", "1")
	for _, dim := range []int{256, 768, 3072} {
		for _, arch := range []string{"ministral", "qwen3"} {
			t.Run(fmt.Sprintf("%s/%d", arch, dim), func(t *testing.T) {
				c, _, w := metalResidualNormFixture(t, arch, dim)
				run := func(mode string) []float32 {
					t.Setenv("GOPHERLLM_METAL_DENSE_Q4_SPLIT_SUM", mode)
					cache, buf := metalPrefillBuffer(t, c, 32)
					var state, logits []float32
					for _, pos := range []int{0, 1, 2, 7, 8, 17, 9, 18} {
						if !tryMetalDenseDecodeOutput(c, w, cache, buf, uint32(pos%16), pos, &logits) {
							t.Fatal("dense decoder inactive:", MetalError())
						}
						state = append(state, logits...)
						state = append(state, buf.X...)
						state = append(state, buf.XN...)
						for i := range cache.K {
							state = append(state, cache.K[i][:((pos+1)*cache.PerPosKDim)]...)
							state = append(state, cache.V[i][:((pos+1)*cache.PerPosVDim)]...)
						}
					}
					return state
				}
				want, got := run("0"), run("1")
				assertMetalFiniteClose(t, got, want)
				var maxScaledError float64
				for i, v := range want {
					scaledError := math.Abs(float64(got[i]-v)) / math.Max(1, math.Abs(float64(v)))
					maxScaledError = math.Max(maxScaledError, scaledError)
				}
				t.Logf("maximum error / max(1, |reference|): %.8g", maxScaledError)
				t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_GATE", "0")
				metalDenseMemoryExact(t, "separate/fused Q4 gate", run("1"), got)
			})
		}
	}
}

// Q6 fusion keeps each projection's addition order, including across streaming
// compaction. Check the residual, normalized vector and newly written KV bits.
func TestVoxtralMetalQ6FusedGateExact(t *testing.T) {
	cfg, w, state, rng := voxtralMetalDecoderFixture(t)
	t.Setenv("GOPHERLLM_METAL_DENSE_VECTOR", "1")
	reference := newVoxtralFastDecoder(cfg, w, state)
	optimized := newVoxtralFastDecoder(cfg, w, state)
	if reference == nil || optimized == nil {
		if reference != nil {
			reference.Close()
		}
		if optimized != nil {
			optimized.Close()
		}
		t.Fatal("Metal decoder inactive")
	}
	defer reference.Close()
	defer optimized.Close()
	a, b := reference.(*voxtralMetalDecoder), optimized.(*voxtralMetalDecoder)
	ka, va := make([]float32, 128), make([]float32, 128)
	kb, vb := make([]float32, 128), make([]float32, 128)
	for pos := 0; pos < a.capacity+9; pos++ {
		input := make([]float32, cfg.Decoder.HiddenSize)
		for i := range input {
			input[i] = float32(rng.NormFloat64()) * .2
		}
		t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_GATE", "0")
		want, err := a.Step(input, pos, true)
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_GATE", "1")
		got, err := b.Step(input, pos, true)
		if err != nil || got != want {
			t.Fatalf("position %d: token %d vs %d, error %v", pos, got, want, err)
		}
		metalDenseMemoryExact(t, "residual", b.residual, a.residual)
		metalDenseMemoryExact(t, "normalized", b.normed, a.normed)
		for layer := range w.Decoder.Layers {
			if !a.decoder.Cache(layer, a.physical-1, ka, va, false) || !b.decoder.Cache(layer, b.physical-1, kb, vb, false) {
				t.Fatal("KV readback failed")
			}
			metalDenseMemoryExact(t, "key", kb, ka)
			metalDenseMemoryExact(t, "value", vb, va)
		}
	}
}
