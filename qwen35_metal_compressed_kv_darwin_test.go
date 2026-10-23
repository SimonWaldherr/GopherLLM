//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
)

// Unlike the CPU prefill path, the Metal schedule retains token decode's
// projection arithmetic and chronological attention reads/writes. Compressed
// K/V must therefore preserve both hidden floats and every encoded cache bit.
func TestQwen35MetalCompressedKVPreservesDecodeBits(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", "1")
	setPerformanceThreads(t, 8)
	cfg, weights := qwen35MetalBatchFixture(t)
	checkValues := func(t *testing.T, name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length=%d, want %d", name, len(got), len(want))
		}
		for i, value := range got {
			if !finite32(value) || math.Float32bits(value) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]=%g (%08x), want %g (%08x)", name, i,
					value, math.Float32bits(value), want[i], math.Float32bits(want[i]))
			}
		}
	}
	checkCache := func(t *testing.T, got, want *KVCache, positions int) {
		t.Helper()
		checkValues(t, "convolution", got.Qwen35.Conv, want.Qwen35.Conv)
		checkValues(t, "recurrent state", got.Qwen35.State, want.Qwen35.State)
		for layer := range got.layerCount() {
			switch got.kvFormat() {
			case kvF16:
				k, v := positions*got.PerPosKDim, positions*got.PerPosVDim
				if !slices.Equal(got.K16[layer][:k], want.K16[layer][:k]) || !slices.Equal(got.V16[layer][:v], want.V16[layer][:v]) {
					t.Fatalf("F16 K/V bits differ at layer %d through position %d", layer, positions)
				}
			case kvI8:
				k, v := positions*q8RowBytes(got.PerPosKDim), positions*q8RowBytes(got.PerPosVDim)
				if !slices.Equal(got.K8[layer][:k], want.K8[layer][:k]) || !slices.Equal(got.V8[layer][:v], want.V8[layer][:v]) {
					t.Fatalf("I8 K/V bits differ at layer %d through position %d", layer, positions)
				}
			default:
				t.Fatal("test requires compressed K/V storage")
			}
		}
	}
	for _, format := range []kvFormat{kvF16, kvI8} {
		for _, q8 := range []bool{true, false} {
			t.Run(fmt.Sprintf("kv=%d/Q8=%t", format, q8), func(t *testing.T) {
				withQ8Activations(q8, func() {
					for _, p := range []int{2, 4, 17} {
						t.Run(fmt.Sprintf("positions=%d", p), func(t *testing.T) {
							cache, ref := qwen35BatchCache(cfg, weights, format), qwen35BatchCache(cfg, weights, format)
							if !qwen35MetalBatchEligible(cfg, weights, cache) {
								t.Fatal("dense Metal graph with compressed K/V did not enter the batch schedule")
							}
							buf, rb := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
							const start = 3
							for pos, token := range []uint32{1, 4, 3} {
								ForwardQwen35BodyInto(cfg, weights, cache, buf, token, pos)
								ForwardQwen35BodyInto(cfg, weights, ref, rb, token, pos)
							}
							checkpoint := cache.Qwen35.snapshot()
							tokens, want := make([]uint32, p), make([][]float32, p)
							keep := p / 2
							kept := qwen35BatchCache(cfg, weights, format)
							var keptHidden []float32
							for i := range tokens {
								tokens[i] = uint32((i*7 + 5) % cfg.VocabSize)
								ForwardQwen35BodyInto(cfg, weights, ref, rb, tokens[i], start+i)
								want[i] = slices.Clone(rb.XN)
								if i+1 == keep {
									if copyKVPrefix(kept, ref, start+keep) != start+keep || !kept.Qwen35.restore(ref.Qwen35) {
										t.Fatal("cannot retain accepted-prefix reference")
									}
									keptHidden = slices.Clone(rb.XN)
								}
							}
							buf.speculative.active = p <= speculativeMaxPositions
							states, err := forwardQwen35MetalBatchStatesInto(context.Background(), cfg, weights, cache, buf, tokens, start)
							if err != nil || len(states) != p {
								t.Fatalf("states=%d error=%v", len(states), err)
							}
							for i := range states {
								checkValues(t, fmt.Sprintf("normalized hidden position %d", i), states[i], want[i])
							}
							checkCache(t, cache, ref, start+p)
							var gotLogits, wantLogits []float32
							weights.Output.MatvecInto(states[p-1], &gotLogits)
							weights.Output.MatvecInto(want[p-1], &wantLogits)
							for step := range 3 {
								checkValues(t, "continuation logits", gotLogits, wantLogits)
								if slices.Min(wantLogits) == slices.Max(wantLogits) {
									t.Fatal("fixture must produce varying greedy logits")
								}
								token := argmaxFiniteToken(wantLogits)
								if argmaxFiniteToken(gotLogits) != token {
									t.Fatal("changed greedy continuation")
								}
								ForwardQwen35Into(cfg, weights, cache, buf, token, start+p+step, &gotLogits)
								ForwardQwen35Into(cfg, weights, ref, rb, token, start+p+step, &wantLogits)
								checkValues(t, "continued normalized hidden", buf.XN, rb.XN)
								checkCache(t, cache, ref, start+p+step+1)
							}
							replay := &buf.speculative.qwenReplay
							if replay.valid != buf.speculative.active {
								t.Fatal("replay availability differs from the speculative block limit")
							}
							if replay.valid {
								if !cache.Qwen35.restore(checkpoint) {
									t.Fatal("cannot restore pre-block recurrent checkpoint")
								}
								if err := replay.replay(context.Background(), cfg, weights, cache.Qwen35, buf, keep); err != nil {
									t.Fatal(err)
								}
								checkCache(t, cache, kept, start+keep)
								// Rejected K/V rows remain present in cache. Continuing
								// from the accepted boundary must overwrite those rows
								// and match a cache that never processed the suffix.
								kb := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
								weights.Output.MatvecInto(keptHidden, &wantLogits)
								for step := range 3 {
									token := argmaxFiniteToken(wantLogits)
									ForwardQwen35Into(cfg, weights, cache, buf, token, start+keep+step, &gotLogits)
									ForwardQwen35Into(cfg, weights, kept, kb, token, start+keep+step, &wantLogits)
									checkValues(t, "replayed continuation logits", gotLogits, wantLogits)
									checkValues(t, "replayed continuation hidden", buf.XN, kb.XN)
									checkCache(t, cache, kept, start+keep+step+1)
								}
							}
						})
					}
				})
			})
		}
	}
}
