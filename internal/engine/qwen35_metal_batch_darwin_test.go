//go:build darwin && cgo && metal

package engine

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Real offload thresholds are retained: 3K model width and 8K FFNs. Reusing
// 32 distinct quantized row patterns bounds fixture construction cost while
// leaving nonzero, nonuniform attention, convolution and recurrent state.
func qwen35MetalBatchFixture(t testing.TB, ffnTypes ...[3]GGMLType) (Config, Qwen35Weights) {
	t.Helper()
	cfg := Config{Arch: "qwen35", Dim: 3072, HiddenDim: 8192, NLayers: 2, VocabSize: 32, MaxSeqLen: 96,
		NHeads: 4, NKVHeads: 2, HeadDim: 256, ValueDim: 256, KVDim: 512, KVMul: 2,
		SSMInner: 512, SSMHeads: 8, SSMGroups: 2, SSMState: 64, SSMConv: 3,
		RMSNormEps: 1e-5, RopeTheta: 10000, RopeDimensionCount: 64, FullAttentionInterval: 2, LogitScale: 1}
	rng := rand.New(rand.NewSource(35052026))
	values := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * .004
		}
		return v
	}
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = .8 + rng.Float32()*.3
		}
		return v
	}
	weight := func(rows, cols int, typ GGMLType) Weight {
		if typ == GGMLTypeF32 {
			return Weight{F32: values(rows * cols), Rows: rows, Cols: cols}
		}
		patterns := make([][]byte, min(rows, 32))
		for i := range patterns {
			v := values(cols)
			if typ == GGMLTypeQ4_K {
				patterns[i] = QuantizeRowQ4K(v, cols)
			} else if typ == GGMLTypeQ8_0 {
				patterns[i] = QuantizeRowQ8_0(v, cols)
			} else {
				patterns[i] = QuantizeRowQ6K(v, cols)
			}
		}
		stride := len(patterns[0])
		raw := make([]byte, rows*stride)
		for row := range rows {
			copy(raw[row*stride:], patterns[row%len(patterns)])
		}
		return Weight{Raw: raw, Rows: rows, Cols: cols, Type: typ}
	}
	prepare := func(w *Weight) {
		w.Metal = prepareMetalWeight(w.Raw, w.Type, w.Rows, w.Cols, false)
		if w.Metal == nil {
			t.Fatalf("prepare %v %dx%d: %s", w.Type, w.Rows, w.Cols, MetalError())
		}
	}
	w := Qwen35Weights{TokenEmbd: weight(cfg.VocabSize, cfg.Dim, GGMLTypeF32), OutputNorm: norm(cfg.Dim), Output: weight(cfg.VocabSize, cfg.Dim, GGMLTypeQ6_K)}
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	ffn := [3]GGMLType{GGMLTypeQ4_K, GGMLTypeQ4_K, GGMLTypeQ6_K}
	if len(ffnTypes) != 0 {
		ffn = ffnTypes[0]
	}
	for i := range cfg.NLayers {
		l := Qwen35LayerWeights{Norm: norm(cfg.Dim), PostAttnNorm: norm(cfg.Dim)}
		l.FFN = Qwen35FFNWeights{Gate: weight(cfg.HiddenDim, cfg.Dim, ffn[0]), Up: weight(cfg.HiddenDim, cfg.Dim, ffn[1]), Down: weight(cfg.Dim, cfg.HiddenDim, ffn[2])}
		prepare(&l.FFN.Gate)
		prepare(&l.FFN.Up)
		prepare(&l.FFN.Down)
		if i == 0 {
			l.Kind = qwen35DeltaNet
			l.DeltaNet = Qwen35DeltaNetWeights{QKVConv: weight(channels, cfg.Dim, GGMLTypeQ4_K), Gate: weight(cfg.SSMInner, cfg.Dim, GGMLTypeQ4_K),
				ConvKernel: weight(channels, cfg.SSMConv, GGMLTypeF32), AlphaProj: weight(cfg.SSMHeads, cfg.Dim, GGMLTypeF32), BetaProj: weight(cfg.SSMHeads, cfg.Dim, GGMLTypeF32),
				A: make([]float32, cfg.SSMHeads), DTBias: values(cfg.SSMHeads), Norm: norm(cfg.SSMState), Out: weight(cfg.Dim, cfg.SSMInner, GGMLTypeQ6_K)}
			for h := range l.DeltaNet.A {
				l.DeltaNet.A[h] = -.2 - rng.Float32()
			}
			prepare(&l.DeltaNet.Out)
		} else {
			l.Kind = qwen35Attention
			l.Attention = Qwen35AttentionWeights{Q: weight(2*cfg.NHeads*cfg.HeadDim, cfg.Dim, GGMLTypeQ4_K), K: weight(cfg.NKVHeads*cfg.HeadDim, cfg.Dim, GGMLTypeQ4_K),
				V: weight(cfg.NKVHeads*cfg.ValueDim, cfg.Dim, GGMLTypeQ6_K), O: weight(cfg.Dim, cfg.NHeads*cfg.ValueDim, GGMLTypeQ6_K), QNorm: norm(cfg.HeadDim), KNorm: norm(cfg.HeadDim)}
			prepare(&l.Attention.O)
		}
		w.Layers = append(w.Layers, l)
	}
	t.Cleanup(func() { releaseQwen35MetalWeights(&w) })
	return cfg, w
}

func TestQwen35MetalBatchPreservesOriginalMixerAndStates(t *testing.T) {
	t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", "0")
	testQwen35MetalBatchPreservesOriginalMixerAndStates(t)
}

func TestQwen35MetalVectorBatchPreservesOriginalMixerAndStates(t *testing.T) {
	t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", "1")
	testQwen35MetalBatchPreservesOriginalMixerAndStates(t)
}

func testQwen35MetalBatchPreservesOriginalMixerAndStates(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	withQ8Activations(true, func() {
		cfg, weights := qwen35MetalBatchFixture(t)
		for _, n := range []int{1, 2, 4, 8, 17} {
			t.Run(fmt.Sprint(n), func(t *testing.T) {
				cache, ref := qwen35BatchCache(cfg, weights, kvF32), qwen35BatchCache(cfg, weights, kvF32)
				buf, rb := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
				for pos, token := range []uint32{1, 4, 3} {
					ForwardQwen35BodyInto(cfg, weights, cache, buf, token, pos)
					ForwardQwen35BodyInto(cfg, weights, ref, rb, token, pos)
				}
				tokens := make([]uint32, n)
				want := make([][]float32, n)
				var qkv, gate, alpha, beta []float32
				for i := range tokens {
					tokens[i] = uint32((i*7 + 5) % cfg.VocabSize)
					ForwardQwen35BodyInto(cfg, weights, ref, rb, tokens[i], 3+i)
					want[i] = slices.Clone(rb.XN)
					qkv, gate = append(qkv, rb.MambaIn...), append(gate, rb.MambaZ...)
					alpha, beta = append(alpha, rb.MambaDT...), append(beta, rb.MambaBeta...)
				}
				buf.speculative.active = n <= speculativeMaxPositions
				states, err := forwardQwen35MetalBatchStatesInto(context.Background(), cfg, weights, cache, buf, tokens, 3)
				if err != nil || len(states) != n {
					t.Fatalf("states=%d error=%v", len(states), err)
				}
				for i := range states {
					requireQwen35BatchValues(t, fmt.Sprintf("hidden position %d", i), states[i], want[i])
					if &states[i][0] != &buf.qwenBatch.XNFlat[i*cfg.Dim] {
						t.Fatal("hidden state does not borrow the batch slab")
					}
					var gotLogits, wantLogits []float32
					weights.Output.MatvecInto(states[i], &gotLogits)
					weights.Output.MatvecInto(want[i], &wantLogits)
					if slices.Min(gotLogits) == slices.Max(gotLogits) || !finite32(slices.Min(gotLogits)) || !finite32(slices.Max(gotLogits)) {
						t.Fatal("fixture must produce finite, varying logits")
					}
					if argmaxFiniteToken(gotLogits) != argmaxFiniteToken(wantLogits) {
						t.Fatal("changed greedy prediction")
					}
				}
				requireQwen35BatchCache(t, cache, ref, 5e-5, false)
				if buf.speculative.qwenReplay.valid {
					l := buf.speculative.qwenReplay.layers[0]
					if !slices.Equal(l.qkv, qkv) || !slices.Equal(l.gate, gate) || !slices.Equal(l.alpha, alpha) || !slices.Equal(l.beta, beta) {
						t.Fatal("original first-layer projection arithmetic changed")
					}
				}
				for i, token := range []uint32{2, 17, 6, 9, 4, 1, 11, 20} {
					var got, want []float32
					ForwardQwen35Into(cfg, weights, cache, buf, token, 3+n+i, &got)
					ForwardQwen35Into(cfg, weights, ref, rb, token, 3+n+i, &want)
					if argmaxFiniteToken(got) != argmaxFiniteToken(want) {
						t.Fatal("changed continuation after the batch")
					}
				}
			})
		}
	})
}

func TestQwen35MetalBatchCPUOutPreservesExactStates(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	withQ8Activations(true, func() {
		cfg, weights := qwen35MetalBatchFixture(t)
		out := &weights.Layers[0].DeltaNet.Out
		var row []float32
		var raw []byte
		for i := range out.Rows {
			out.RowInto(i, out.Cols, &row)
			raw = append(raw, QuantizeRowQ4K(row, out.Cols)...)
		}
		releaseMetalWeight(out.Metal)
		out.Metal, out.Raw, out.Type = nil, raw, GGMLTypeQ4_K
		if !qwen35MetalBatchCPUOut(*out) {
			t.Fatal("Q4_K CPU output projection did not enter the batching path")
		}
		// A later recurrent layer catches any changed activation that crosses
		// the CPU-Q8 boundary after the first batched output projection.
		tail := weights.Layers[0]
		tail.RecurrentCacheSlot = 1
		weights.Layers = append(weights.Layers, tail)
		cfg.NLayers = len(weights.Layers)
		for _, n := range []int{1, 2, 4, 8, 17} {
			t.Run(fmt.Sprint(n), func(t *testing.T) {
				cache, ref := qwen35BatchCache(cfg, weights, kvF32), qwen35BatchCache(cfg, weights, kvF32)
				buf, rb := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
				for pos, token := range []uint32{1, 4, 3} {
					ForwardQwen35BodyInto(cfg, weights, cache, buf, token, pos)
					ForwardQwen35BodyInto(cfg, weights, ref, rb, token, pos)
				}
				tokens := make([]uint32, n)
				want := make([][]float32, n)
				for i := range tokens {
					tokens[i] = uint32((i*7 + 5) % cfg.VocabSize)
					ForwardQwen35BodyInto(cfg, weights, ref, rb, tokens[i], 3+i)
					want[i] = slices.Clone(rb.XN)
				}
				buf.speculative.active = n <= speculativeMaxPositions
				states, err := forwardQwen35MetalBatchStatesInto(context.Background(), cfg, weights, cache, buf, tokens, 3)
				if err != nil || len(states) != n {
					t.Fatalf("states=%d error=%v", len(states), err)
				}
				if n > 1 && len(buf.qwenBatch.Mixed) != n {
					t.Fatal("CPU output batching did not retain the independent recurrent output rows")
				}
				for i := range states {
					if !slices.Equal(states[i], want[i]) {
						t.Fatalf("CPU output batching changed hidden position %d", i)
					}
					var gotLogits, wantLogits []float32
					weights.Output.MatvecInto(states[i], &gotLogits)
					weights.Output.MatvecInto(want[i], &wantLogits)
					if slices.Min(gotLogits) == slices.Max(gotLogits) || argmaxFiniteToken(gotLogits) != argmaxFiniteToken(wantLogits) {
						t.Fatal("CPU output batching changed the varying greedy prediction")
					}
				}
				if !slices.Equal(cache.Qwen35.Conv, ref.Qwen35.Conv) || !slices.Equal(cache.Qwen35.State, ref.Qwen35.State) {
					t.Fatal("CPU output batching changed the chronological recurrent state")
				}
				for i, token := range []uint32{2, 17, 6} {
					ForwardQwen35BodyInto(cfg, weights, cache, buf, token, 3+n+i)
					ForwardQwen35BodyInto(cfg, weights, ref, rb, token, 3+n+i)
					if !slices.Equal(buf.XN, rb.XN) || !slices.Equal(cache.Qwen35.State, ref.Qwen35.State) {
						t.Fatal("CPU output batching changed the subsequent target boundary")
					}
				}
			})
		}
	})
}

func TestQwen35MetalBatchFFNReductionBound(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	cfg, weights := qwen35MetalBatchFixture(t)
	for _, n := range []int{2, 4, 8, 17} {
		input := make([]float32, n*cfg.Dim)
		for i := range input {
			input[i] = float32(math.Sin(float64(i)*.271) + .3*math.Cos(float64(i)*.037))
		}
		for layer, w := range weights.Layers {
			var got []float32
			if !matvecMetalSwiGLUBatchInto(w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal, input, n, &got) {
				t.Fatal("Metal FFN batch declined")
			}
			var maxError float64
			for i := range n {
				var want []float32
				if !matvecMetalSwiGLUInto(w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal, input[i*cfg.Dim:(i+1)*cfg.Dim], &want) {
					t.Fatal("Metal serial FFN declined")
				}
				for j, value := range want {
					maxError = max(maxError, math.Abs(float64(value-got[i*cfg.Dim+j])))
				}
				requireQwen35BatchValuesWithin(t, "FFN projection", got[i*cfg.Dim:(i+1)*cfg.Dim], want, 1e-6)
			}
			t.Logf("batch=%d layer=%d max FFN error=%.3g", n, layer, maxError)
		}
	}
}

func TestQwen35MetalFFNDecodeOrder(t *testing.T) {
	t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", "0")
	testQwen35MetalFFNDecodeOrder(t)
}

func TestQwen35MetalVectorFFNDecodeOrder(t *testing.T) {
	t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", "1")
	testQwen35MetalFFNDecodeOrder(t)
}

func testQwen35MetalFFNDecodeOrder(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	for _, types := range [][3]GGMLType{
		{GGMLTypeQ4_K, GGMLTypeQ4_K, GGMLTypeQ6_K},
		{GGMLTypeQ4_K, GGMLTypeQ4_K, GGMLTypeQ4_K},
		{GGMLTypeQ8_0, GGMLTypeQ8_0, GGMLTypeQ8_0},
	} {
		t.Run(fmt.Sprint(types), func(t *testing.T) {
			cfg, weights := qwen35MetalBatchFixture(t, types)
			for _, n := range []int{2, 3, 4, 8} {
				input := make([]float32, n*cfg.Dim)
				for i := range input {
					input[i] = float32(math.Sin(float64(i)*.271) + .3*math.Cos(float64(i)*.037))
				}
				for layer, w := range weights.Layers {
					var got []float32
					if !matvecMetalSwiGLUDecodeBatchInto(w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal, input, n, &got) {
						t.Fatal("Metal FFN decode batch declined")
					}
					for i := range n {
						var want []float32
						if !matvecMetalQwen35SwiGLUInto(w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal, input[i*cfg.Dim:(i+1)*cfg.Dim], &want) {
							t.Fatal("Metal serial FFN declined")
						}
						var legacy []float32
						if !matvecMetalSwiGLUInto(w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal, input[i*cfg.Dim:(i+1)*cfg.Dim], &legacy) {
							t.Fatal("legacy Metal serial FFN declined")
						}
						if relative := mtpHeadRelativeL2(t, want, legacy); relative > 1e-5 {
							t.Fatalf("FFN float reduction error %g exceeds 1e-5", relative)
						}
						for j, value := range want {
							actual := got[i*cfg.Dim+j]
							if !finite32(actual) || math.Float32bits(value) != math.Float32bits(actual) {
								t.Fatalf("P%d layer %d position %d output[%d]: 0x%08x != decode 0x%08x", n, layer, i, j, math.Float32bits(actual), math.Float32bits(value))
							}
						}
					}
				}
			}
		})
	}
}

func TestQwen35MetalBatchEligibilityAndCancellation(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	cfg, weights := qwen35MetalBatchFixture(t)
	cache := qwen35BatchCache(cfg, weights, kvF32)
	r := &Runner{kind: loadedQwen35, config: cfg, qwen35: weights}
	if !r.canBatchQwen35Metal(cache) {
		t.Fatal("eligible Metal model declined")
	}
	for _, format := range []kvFormat{kvF16, kvI8} {
		if !r.canBatchQwen35Metal(qwen35BatchCache(cfg, weights, format)) {
			t.Fatal("dense compressed KV declined")
		}
	}
	for _, mutate := range []func(*Runner){
		func(r *Runner) { r.outOfCore = true },
		func(r *Runner) { r.config.Arch = "qwen35moe" },
		func(r *Runner) { r.qwen35.Layers[0].FFN.Gate.Metal = nil },
	} {
		candidate := Runner{kind: loadedQwen35, config: cfg, qwen35: weights}
		candidate.qwen35.Layers = slices.Clone(weights.Layers)
		mutate(&candidate)
		if candidate.canBatchQwen35Metal(cache) {
			t.Fatal("unsupported schedule accepted")
		}
	}
	buf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
	before := cache.Qwen35.snapshot()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := forwardQwen35MetalBatchStatesInto(ctx, cfg, weights, cache, buf, []uint32{1, 2}, 0); err != context.Canceled {
		t.Fatalf("cancellation=%v", err)
	}
	if !slices.Equal(before.Conv, cache.Qwen35.Conv) || !slices.Equal(before.State, cache.Qwen35.State) {
		t.Fatal("pre-canceled batch changed state")
	}
}
