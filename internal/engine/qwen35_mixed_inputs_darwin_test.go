//go:build darwin && cgo && metal

package engine

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
)

// Keep the 27B input and FFN geometry and the real offload thresholds. The
// quantized helper repeats 32 encoded rows, avoiding a full F32 weight bank.
func qwen35MixedInputsFixture(t testing.TB) (Config, Qwen35Weights) {
	t.Helper()
	cfg := Config{Arch: "qwen35", Dim: 5120, HiddenDim: 17408, NLayers: 1, VocabSize: 32, MaxSeqLen: 64,
		NHeads: 24, NKVHeads: 4, HeadDim: 256, ValueDim: 256, KVDim: 1024, KVMul: 6,
		SSMInner: 6144, SSMHeads: 48, SSMGroups: 16, SSMState: 128, SSMConv: 4,
		RMSNormEps: 1e-6, RopeTheta: 10000, RopeDimensionCount: 64, LogitScale: 1}
	values := func(n int, seed int64) []float32 {
		v := randomExpertInput(n, seed)
		for i := range v {
			v[i] *= .004
		}
		return v
	}
	norm := func(n int, seed int64) []float32 {
		v := randomExpertInput(n, seed)
		for i := range v {
			v[i] = 1 + v[i]*.1
		}
		return v
	}
	f32 := func(rows, cols int, seed int64) Weight {
		return Weight{F32: values(rows*cols, seed), Rows: rows, Cols: cols}
	}
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	layer := Qwen35LayerWeights{Kind: qwen35DeltaNet, Norm: norm(cfg.Dim, 44100), PostAttnNorm: norm(cfg.Dim, 44101)}
	layer.DeltaNet = Qwen35DeltaNetWeights{
		QKVConv:    qwen35InputWeight(t, GGMLTypeQ6_K, channels, cfg.Dim),
		Gate:       qwen35InputWeight(t, GGMLTypeQ4_K, cfg.SSMInner, cfg.Dim),
		AlphaProj:  qwen35InputWeight(t, GGMLTypeQ4_K, cfg.SSMHeads, cfg.Dim),
		BetaProj:   qwen35InputWeight(t, GGMLTypeQ4_K, cfg.SSMHeads, cfg.Dim),
		Out:        qwen35InputWeight(t, GGMLTypeQ4_K, cfg.Dim, cfg.SSMInner),
		ConvKernel: f32(channels, cfg.SSMConv, 44102),
		A:          make([]float32, cfg.SSMHeads), DTBias: values(cfg.SSMHeads, 44103), Norm: norm(cfg.SSMState, 44104),
	}
	for i := range layer.DeltaNet.A {
		layer.DeltaNet.A[i] = -.2 - float32(i)*.003
	}
	layer.FFN = Qwen35FFNWeights{
		Gate: qwen35InputWeight(t, GGMLTypeQ4_K, cfg.HiddenDim, cfg.Dim),
		Up:   qwen35InputWeight(t, GGMLTypeQ4_K, cfg.HiddenDim, cfg.Dim),
		Down: qwen35InputWeight(t, GGMLTypeQ6_K, cfg.Dim, cfg.HiddenDim),
	}
	w := Qwen35Weights{Layers: []Qwen35LayerWeights{layer}, TokenEmbd: f32(cfg.VocabSize, cfg.Dim, 44105),
		OutputNorm: norm(cfg.Dim, 44106), Output: f32(cfg.VocabSize, cfg.Dim, 44107)}
	t.Cleanup(func() { releaseQwen35MetalWeights(&w) })
	for _, weight := range []*Weight{&w.Layers[0].DeltaNet.QKVConv, &w.Layers[0].FFN.Gate, &w.Layers[0].FFN.Up, &w.Layers[0].FFN.Down} {
		weight.Metal = prepareMetalWeight(weight.Raw, weight.Type, weight.Rows, weight.Cols, false)
		if weight.Metal == nil {
			t.Fatalf("prepare %s %dx%d: %s", weight.Type, weight.Rows, weight.Cols, MetalError())
		}
	}
	if !metalWeightUsesDirect(w.Layers[0].DeltaNet.QKVConv.Metal) {
		t.Fatal("fixture must retain the direct Q6_K QKV projection")
	}
	return cfg, w
}

func TestQwen35MetalMixedInputsPreserveProductionStateBits(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", "1")
	setPerformanceThreads(t, 8)
	cfg, weights := qwen35MixedInputsFixture(t)
	checkBits := func(t *testing.T, name string, got, want []float32) {
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
	for _, q8 := range []bool{true, false} {
		t.Run(fmt.Sprintf("Q8=%t", q8), func(t *testing.T) {
			withQ8Activations(q8, func() {
				for _, p := range []int{2, 4, 17} {
					t.Run(fmt.Sprintf("positions=%d", p), func(t *testing.T) {
						cache, ref := qwen35BatchCache(cfg, weights, kvF32), qwen35BatchCache(cfg, weights, kvF32)
						if !qwen35MetalBatchEligible(cfg, weights, cache) {
							t.Fatal("production mixed fixture must enter the Metal batch schedule")
						}
						buf, rb := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
						const start = 2
						for pos, token := range []uint32{1, 4} {
							ForwardQwen35BodyInto(cfg, weights, cache, buf, token, pos)
							ForwardQwen35BodyInto(cfg, weights, ref, rb, token, pos)
						}
						checkpoint := cache.Qwen35.snapshot()
						tokens := make([]uint32, p)
						want := make([][]float32, p)
						var qkv, gate, alpha, beta []float32
						keep := p / 2
						var kept *Qwen35Cache
						// Probe the actual batch admission independently. Q8 opt-out
						// must leave gate and alpha/beta on token decode arithmetic.
						var inputs qwen35BatchScratch
						prepareQwen35InputRows(&inputs, cfg, p)
						for i := range tokens {
							tokens[i] = uint32((i*7 + 5) % cfg.VocabSize)
							weights.TokenEmbd.RowInto(int(tokens[i]), cfg.Dim, &inputs.XN[i])
							rmsNormInto(inputs.XN[i], weights.Layers[0].Norm, cfg.RMSNormEps, &inputs.XN[i])
						}
						haveQKV, haveGate, haveAB := qwen35MetalBatchDeltaInputs(weights.Layers[0].DeltaNet, inputs.XN, &inputs)
						if !haveQKV || haveGate != q8 || haveAB != q8 {
							t.Fatalf("unexpected input routes: QKV=%t gate=%t alpha/beta=%t Q8=%t", haveQKV, haveGate, haveAB, q8)
						}
						for i, token := range tokens {
							ForwardQwen35BodyInto(cfg, weights, ref, rb, token, start+i)
							want[i] = slices.Clone(rb.XN)
							qkv, gate = append(qkv, rb.MambaIn...), append(gate, rb.MambaZ...)
							alpha, beta = append(alpha, rb.MambaDT...), append(beta, rb.MambaBeta...)
							if i+1 == keep {
								kept = ref.Qwen35.snapshot()
							}
						}
						buf.speculative.active = p <= speculativeMaxPositions
						states, err := forwardQwen35MetalBatchStatesInto(context.Background(), cfg, weights, cache, buf, tokens, start)
						if err != nil || len(states) != p {
							t.Fatalf("states=%d error=%v", len(states), err)
						}
						for i := range states {
							checkBits(t, fmt.Sprintf("normalized hidden %d", i), states[i], want[i])
							if &states[i][0] != &buf.qwenBatch.XNFlat[i*cfg.Dim] {
								t.Fatal("normalized states must borrow the batch slab")
							}
						}
						checkBits(t, "convolution", cache.Qwen35.Conv, ref.Qwen35.Conv)
						checkBits(t, "recurrent state", cache.Qwen35.State, ref.Qwen35.State)
						replay := &buf.speculative.qwenReplay
						if replay.valid != buf.speculative.active {
							t.Fatal("replay availability does not match the speculative block limit")
						}
						if replay.valid {
							captured := replay.layers[0]
							checkBits(t, "replay QKV", captured.qkv, qkv)
							checkBits(t, "replay gate", captured.gate, gate)
							checkBits(t, "replay alpha", captured.alpha, alpha)
							checkBits(t, "replay beta", captured.beta, beta)
							if !cache.Qwen35.restore(checkpoint) {
								t.Fatal("cannot restore the rejected block checkpoint")
							}
							if err := replay.replay(context.Background(), cfg, weights, cache.Qwen35, buf, keep); err != nil {
								t.Fatal(err)
							}
							checkBits(t, "accepted prefix convolution", cache.Qwen35.Conv, kept.Conv)
							checkBits(t, "accepted prefix state", cache.Qwen35.State, kept.State)
							if !cache.Qwen35.restore(ref.Qwen35) {
								t.Fatal("cannot restore the complete verified state")
							}
						}
						var gotLogits, wantLogits []float32
						weights.Output.MatvecInto(states[p-1], &gotLogits)
						weights.Output.MatvecInto(want[p-1], &wantLogits)
						for step := range 3 {
							checkBits(t, "continuation logits", gotLogits, wantLogits)
							if slices.Min(gotLogits) == slices.Max(gotLogits) {
								t.Fatal("fixture must produce varying greedy logits")
							}
							token := argmaxFiniteToken(wantLogits)
							if argmaxFiniteToken(gotLogits) != token {
								t.Fatal("changed greedy continuation")
							}
							ForwardQwen35Into(cfg, weights, cache, buf, token, start+p+step, &gotLogits)
							ForwardQwen35Into(cfg, weights, ref, rb, token, start+p+step, &wantLogits)
							checkBits(t, "continued normalized state", buf.XN, rb.XN)
							checkBits(t, "continued convolution", cache.Qwen35.Conv, ref.Qwen35.Conv)
							checkBits(t, "continued recurrent state", cache.Qwen35.State, ref.Qwen35.State)
						}
					})
				}
			})
		})
	}
}
