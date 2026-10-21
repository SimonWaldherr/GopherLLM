package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"
)

func moePrefillFixture(gateType, downType GGMLType) *SparseMoEWeights {
	const dim, hidden, experts = 256, 256, 8
	return &SparseMoEWeights{
		Router:     Weight{F32: randomExpertInput(experts*dim, 39000), Rows: experts, Cols: dim},
		Gate:       moeDownWeightForTest(gateType, dim, hidden, experts),
		Up:         moeDownWeightForTest(gateType, dim, hidden, experts),
		Down:       moeDownWeightForTest(downType, hidden, dim, experts),
		RouterBias: randomExpertInput(experts, 39001),
		GateBias:   ExpertBias{Values: randomExpertInput(experts*hidden, 39002), Output: hidden, Experts: experts},
		UpBias:     ExpertBias{Values: randomExpertInput(experts*hidden, 39003), Output: hidden, Experts: experts},
		DownBias:   ExpertBias{Values: randomExpertInput(experts*dim, 39004), Output: dim, Experts: experts},
		ExpertUsed: 3, NormalizeTopK: true, Scale: 1.7,
	}
}

func requireMoEPrefillExact(t testing.TB, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("row %d got %.9g want %.9g", i, got[i], want[i])
		}
	}
}

func TestMoEPrefillPreservesSerialRoutingAndDots(t *testing.T) {
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	withQ8Activations(true, func() {
		for _, gateType := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
			for _, downType := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0, GGMLTypeMXFP4} {
				w := moePrefillFixture(gateType, downType)
				for _, p := range []int{1, 2, 4, 8, 17, 32} {
					t.Run(fmt.Sprintf("gate=%s/down=%s/p=%d", gateType, downType, p), func(t *testing.T) {
						if !moePrefillReady(w, p) {
							t.Fatal("supported layout declined")
						}
						var xf, of []float32
						var xs, outs [][]float32
						xs = reuseBatchViews(&xf, &xs, p, w.Gate.Input)
						outs = reuseBatchViews(&of, &outs, p, w.Down.Output)
						copy(xf, randomExpertInput(len(xf), 39005))
						got, want := &DecodeBuffer{}, &DecodeBuffer{}
						if err := sparseMoEPrefill(context.Background(), w, xs, outs, got); err != nil {
							t.Fatal(err)
						}
						for i, x := range xs {
							sparseMoEForward(w, x, want)
							requireMoEPrefillExact(t, outs[i], want.Proj)
						}
						if got.moePrefill.bytes() > moePrefillScratchBytes {
							t.Fatal("scratch budget exceeded")
						}
					})
				}
			}
		}
	})
}

func TestMoEPrefillGroupedRoutingSharedAndCancellation(t *testing.T) {
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	withQ8Activations(true, func() {
		w := moePrefillFixture(GGMLTypeQ4_K, GGMLTypeQ6_K)
		w.RoutingSigmoid, w.GroupCount, w.GroupUsed = true, 2, 1
		w.RouterCorrectionBias = randomExpertInput(w.Gate.Experts, 39006)
		w.SharedGate = &Weight{F32: randomExpertInput(256*32, 39007), Rows: 32, Cols: 256}
		w.SharedUp = &Weight{F32: randomExpertInput(256*32, 39008), Rows: 32, Cols: 256}
		w.SharedDown = &Weight{F32: randomExpertInput(32*256, 39009), Rows: 256, Cols: 32}
		w.SharedGateIn = &Weight{F32: randomExpertInput(256, 39010), Rows: 1, Cols: 256}
		xs, outs := make([][]float32, 8), make([][]float32, 8)
		for i := range xs {
			xs[i], outs[i] = randomExpertInput(256, int64(39011+i)), make([]float32, 256)
		}
		buf := &DecodeBuffer{}
		for _, oai := range []bool{false, true} {
			w.OAIActivation = oai
			for _, always := range []bool{false, true} {
				w.SharedAlways = always
				if err := sparseMoEPrefill(context.Background(), w, xs, outs, buf); err != nil {
					t.Fatal(err)
				}
				for i, x := range xs {
					want := &DecodeBuffer{}
					sparseMoEForward(w, x, want)
					requireMoEPrefillExact(t, outs[i], want.Proj)
				}
			}
		}
		before := slices.Clone(outs[0])
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := sparseMoEPrefill(ctx, w, xs, outs, buf); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation: %v", err)
		}
		requireMoEPrefillExact(t, outs[0], before)
	})
}

func TestQwen35MoEPrefillMatchesSerialContinuation(t *testing.T) {
	t.Setenv("GOPHERLLM_MOE_PREFILL", "1")
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	t.Setenv("GOPHERLLM_NO_MOE_PREFILL", "")
	withQ8Activations(true, func() {
		cfg, weights := qwen35BatchFixture(true)
		cfg.Arch, cfg.ExpertCount, cfg.ExpertUsedCount = "qwen35moe", 8, 3
		for i := range weights.Layers {
			weights.Layers[i].FFN = Qwen35FFNWeights{MoE: moePrefillFixture(GGMLTypeQ4_K, GGMLTypeQ6_K)}
		}
		for _, p := range []int{1, 4, 17, 32} {
			t.Run(fmt.Sprintf("p=%d", p), func(t *testing.T) {
				gotCache, wantCache := qwen35BatchCache(cfg, weights, kvF32), qwen35BatchCache(cfg, weights, kvF32)
				if !qwen35MetalBatchEligible(cfg, weights, gotCache) {
					t.Fatal("MoE batch declined")
				}
				got, want := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
				// Resume a nonempty recurrent/attention cache.
				for i := range 3 {
					ForwardQwen35BodyInto(cfg, weights, gotCache, got, uint32(i+1), i)
					ForwardQwen35BodyInto(cfg, weights, wantCache, want, uint32(i+1), i)
				}
				tokens := make([]uint32, p)
				for i := range tokens {
					tokens[i] = uint32((i*7 + 5) % cfg.VocabSize)
					ForwardQwen35BodyInto(cfg, weights, wantCache, want, tokens[i], 3+i)
				}
				if _, err := forwardQwen35MetalBatchStatesInto(context.Background(), cfg, weights, gotCache, got, tokens, 3); err != nil {
					t.Fatal(err)
				}
				requireMoEPrefillExact(t, got.XN, want.XN)
				requireQwen35BatchCache(t, gotCache, wantCache, 0, false)
				for i := range 4 {
					var a, b []float32
					ForwardQwen35Into(cfg, weights, gotCache, got, uint32(i+2), p+3+i, &a)
					ForwardQwen35Into(cfg, weights, wantCache, want, uint32(i+2), p+3+i, &b)
					requireMoEPrefillExact(t, a, b)
				}
			})
		}
	})
}

func TestMoEPrefillAdmissionAndRetainedScratch(t *testing.T) {
	withQ8Activations(true, func() {
		w := moePrefillFixture(GGMLTypeQ4_K, GGMLTypeQ6_K)
		for _, p := range []int{0, 33} {
			if moePrefillReady(w, p) {
				t.Fatalf("invalid token count %d admitted", p)
			}
		}
		for _, change := range []func(*SparseMoEWeights){
			func(w *SparseMoEWeights) { w.ExpertUsed = 33 },
			func(w *SparseMoEWeights) { w.Gate.Weight.Raw = nil },
			func(w *SparseMoEWeights) { w.Gate.RowOffset = 1 },
			func(w *SparseMoEWeights) { w.Down.Weight.Type = GGMLTypeF16 },
			func(w *SparseMoEWeights) { w.Up.Input++ },
			func(w *SparseMoEWeights) { w.Gate.Input = 1 << 30 },
		} {
			candidate := *w
			change(&candidate)
			if moePrefillReady(&candidate, 32) {
				t.Fatal("unsupported layout admitted")
			}
		}
		var scratch moePrefillScratch
		for _, dims := range [][2]int{{4096, 256}, {256, 4096}, {4096, 256}} {
			shape := &SparseMoEWeights{Gate: ExpertWeight{Input: dims[0], Output: dims[1], Experts: 8}, ExpertUsed: 8}
			scratch.prepare(shape, 24)
			if scratch.bytes() > moePrefillScratchBytes {
				t.Fatalf("retained %d bytes", scratch.bytes())
			}
		}
		shape := &SparseMoEWeights{Gate: ExpertWeight{Input: 256, Output: 256, Experts: 8}, ExpertUsed: 3}
		scratch = moePrefillScratch{}
		scratch.prepare(shape, 32)
		shape.Gate.Output = 16384
		scratch.prepare(shape, 1)
		for _, rows := range [][][]float32{scratch.gateRows, scratch.upRows, scratch.downRows} {
			for _, spare := range rows[len(rows):cap(rows)] {
				if spare != nil {
					t.Fatal("short wider chunk retained an old activation slab through spare row views")
				}
			}
		}
	})
}

func TestQwen35MoEPrefillEligibility(t *testing.T) {
	t.Setenv("GOPHERLLM_MOE_PREFILL", "")
	t.Setenv("GOPHERLLM_NO_MOE_PREFILL", "")
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "")
	withQ8Activations(true, func() {
		cfg, weights := qwen35BatchFixture(true)
		cfg.Arch = "qwen35moe"
		for i := range weights.Layers {
			weights.Layers[i].FFN = Qwen35FFNWeights{MoE: moePrefillFixture(GGMLTypeQ4_K, GGMLTypeQ6_K)}
		}
		r := &Runner{kind: loadedQwen35, config: cfg, qwen35: weights}
		cache := qwen35BatchCache(cfg, weights, kvF32)
		if r.canBatchQwen35Metal(cache) {
			t.Fatal("experimental MoE schedule enabled by default")
		}
		t.Setenv("GOPHERLLM_MOE_PREFILL", "1")
		if !r.canBatchQwen35Metal(cache) || r.canVerifyBlock(cache) {
			t.Fatal("MoE prefill and speculative verification admission differ from their contracts")
		}
		for _, format := range []kvFormat{kvF16, kvI8} {
			if r.canBatchQwen35Metal(qwen35BatchCache(cfg, weights, format)) {
				t.Fatal("compressed KV admitted")
			}
		}
		r.outOfCore = true
		if r.canBatchQwen35Metal(cache) {
			t.Fatal("out-of-core admitted")
		}
		r.outOfCore = false
		useQ8Activations.Store(false)
		if r.canBatchQwen35Metal(cache) {
			t.Fatal("changed float-activation arithmetic admitted")
		}
		useQ8Activations.Store(true)
		t.Setenv("GOPHERLLM_NO_MOE_PREFILL", "1")
		if r.canBatchQwen35Metal(cache) {
			t.Fatal("prefill opt-out ignored")
		}
	})
}
