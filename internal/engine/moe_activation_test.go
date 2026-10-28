package engine

import (
	"context"
	"fmt"
	"math"
	"testing"
)

func TestMoEActivationBatchPreservesRows(t *testing.T) {
	for _, width := range []int{8, 256, 512} {
		for _, n := range []int{2, 8, 32} {
			t.Run(fmt.Sprintf("width=%d/experts=%d", width, n), func(t *testing.T) {
				w := &SparseMoEWeights{Gate: ExpertWeight{Output: width}}
				gate, up := randomExpertInput(n*width, 43000), randomExpertInput(n*width, 43001)
				for i := range gate {
					gate[i] *= 50
				}
				copy(gate, []float32{0, math.Float32frombits(1 << 31), 100, -100,
					float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())})
				got, want := make([]float32, len(gate)), make([]float32, len(gate))
				for i := range n {
					start, end := i*width, (i+1)*width
					siluMulF32(gate[start:end], up[start:end], want[start:end])
				}
				if !sparseMoEActivateBatch(w, gate, up, got) {
					t.Fatal("aligned bias-free activation declined")
				}
				requireMoEPrefillExact(t, got, want)
			})
		}
	}
}

func TestMoEActivationBatchRetainsBiasAndTailFallback(t *testing.T) {
	for _, w := range []SparseMoEWeights{
		{Gate: ExpertWeight{Output: 0}},
		{Gate: ExpertWeight{Output: 7}},
		{Gate: ExpertWeight{Output: 12}}, // ARM aligned, AMD64 scalar tail.
		{Gate: ExpertWeight{Output: 256}, OAIActivation: true},
		{Gate: ExpertWeight{Output: 256}, GateBias: ExpertBias{Values: []float32{1}}},
		{Gate: ExpertWeight{Output: 256}, UpBias: ExpertBias{Values: []float32{1}}},
	} {
		hidden := []float32{123}
		if sparseMoEActivateBatch(&w, nil, nil, hidden) || hidden[0] != 123 {
			t.Fatal("unsupported activation accepted or output modified")
		}
	}
}

func TestMoEBiasFreeActivationPreservesMixture(t *testing.T) {
	enableExpertDownBatchForTest(t)
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		for _, q8 := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/q8=%t", typ, q8), func(t *testing.T) {
				if q8 && !q8ActivationsAvailable() {
					t.Skip("Q8 activation instructions unavailable")
				}
				withQ8Activations(q8, func() {
					w := moePrefillFixture(typ, typ)
					w.GateBias, w.UpBias = ExpertBias{}, ExpertBias{}
					const p = 4
					var xf, of []float32
					var xs, outs [][]float32
					xs = reuseBatchViews(&xf, &xs, p, w.Gate.Input)
					outs = reuseBatchViews(&of, &outs, p, w.Down.Output)
					copy(xf, randomExpertInput(len(xf), 43002))
					if q8 {
						if err := sparseMoEPrefill(context.Background(), w, xs, outs, &DecodeBuffer{}); err != nil {
							t.Fatal(err)
						}
					}
					got, want := &DecodeBuffer{}, &DecodeBuffer{}
					for i, x := range xs {
						sparseMoEForward(w, x, got)
						if len(got.moeDown.hidden) != w.ExpertUsed*w.Gate.Output {
							t.Fatal("batched activation was not reached")
						}
						previous := disableExpertDownBatch.Swap(true)
						func() {
							defer disableExpertDownBatch.Store(previous)
							sparseMoEForward(w, x, want)
						}()
						requireMoEPrefillExact(t, got.Proj, want.Proj)
						if q8 {
							requireMoEPrefillExact(t, outs[i], want.Proj)
						}
					}
				})
			})
		}
	}
}

// Compare the same SIMD kernel with either one call per expert or one call
// for the contiguous slab. Matrix multiplication is outside this benchmark.
func BenchmarkMoEActivation(b *testing.B) {
	for _, n := range []int{8, 32} {
		for _, bulk := range []bool{false, true} {
			b.Run(fmt.Sprintf("experts=%d/bulk=%t", n, bulk), func(b *testing.B) {
				setPerformanceThreads(b, 1)
				const width = 512
				w := &SparseMoEWeights{Gate: ExpertWeight{Output: width}}
				gate, up := randomExpertInput(n*width, 43003), randomExpertInput(n*width, 43004)
				hidden := make([]float32, n*width)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if bulk {
						sparseMoEActivateBatch(w, gate, up, hidden)
					} else {
						for i := range n {
							start, end := i*width, (i+1)*width
							siluMulF32(gate[start:end], up[start:end], hidden[start:end])
						}
					}
				}
			})
		}
	}
}
