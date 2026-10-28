package engine

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func enableExpertDownBatchForTest(t testing.TB) {
	previous := disableExpertDownBatch.Swap(false)
	t.Cleanup(func() { disableExpertDownBatch.Store(previous) })
}

func moeDownWeightForTest(typ GGMLType, input, output, experts int) ExpertWeight {
	if typ != GGMLTypeF32 {
		return quantExpertWeightForTest(typ, input, output, experts, int64(33000+typ))
	}
	return ExpertWeight{Weight: Weight{F32: randomExpertInput(input*output*experts, 33000)}, Input: input, Output: output, Experts: experts}
}

func TestExpertDownBatchMatchesIndependentExperts(t *testing.T) {
	enableExpertDownBatchForTest(t)
	const input, output, experts = 512, 37, 8
	selected := []ExpertScore{{Index: 7}, {Index: 0}, {Index: 3}, {Index: 1}, {Index: 7}}
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0, GGMLTypeMXFP4} {
		for _, q8 := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/q8=%t", typ, q8), func(t *testing.T) {
				if q8 && !q8ActivationsAvailable() {
					t.Skip("Q8 activation instructions unavailable")
				}
				withQ8Activations(q8, func() {
					for _, offset := range []int{0, output} {
						w := moeDownWeightForTest(typ, input, 2*output, experts)
						w.StorageOutput, w.Output, w.RowOffset = 2*output, output, offset
						scratch := moeDownBatchScratch{hidden: randomExpertInput(len(selected)*input, 33001)}
						if !expertDownBatchReady(w, selected) {
							t.Fatal("valid expert planes declined")
						}
						expertDownBatchInto(w, selected, &scratch)
						for j, choice := range selected {
							var want, row []float32
							expertMatvecInto(w, choice.Index, scratch.hidden[j*input:(j+1)*input], &want, &row)
							got := scratch.down[j*output : (j+1)*output]
							for i := range want {
								if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
									t.Fatalf("offset=%d expert=%d row=%d got=%g want=%g", offset, choice.Index, i, got[i], want[i])
								}
							}
						}
					}
				})
			})
		}
	}
}

func TestExpertDownBatchBounds(t *testing.T) {
	enableExpertDownBatchForTest(t)
	w := moeDownWeightForTest(GGMLTypeQ4_K, 256, 37, 4)
	selected := []ExpertScore{{Index: 0}, {Index: 3}}
	if !expertDownBatchReady(w, selected) {
		t.Fatal("valid batch declined")
	}
	for _, choices := range [][]ExpertScore{nil, selected[:1], {{Index: -1}, {Index: 0}}, {{Index: 4}, {Index: 0}}, make([]ExpertScore, moeDownBatchMaxExperts+1)} {
		if expertDownBatchReady(w, choices) {
			t.Fatalf("invalid selection accepted: %v", choices)
		}
	}
	for _, mutate := range []func(*ExpertWeight){
		func(w *ExpertWeight) { w.Input = 255 },
		func(w *ExpertWeight) { w.Output = 0 },
		func(w *ExpertWeight) { w.Input = moeDownBatchScratchBytes },
		func(w *ExpertWeight) { w.Weight.Raw = w.Weight.Raw[:1] },
		func(w *ExpertWeight) { w.RowOffset = 1 },
		func(w *ExpertWeight) { w.Weight.Type = GGMLTypeF16 },
	} {
		candidate := w
		mutate(&candidate)
		if expertDownBatchReady(candidate, selected) {
			t.Fatal("invalid weight accepted")
		}
	}
	disableExpertDownBatch.Store(true)
	if expertDownBatchReady(w, selected) {
		t.Fatal("explicit opt-out ignored")
	}
}

func TestExpertDownBatchScratchBoundsAcrossLayers(t *testing.T) {
	var scratch moeDownBatchScratch
	for _, w := range []ExpertWeight{{Input: 600000, Output: 1}, {Input: 1, Output: 600000}} {
		scratch.prepare(w, 2)
		ensureLenNoClear(&scratch.down, 2*w.Output)
		ensureLenNoClear(&scratch.sums, 2*w.Input/16)
		if (cap(scratch.hidden)+cap(scratch.down)+cap(scratch.sums))*4 > moeDownBatchScratchBytes {
			t.Fatal("scratch retained both layers' maximum slabs")
		}
	}
}

func TestSparseMoEDownBatchPreservesMixture(t *testing.T) {
	enableExpertDownBatchForTest(t)
	const dim, hidden, experts, used = 256, 256, 8, 6
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0, GGMLTypeMXFP4} {
		for _, q8 := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/q8=%t", typ, q8), func(t *testing.T) {
				if q8 && !q8ActivationsAvailable() {
					t.Skip("Q8 activation instructions unavailable")
				}
				withQ8Activations(q8, func() {
					w := &SparseMoEWeights{
						Router:        Weight{F32: randomExpertInput(experts*dim, 34000)},
						Gate:          moeDownWeightForTest(GGMLTypeQ4_K, dim, hidden, experts),
						Up:            moeDownWeightForTest(GGMLTypeQ4_K, dim, hidden, experts),
						Down:          moeDownWeightForTest(typ, hidden, dim, experts),
						RouterBias:    randomExpertInput(experts, 34001),
						GateBias:      ExpertBias{Values: randomExpertInput(experts*hidden, 34002), Output: hidden, Experts: experts},
						UpBias:        ExpertBias{Values: randomExpertInput(experts*hidden, 34003), Output: hidden, Experts: experts},
						DownBias:      ExpertBias{Values: randomExpertInput(experts*dim, 34004), Output: dim, Experts: experts},
						NormalizeTopK: true, Scale: 2.5, ExpertUsed: used,
					}
					x := randomExpertInput(dim, 34005)
					for _, oai := range []bool{false, true} {
						w.OAIActivation = oai
						got, want := &DecodeBuffer{}, &DecodeBuffer{}
						sparseMoEForward(w, x, got)
						if len(got.moeDown.down) != used*dim {
							t.Fatal("batched down was not executed")
						}
						previous := disableExpertDownBatch.Swap(true)
						func() {
							defer disableExpertDownBatch.Store(previous)
							sparseMoEForward(w, x, want)
						}()
						if !slices.Equal(got.Proj, want.Proj) {
							t.Fatalf("mixture changed: oai=%t", oai)
						}
					}
				})
			})
		}
	}
}

func TestGemma4MoEBatchPreservesBranchNorms(t *testing.T) {
	enableExpertDownBatchForTest(t)
	const dim, hidden, experts = 256, 256, 4
	config := Config{Dim: dim, RMSNormEps: 1e-6}
	layer := Gemma4LayerWeights{
		FFNGate:      Weight{F32: randomExpertInput(128*dim, 37000)},
		FFNUp:        Weight{F32: randomExpertInput(128*dim, 37001)},
		FFNDown:      Weight{F32: randomExpertInput(dim*128, 37002)},
		FFNHiddenDim: 128, PostFFNNorm: randomExpertInput(dim, 37003),
	}
	// The checkpoint's fused bank has separate gate/up row ranges within
	// each expert plane. Both offsets must survive the coalesced dispatch.
	gate := quantExpertWeightForTest(GGMLTypeQ4_K, dim, 2*hidden, experts, 37004)
	gate.StorageOutput, gate.Output = 2*hidden, hidden
	up := gate
	up.RowOffset = hidden
	moe := &Gemma4MoEWeights{
		Router:      Weight{F32: randomExpertInput(experts*dim, 37005)},
		RouterScale: randomExpertInput(dim, 37006), PreNorm2: randomExpertInput(dim, 37007),
		PostNorm1: randomExpertInput(dim, 37008), PostNorm2: randomExpertInput(dim, 37009),
		Gate: gate, Up: up, Down: quantExpertWeightForTest(GGMLTypeQ6_K, hidden, dim, experts, 37010),
		DownScale: []float32{0.3, 1.7, -0.8, 2.5}, ExpertUsed: 3,
	}
	for _, q8 := range []bool{false, true} {
		if q8 && !q8ActivationsAvailable() {
			continue
		}
		withQ8Activations(q8, func() {
			got, want := &DecodeBuffer{X: randomExpertInput(dim, 37011)}, &DecodeBuffer{X: randomExpertInput(dim, 37011)}
			forwardNativeGemma4MoE(config, layer, moe, got)
			if len(got.moeDown.down) != 3*dim {
				t.Fatal("Gemma expert down batch was not executed")
			}
			previous := disableExpertGateUpBatchForTest.Swap(true)
			func() {
				defer disableExpertGateUpBatchForTest.Store(previous)
				forwardNativeGemma4MoE(config, layer, moe, want)
			}()
			if !slices.Equal(got.X, want.X) {
				t.Fatalf("Gemma shared/expert branch normalization changed: q8=%t", q8)
			}
		})
	}
}

func BenchmarkSparseMoEDownBatch(b *testing.B) {
	const dim, hidden, experts, used = 1024, 512, 16, 8
	w := &SparseMoEWeights{
		Router:        Weight{F32: randomExpertInput(experts*dim, 35000)},
		Gate:          quantExpertWeightForTest(GGMLTypeQ4_K, dim, hidden, experts, 35001),
		Up:            quantExpertWeightForTest(GGMLTypeQ4_K, dim, hidden, experts, 35002),
		Down:          quantExpertWeightForTest(GGMLTypeQ6_K, hidden, dim, experts, 35003),
		NormalizeTopK: true, Scale: 1, ExpertUsed: used,
	}
	x := randomExpertInput(dim, 35004)
	for _, disabled := range []bool{true, false} {
		b.Run(fmt.Sprintf("serial_down=%t", disabled), func(b *testing.B) {
			previous := disableExpertDownBatch.Swap(disabled)
			defer disableExpertDownBatch.Store(previous)
			buf := &DecodeBuffer{}
			sparseMoEForward(w, x, buf)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				sparseMoEForward(w, x, buf)
			}
		})
	}
}
