package gopherllm

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
)

func TestExpertGateUpBatchFusedPlanes(t *testing.T) {
	const input, output, experts = 512, 37, 8
	selected := []ExpertScore{{Index: 7}, {Index: 0}, {Index: 3}, {Index: 1}, {Index: 7}}
	for _, threads := range []int{1, 3, 8} {
		for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ6_K} {
			for _, q8 := range []bool{false, true} {
				t.Run(fmt.Sprintf("threads=%d/%s/q8=%t", threads, typ, q8), func(t *testing.T) {
					setPerformanceThreads(t, threads)
					if q8 && !q8ActivationsAvailable() {
						t.Skip("Q8 activation instructions unavailable")
					}
					withQ8Activations(q8, func() {
						gate := moeDownWeightForTest(typ, input, 2*output, experts)
						gate.StorageOutput, gate.Output = 2*output, output
						up := gate
						up.RowOffset = output
						x := randomExpertInput(input, 41002)
						var g, u []float32
						if !expertMatvecGateUpBatchInto(gate, up, selected, x, nil, &g, &u) {
							t.Fatal("fused planes declined")
						}
						for j, choice := range selected {
							var wantG, wantU, sums []float32
							if !expertMatvec2Into(gate, up, choice.Index, x, &sums, &wantG, &wantU) {
								t.Fatal("reference declined")
							}
							for i := range output {
								if math.Float32bits(g[j*output+i]) != math.Float32bits(wantG[i]) || math.Float32bits(u[j*output+i]) != math.Float32bits(wantU[i]) {
									t.Fatalf("expert=%d row=%d differs", choice.Index, i)
								}
							}
						}
					})
				})
			}
		}
	}
}

func TestExpertGateUpBatchInvalidPlanePreservesOutputs(t *testing.T) {
	gate := moeDownWeightForTest(GGMLTypeQ4_K, 256, 37, 3)
	up := gate
	up.Weight.Raw = up.Weight.Raw[:len(up.Weight.Raw)-1]
	g, u := []float32{11}, []float32{22}
	if expertMatvecGateUpBatchInto(gate, up, []ExpertScore{{Index: 0}, {Index: 2}}, make([]float32, 256), nil, &g, &u) {
		t.Fatal("truncated last plane accepted")
	}
	if !slices.Equal(g, []float32{11}) || !slices.Equal(u, []float32{22}) {
		t.Fatal("declined batch changed caller outputs")
	}
}

func TestExpertGateUpBatchConcurrent(t *testing.T) {
	setPerformanceThreads(t, 3)
	w := moeDownWeightForTest(GGMLTypeQ4_K, 256, 37, 8)
	selected := []ExpertScore{{Index: 7}, {Index: 0}, {Index: 3}}
	var workers sync.WaitGroup
	for request := range 8 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			x := randomExpertInput(256, int64(41100+request))
			var sums, wantG, wantU []float32
			expertMatvecGateUpBatchInto(w, w, selected, x, &sums, &wantG, &wantU)
			for range 10 {
				var g, u []float32
				if !expertMatvecGateUpBatchInto(w, w, selected, x, &sums, &g, &u) || !slices.Equal(g, wantG) || !slices.Equal(u, wantU) {
					t.Error("concurrent request changed gate/up outputs")
					return
				}
			}
		}()
	}
	workers.Wait()
}

func BenchmarkExpertGateUpBatch(b *testing.B) {
	const input, output, experts = 2048, 512, 8
	selected := []ExpertScore{{Index: 7}, {Index: 0}, {Index: 4}, {Index: 1}, {Index: 6}, {Index: 2}, {Index: 5}, {Index: 3}}
	x := randomExpertInput(input, 41000)
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ6_K} {
		b.Run(typ.String(), func(b *testing.B) {
			setPerformanceThreads(b, 8)
			gate := moeDownWeightForTest(typ, input, output, experts)
			up := moeDownWeightForTest(typ, input, output, experts)
			var g, u, sums []float32
			if !expertMatvecGateUpBatchInto(gate, up, selected, x, &sums, &g, &u) {
				b.Fatal("gate/up batch declined")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				expertMatvecGateUpBatchInto(gate, up, selected, x, &sums, &g, &u)
			}
		})
	}
}

func BenchmarkMoESharedExpert(b *testing.B) {
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
		b.Run(fmt.Sprint(typ), func(b *testing.B) {
			setPerformanceThreads(b, 8)
			gate, _ := expertPlaneWeight(moeDownWeightForTest(typ, 2048, 512, 1), 0)
			up, _ := expertPlaneWeight(moeDownWeightForTest(typ, 2048, 512, 1), 0)
			down, _ := expertPlaneWeight(moeDownWeightForTest(GGMLTypeQ6_K, 512, 2048, 1), 0)
			w := &SparseMoEWeights{SharedGate: &gate, SharedUp: &up, SharedDown: &down, SharedAlways: true, Down: ExpertWeight{Output: 2048}}
			x, buf := randomExpertInput(2048, 41001), &DecodeBuffer{Proj: make([]float32, 2048)}
			sparseMoESharedForward(w, x, buf)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				clear(buf.Proj)
				sparseMoESharedForward(w, x, buf)
			}
		})
	}
}
