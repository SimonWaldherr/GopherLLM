package engine

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
)

func TestExpertDownBatchConcurrentInputs(t *testing.T) {
	setPerformanceThreads(t, 3)
	w := moeDownWeightForTest(GGMLTypeQ6_K, 512, 37, 8)
	selected := []ExpertScore{{Index: 7}, {Index: 0}, {Index: 3}, {Index: 7}}
	var workers sync.WaitGroup
	for request := range 8 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			scratch := moeDownBatchScratch{hidden: randomExpertInput(len(selected)*w.Input, int64(42200+request))}
			var reference []float32
			for j, expert := range selected {
				var out, row []float32
				expertMatvecInto(w, expert.Index, scratch.hidden[j*w.Input:(j+1)*w.Input], &out, &row)
				reference = append(reference, out...)
			}
			for range 10 {
				expertDownBatchInto(w, selected, &scratch)
				if !slices.Equal(scratch.down, reference) {
					t.Error("concurrent request changed expert output")
					return
				}
			}
		}()
	}
	workers.Wait()
}

// Original per-expert input preparation, retained as an independent oracle
// and a benchmark baseline. No weight projection or worker dispatch is timed.
type separateExpertDownInputs struct {
	q8     [moeDownBatchMaxExperts][]int8
	scales [moeDownBatchMaxExperts][]float32
	leases [moeDownBatchMaxExperts]q8ScratchLease
	sums   []float32
}

func (s *separateExpertDownInputs) prepare(w ExpertWeight, x []float32, n int) {
	width := 0
	switch w.Weight.Type {
	case GGMLTypeQ4_K, GGMLTypeQ5_K:
		width = w.Input / 32
	case GGMLTypeQ6_K:
		width = w.Input / 16
	}
	ensureLenNoClear(&s.sums, n*width)
	for j := range n {
		row := x[j*w.Input : (j+1)*w.Input]
		sums := s.sums[j*width : (j+1)*width : (j+1)*width]
		switch w.Weight.Type {
		case GGMLTypeQ4_K, GGMLTypeQ5_K:
			fillQ4KXSums(row, w.Input, &sums)
		case GGMLTypeQ6_K:
			fillQ6KXSums16(row, w.Input, &sums)
			ScaleF32(sums, 32)
		}
		s.q8[j], s.scales[j], s.leases[j] = acquireQ8(row, w.Input)
	}
}

func (s *separateExpertDownInputs) release(n int) {
	for j := range n {
		releaseQ8(s.q8[j], s.scales[j], s.leases[j])
	}
}

func TestExpertDownInputsKeepIndependentBlockScales(t *testing.T) {
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0, GGMLTypeMXFP4} {
		for _, cols := range []int{256, 512, 2048} {
			for _, n := range []int{2, 8, moeDownBatchMaxExperts} {
				t.Run(fmt.Sprintf("%s/cols=%d/experts=%d", typ, cols, n), func(t *testing.T) {
					x := randomExpertInput(cols*n, 42000)
					// Different experts and blocks have very different magnitudes.
					// A shared scale or a shifted row boundary must fail this test.
					for i := range x {
						x[i] *= float32(math.Ldexp(1, (i/256)%25-12))
					}
					clear(x[cols : 2*cols])
					w := ExpertWeight{Input: cols, Weight: Weight{Type: typ}}
					task := &expertDownTask{w: w, x: x, useQ8: true}
					var sums []float32
					lease := task.prepareInputs(&sums)
					defer releaseQ8(task.q8, task.scales, lease)
					var want separateExpertDownInputs
					want.prepare(w, x, n)
					defer want.release(n)
					for j := range n {
						if !slices.Equal(task.q8[j*cols:(j+1)*cols], want.q8[j]) {
							t.Fatalf("expert %d quantized input differs", j)
						}
						for b, scale := range want.scales[j] {
							if math.Float32bits(task.scales[j*(cols/256)+b]) != math.Float32bits(scale) {
								t.Fatalf("expert %d block %d scale differs", j, b)
							}
						}
					}
					if len(task.sums) != len(want.sums) {
						t.Fatal("offset sum count differs")
					}
					for i, sum := range want.sums {
						if math.Float32bits(task.sums[i]) != math.Float32bits(sum) {
							t.Fatalf("offset sum %d differs", i)
						}
					}
				})
			}
		}
	}
}

func BenchmarkExpertDownInputs(b *testing.B) {
	if !q8ActivationsAvailable() {
		b.Skip("Q8 activation instructions unavailable")
	}
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		for _, n := range []int{8, 32} {
			for _, bulk := range []bool{false, true} {
				b.Run(fmt.Sprintf("%s/experts=%d/bulk=%t", typ, n, bulk), func(b *testing.B) {
					setPerformanceThreads(b, 1)
					const cols = 512
					w := ExpertWeight{Input: cols, Weight: Weight{Type: typ}}
					x := randomExpertInput(n*cols, 42001)
					task := &expertDownTask{w: w, x: x, useQ8: true}
					var sums []float32
					var separate separateExpertDownInputs
					run := func() {
						if bulk {
							lease := task.prepareInputs(&sums)
							releaseQ8(task.q8, task.scales, lease)
						} else {
							separate.prepare(w, x, n)
							separate.release(n)
						}
					}
					run()
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						run()
					}
				})
			}
		}
	}
}
