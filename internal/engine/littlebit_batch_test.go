package engine

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func littleBitBatchInput(rng *rand.Rand, size int) []float32 {
	x := make([]float32, size)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	return x
}

func TestLittleBitCPUBatchScratchOverflow(t *testing.T) {
	const tokens = 4
	xs, outs := make([][]float32, tokens), make([][]float32, tokens)
	for i := range xs {
		xs[i] = []float32{1}
	}
	oversized := math.MaxInt/tokens + 1
	w := &LittleBitWeight{Main: LittleBitFactors{Rank: 1}}
	if w.matvecBatchCPU(xs, outs, 1, oversized) {
		t.Fatal("overflowing input scratch dimensions were accepted")
	}
	w.Main.Rank = oversized
	if w.matvecBatchCPU(xs, outs, 1, 1) {
		t.Fatal("overflowing main rank scratch dimensions were accepted")
	}
	w.Main.Rank = 1
	w.Residual = &LittleBitFactors{Rank: oversized}
	if w.matvecBatchCPU(xs, outs, 1, 1) {
		t.Fatal("overflowing residual rank scratch dimensions were accepted")
	}
	for _, out := range outs {
		if out != nil {
			t.Fatal("an unsuccessful batch modified its outputs")
		}
	}
}

func TestLittleBitCPUWeightStationaryBatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(53, 3))
	const rows, cols = 65, 91
	main, _ := makeLittleBitFixture(rng, cols, rows, 35, GGMLTypeBF16)
	residual, _ := makeLittleBitFixture(rng, cols, rows, 13, GGMLTypeF16)
	bias := encodeLittleBitScale(littleBitBatchInput(rng, rows), GGMLTypeF32)
	w := Weight{Rows: rows, Cols: cols, LittleBit: &LittleBitWeight{Main: main, Residual: &residual, Bias: bias}}
	for _, n := range []int{2, 4, 5, littleBitCPUBatchTile + 5} {
		xs, got := make([][]float32, n), make([][]float32, n)
		for i := range xs {
			xs[i] = littleBitBatchInput(rng, cols)
		}
		matvecBatch(w, xs, got)
		for i, out := range got {
			want := w.Matvec(xs[i])
			if !slices.Equal(out, want) {
				t.Fatalf("batch %d token %d: got %v want %v", n, i, out, want)
			}
		}
	}
}

func BenchmarkLittleBitCPUPrefillProjection(b *testing.B) {
	const batch = 26
	for _, d := range []struct{ in, out, rank int }{{4096, 1024, 208}, {4096, 4096, 544}, {12288, 4096, 824}} {
		b.Run(fmt.Sprintf("%dx%dRank%d", d.out, d.in, d.rank), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(211, 2))
			main, _ := makeLittleBitFixture(rng, d.in, d.out, d.rank, GGMLTypeBF16)
			residual, _ := makeLittleBitFixture(rng, d.in, d.out, d.rank, GGMLTypeBF16)
			w := Weight{Rows: d.out, Cols: d.in, LittleBit: &LittleBitWeight{Main: main, Residual: &residual}}
			xs, outs := make([][]float32, batch), make([][]float32, batch)
			for i := range xs {
				xs[i] = littleBitBatchInput(rng, d.in)
				outs[i] = make([]float32, d.out)
			}
			for _, n := range []int{2, 4, batch} {
				for _, impl := range []struct {
					name string
					run  func()
				}{
					{"PerToken", func() {
						for i := range n {
							w.MatvecInto(xs[i], &outs[i])
						}
					}},
					{"WeightStationary", func() { matvecBatch(w, xs[:n], outs[:n]) }},
				} {
					b.Run(fmt.Sprintf("Batch%d/%s", n, impl.name), func(b *testing.B) {
						b.ReportAllocs()
						impl.run()
						b.ResetTimer()
						for range b.N {
							impl.run()
						}
					})
				}
			}
		})
	}
}
