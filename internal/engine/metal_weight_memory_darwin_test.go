//go:build darwin && cgo && metal

package engine

import (
	"math"
	"math/rand"
	"testing"
)

func TestMetalMissingHandleFallsBackToRawWeights(t *testing.T) {
	const rows, cols = 8192, 256
	formats := []struct {
		typ    GGMLType
		row    func(*rand.Rand, int) []byte
		matvec func([]byte, []float32, int, int, *[]float32)
	}{
		{GGMLTypeQ4_K, randomQ4KRow, MatvecQ4KInto},
		{GGMLTypeQ5_K, randomQ5KRow, MatvecQ5KInto},
		{GGMLTypeQ6_K, randomQ6KRow, MatvecQ6KInto},
		{GGMLTypeQ8_0, randomQ8_0Row, MatvecQ8_0Into},
	}
	for _, f := range formats {
		t.Run(f.typ.String(), func(t *testing.T) {
			rng := rand.New(rand.NewSource(209))
			var raw []byte
			for range rows {
				raw = append(raw, f.row(rng, cols)...)
			}
			x := metalTestVector(cols)
			var want []float32
			f.matvec(raw, x, rows, cols, &want)
			// A rejected GPU view must leave the original complete CPU view
			// usable. Also check a handle whose backend is unavailable.
			rejected := prepareMetalWeight(raw[:len(raw)-1], f.typ, rows, cols, true)
			if rejected != nil {
				releaseMetalWeight(rejected)
				t.Fatal("truncated GPU view was accepted")
			}
			for _, handle := range []*MetalWeight{rejected, {typ: f.typ, rows: rows, cols: cols}} {
				w := Weight{Raw: raw, Type: f.typ, Rows: rows, Cols: cols, Metal: handle}
				var got []float32
				w.MatvecInto(x, &got)
				if len(got) != len(want) {
					t.Fatalf("output length = %d, want %d", len(got), len(want))
				}
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("CPU fallback row %d = %g, want %g", i, got[i], want[i])
					}
				}
			}
		})
	}
}
