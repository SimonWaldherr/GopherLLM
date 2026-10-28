//go:build darwin && arm64 && cgo

package engine

import (
	"fmt"
	"math"
	"testing"
)

func TestPrefillBLASExplicitWorkspaceSlabs(t *testing.T) {
	for _, tokens := range []int{1, 16, 33, 128} {
		t.Run(fmt.Sprint(tokens), func(t *testing.T) {
			const rows, cols = 259, 256
			w := prefillBLASTestWeight(GGMLTypeF32, rows, cols)
			var input, output []float32
			var xs, outs [][]float32
			reuseBatchViews(&input, &xs, tokens, cols)
			reuseBatchViews(&output, &outs, tokens, rows)
			copy(input, benchFloatSlice(len(input)))
			want := prefillBLASReference(w, xs)
			matvecBatchSlabs(w, xs, outs, input, output)
			checkPrefillBLASOutput(t, outs, want)
			for _, row := range xs {
				if cap(row) != cols {
					t.Fatal("input capacity widened")
				}
			}
			for _, row := range outs {
				if cap(row) != rows {
					t.Fatal("output capacity widened")
				}
			}
			// Wrong provenance is only a fast-path miss. The caller's valid
			// row views retain the established packing/portable fallback.
			clear(output)
			matvecBatchSlabs(w, xs, outs, input[:len(input)-1], make([]float32, len(output)))
			checkPrefillBLASOutput(t, outs, want)
		})
	}
}

func TestPrefillBLASExplicitSlabAliases(t *testing.T) {
	const tokens, width = 33, 256
	w := prefillBLASTestWeight(GGMLTypeF32, width, width)
	for _, offset := range []int{0, 1, width} {
		t.Run(fmt.Sprint(offset), func(t *testing.T) {
			data := benchFloatSlice(tokens*width + offset + 2)
			first, last := data[0], data[len(data)-1]
			input := data[1 : 1+tokens*width]
			output := data[1+offset : 1+offset+tokens*width]
			xs, outs := make([][]float32, tokens), make([][]float32, tokens)
			for token := range tokens {
				xs[token] = input[token*width : (token+1)*width : (token+1)*width]
				outs[token] = output[token*width : (token+1)*width : (token+1)*width]
			}
			want := prefillBLASReference(w, xs)
			if !matvecBatchBLASWithSlabs(w, xs, outs, false, input, output) {
				t.Fatal("valid aliased projection declined")
			}
			checkPrefillBLASOutput(t, outs, want)
			if data[0] != first || data[len(data)-1] != last {
				t.Fatal("aliased projection changed guards")
			}
		})
	}
	// An explicit slab must not bypass the existing output/weight guard.
	var input []float32
	var xs [][]float32
	reuseBatchViews(&input, &xs, tokens, width)
	output := w.F32[:tokens*width]
	outs := make([][]float32, tokens)
	for token := range tokens {
		outs[token] = output[token*width : (token+1)*width : (token+1)*width]
	}
	before := append([]float32(nil), w.F32...)
	if matvecBatchBLASWithSlabs(w, xs, outs, false, input, output) {
		t.Fatal("output/weight alias accepted")
	}
	for i, value := range before {
		if math.Float32bits(value) != math.Float32bits(w.F32[i]) {
			t.Fatal("declined weight alias changed data")
		}
	}
}

func TestPrefillBLASExplicitSlabsPreserveOverlappingOutputOrder(t *testing.T) {
	const tokens, rows, cols, overlap = 33, 259, 256, 13
	w := prefillBLASTestWeight(GGMLTypeF32, rows, cols)
	var input []float32
	var xs [][]float32
	reuseBatchViews(&input, &xs, tokens, cols)
	copy(input, benchFloatSlice(len(input)))
	actual, expected := make([]float32, tokens*rows), make([]float32, tokens*rows)
	outs, refs := make([][]float32, tokens), make([][]float32, tokens)
	for token := range tokens {
		start := token * (rows - overlap)
		outs[token] = actual[start : start+rows : start+rows]
		refs[token] = expected[start : start+rows : start+rows]
	}
	if !matvecBatchBLAS(w, xs, refs, false) || !matvecBatchBLASWithSlabs(w, xs, outs, false, input, actual) {
		t.Fatal("overlapping-output fallback declined")
	}
	checkPrefillBLASOutput(t, [][]float32{actual}, [][]float32{expected})
}

func BenchmarkPrefillBLASWorkspaceSlabs(b *testing.B) {
	const rows, cols = 3072, 768
	w := prefillBLASTestWeight(GGMLTypeF32, rows, cols)
	for _, tokens := range []int{32, 128} {
		b.Run(fmt.Sprintf("P%d", tokens), func(b *testing.B) {
			var input, output []float32
			var xs, outs [][]float32
			reuseBatchViews(&input, &xs, tokens, cols)
			reuseBatchViews(&output, &outs, tokens, rows)
			copy(input, benchFloatSlice(len(input)))
			for _, explicit := range []bool{false, true} {
				name := "packing"
				if explicit {
					name = "explicit_slabs"
				}
				b.Run(name, func(b *testing.B) {
					if explicit {
						matvecBatchWithQ8AndSlabs(w, xs, outs, false, input, output)
					} else {
						matvecBatchWithQ8(w, xs, outs, false)
					}
					b.ReportAllocs()
					for b.Loop() {
						if explicit {
							matvecBatchWithQ8AndSlabs(w, xs, outs, false, input, output)
						} else {
							matvecBatchWithQ8(w, xs, outs, false)
						}
					}
				})
			}
		})
	}
}
