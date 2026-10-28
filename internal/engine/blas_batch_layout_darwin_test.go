//go:build darwin && cgo

package engine

import (
	"math"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/voxtralblas"
)

func blasLayoutRows(n, width int, flat bool) [][]float32 {
	rows := make([][]float32, n)
	if flat {
		slab := make([]float32, n*width)
		for i := range rows {
			rows[i] = slab[i*width : (i+1)*width]
		}
	} else {
		for i := range rows {
			rows[i] = make([]float32, width)
		}
	}
	return rows
}

// Reference keeps the previous packing/copying contract, including aliases.
func packedBlasMatvecReference(w Weight, xs, outs [][]float32) {
	cols, rows := len(xs[0]), len(w.F32)/len(xs[0])
	input, output := make([]float32, len(xs)*cols), make([]float32, len(xs)*rows)
	for i, x := range xs {
		copy(input[i*cols:], x)
	}
	voxtralblas.Mul(len(xs), rows, cols, w.F32, input, output)
	for i := range outs {
		copy(outs[i], output[i*rows:(i+1)*rows])
	}
}

func TestBlasBatchLayoutsMatchPackedReference(t *testing.T) {
	const n, width = 4, 8
	for _, inputFlat := range []bool{false, true} {
		for _, outputFlat := range []bool{false, true} {
			xs, outs, want := blasLayoutRows(n, width, inputFlat), blasLayoutRows(n, width, outputFlat), blasLayoutRows(n, width, false)
			w := Weight{F32: make([]float32, width*width), Rows: width, Cols: width}
			for i := range w.F32 {
				w.F32[i] = float32(i%11-5) * .03125
			}
			for i := range xs {
				for j := range xs[i] {
					xs[i][j] = float32(i+j%3-2) * .125
				}
			}
			packedBlasMatvecReference(w, xs, want)
			blasMatvecBatch(w, xs, outs)
			for i := range outs {
				for j, got := range outs[i] {
					if got != want[i][j] {
						t.Fatalf("flat input/output %v/%v: [%d,%d]=%g, want %g", inputFlat, outputFlat, i, j, got, want[i][j])
					}
				}
			}
			// Overlapping input and output must still read the original input batch.
			if inputFlat {
				packedBlasMatvecReference(w, xs, want)
				blasMatvecBatch(w, xs, xs)
				for i := range xs {
					for j, got := range xs[i] {
						if got != want[i][j] {
							t.Fatal("in-place batch changed")
						}
					}
				}
			}
		}
	}
	// A caller may also use part of the weight slab as output views.
	w := Weight{F32: make([]float32, width*width), Rows: width, Cols: width}
	for i := range w.F32 {
		w.F32[i] = float32(i%13-6) * .0625
	}
	xs, want := blasLayoutRows(n, width, true), blasLayoutRows(n, width, false)
	for i := range xs {
		for j := range xs[i] {
			xs[i][j] = float32(i+j) * .125
		}
	}
	packedBlasMatvecReference(w, xs, want)
	outs := make([][]float32, n)
	for i := range outs {
		outs[i] = w.F32[i*width : (i+1)*width]
	}
	blasMatvecBatch(w, xs, outs)
	for i := range outs {
		for j := range outs[i] {
			if outs[i][j] != want[i][j] {
				t.Fatal("output aliasing weight changed")
			}
		}
	}
}

func TestBlasAttentionLayoutsAndAliases(t *testing.T) {
	const n, heads, dim, past = 5, 2, 4, 2
	const width, total = heads * dim, past + n
	k, v := make([]float32, heads*total*dim), make([]float32, heads*total*dim)
	for i := range k {
		k[i] = float32(i%17-8) * .125
		v[i] = float32(i%13-6) * .125
	}
	for _, flat := range []bool{false, true} {
		q, out, want := blasLayoutRows(n, width, flat), blasLayoutRows(n, width, flat), blasLayoutRows(n, width, false)
		queries, expected, scores := make([]float32, n*width), make([]float32, n*width), make([]float32, n*total)
		for i := range q {
			for j := range q[i] {
				q[i][j] = float32(i+j%7-3) * .0625
			}
			copy(queries[i*width:], q[i])
		}
		voxtralblas.Attention(n, total, heads, dim, past, 3, total, .5, queries, k, v, scores, expected)
		for i := range want {
			copy(want[i], expected[i*width:])
		}
		if !blasAttentionBatch(q, k, v, out, heads, dim, past, 3, total, .5) {
			t.Fatal("attention rejected")
		}
		for i := range out {
			for j := range out[i] {
				if math.Float32bits(out[i][j]) != math.Float32bits(want[i][j]) {
					t.Fatal("attention layout changed")
				}
			}
		}
		if flat {
			blasAttentionBatch(q, k, v, q, heads, dim, past, 3, total, .5)
			for i := range q {
				for j := range q[i] {
					if q[i][j] != want[i][j] {
						t.Fatal("in-place attention changed")
					}
				}
			}
		}
	}
}

func TestBlasAttentionOutputAliasesKV(t *testing.T) {
	const n, heads, dim, past = 3, 2, 4, 1
	const width, total = heads * dim, past + n
	for _, aliasKeys := range []bool{false, true} {
		q := blasLayoutRows(n, width, true)
		k, v := make([]float32, heads*total*dim), make([]float32, heads*total*dim)
		for i := range k {
			k[i], v[i] = float32(i%11-5)*.125, float32(i%7-3)*.25
		}
		for i := range q {
			for j := range q[i] {
				q[i][j] = float32(i+j%3) * .125
			}
		}
		queries, _ := contiguousBatchRows(q, width)
		expected, scores := make([]float32, n*width), make([]float32, n*total)
		voxtralblas.Attention(n, total, heads, dim, past, 2, total, .5, queries, k, v, scores, expected)
		slab := v
		if aliasKeys {
			slab = k
		}
		out := make([][]float32, n)
		for i := range out {
			out[i] = slab[i*width : (i+1)*width]
		}
		blasAttentionBatch(q, k, v, out, heads, dim, past, 2, total, .5)
		for i, want := range expected {
			if slab[i] != want {
				t.Fatalf("aliasKeys=%v [%d]: got %g, want %g", aliasKeys, i, slab[i], want)
			}
		}
	}
}

func BenchmarkBlasBatchContiguousLayout(b *testing.B) {
	const n, rows, cols = 128, 1024, 1024
	w := Weight{F32: make([]float32, rows*cols), Rows: rows, Cols: cols}
	for i := range w.F32 {
		w.F32[i] = float32(i%17) * .0625
	}
	for _, flat := range []bool{false, true} {
		name := "packed"
		if flat {
			name = "contiguous"
		}
		b.Run(name, func(b *testing.B) {
			xs, outs := blasLayoutRows(n, cols, flat), blasLayoutRows(n, rows, flat)
			for i := range xs {
				for j := range xs[i] {
					xs[i][j] = float32((i+j)%7) * .125
				}
			}
			blasMatvecBatch(w, xs, outs)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				blasMatvecBatch(w, xs, outs)
			}
		})
	}
}
