//go:build arm64

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestQ8_0PackedActivationLayout(t *testing.T) {
	requireDotProd(t)
	if !q8_0PackedAsmOK {
		t.Fatal("packed Q8_0 startup self-check failed")
	}
	for _, cols := range []int{256, 512, 2560} {
		for _, tokens := range []int{4, 5, 8, 9, 32} {
			q8 := make([]int8, tokens*cols)
			for i := range q8 {
				q8[i] = int8(i*13 + i/cols*17)
			}
			before := slices.Clone(q8)
			var packed []int8
			if !prepareQ8_0PackedBatch(&packed, q8, cols, tokens) {
				t.Fatal("valid activation batch declined")
			}
			if len(packed) != (tokens&^3)*cols {
				t.Fatalf("packed length = %d, want %d", len(packed), (tokens&^3)*cols)
			}
			for first := 0; first+4 <= tokens; first += 4 {
				for col := 0; col < cols; col += 4 {
					for token := range 4 {
						for i := range 4 {
							got := packed[first*cols+col*4+token*4+i]
							want := q8[(first+token)*cols+col+i]
							if got != want {
								t.Fatalf("cols=%d token=%d col=%d: got %d, want %d", cols, first+token, col+i, got, want)
							}
						}
					}
				}
			}
			if !slices.Equal(q8, before) {
				t.Fatal("packing changed the original activations")
			}
		}
	}
}

func TestQ8_0PackedBatchBitExact(t *testing.T) {
	requireDotProd(t)
	if !q8_0PackedAsmOK {
		t.Fatal("packed Q8_0 startup self-check failed")
	}
	rng := rand.New(rand.NewSource(9301))
	scaleBits := [...]uint16{0, 0x8000, 1, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff, 0xfbff}
	for _, cols := range []int{256, 512, 2560, 4096, 9728} {
		blocks := cols / 256
		for trial := range 12 {
			row := randomQ8_0Row(rng, cols)
			q8 := make([]int8, 4*cols)
			scales := make([]float32, 4*blocks)
			for token := range 4 {
				x := randomVec(rng, cols)
				if trial == 0 && token == 0 {
					clear(x)
				}
				q8kQuantize(x, q8[token*cols:], scales[token*blocks:], blocks)
			}
			if trial == 1 {
				for j := range blocks * 8 {
					scale := scaleBits[j%len(scaleBits)]
					row[j*34], row[j*34+1] = byte(scale), byte(scale>>8)
				}
				for i := range q8 {
					q8[i] = int8(i*17 + i/cols*13)
				}
			}
			var packed []int8
			prepareQ8_0PackedBatch(&packed, q8, cols, 4)
			got := q8_0RowBatch4Packed(row, packed, scales, blocks)
			want := q8_0RowBatch4(row, q8, scales, cols, blocks)
			for token := range 4 {
				if math.Float32bits(got[token]) != math.Float32bits(want[token]) {
					t.Fatalf("cols=%d trial=%d token=%d: got %g (%08x), want %g (%08x)", cols, trial, token, got[token], math.Float32bits(got[token]), want[token], math.Float32bits(want[token]))
				}
			}
		}
	}
}

func TestQ8_0PackedBatchRowsTailAndRange(t *testing.T) {
	requireDotProd(t)
	rng := rand.New(rand.NewSource(9302))
	const rows, cols = 19, 512
	const blocks = cols / 256
	for _, tokens := range []int{4, 5, 9} {
		raw := make([]byte, 0, rows*blocks*272)
		for range rows {
			raw = append(raw, randomQ8_0Row(rng, cols)...)
		}
		q8 := make([]int8, tokens*cols)
		scales := make([]float32, tokens*blocks)
		outs := make([][]float32, tokens)
		for token := range tokens {
			q8kQuantize(randomVec(rng, cols), q8[token*cols:], scales[token*blocks:], blocks)
			outs[token] = make([]float32, rows)
			for row := range rows {
				outs[token][row] = -999
			}
		}
		var packed []int8
		prepareQ8_0PackedBatch(&packed, q8, cols, tokens)
		if !batchQ8_0PackedRows4(Weight{Type: GGMLTypeQ8_0, Raw: raw, Rows: rows, Cols: cols}, outs, q8, packed, scales, 3, rows-1) {
			t.Fatal("valid batch declined")
		}
		for token := range tokens {
			for row := range rows {
				want := float32(-999)
				if row >= 3 && row < rows-1 {
					want = q8_0DotQ8KRow(raw[row*blocks*272:], q8[token*cols:], scales[token*blocks:], blocks)
				}
				if math.Float32bits(outs[token][row]) != math.Float32bits(want) {
					t.Fatalf("tokens=%d token=%d row=%d: got %g, want %g", tokens, token, row, outs[token][row], want)
				}
			}
		}
	}
}

func TestQ8_0PackedBatchBoundsAndFallback(t *testing.T) {
	if got := q8_0RowBatch4Packed(nil, nil, nil, 0); got != [4]float32{} {
		t.Fatalf("empty batch = %v", got)
	}
	for _, tc := range []struct {
		name                        string
		row, packed, scales, blocks int
	}{
		{"weights", 271, 1024, 4, 1},
		{"activations", 272, 1023, 4, 1},
		{"scales", 272, 1024, 3, 1},
		{"overflow", 272, 1024, 4, int(^uint(0) >> 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("short buffers reached assembly")
				}
			}()
			q8_0RowBatch4Packed(make([]byte, tc.row), make([]int8, tc.packed), make([]float32, tc.scales), tc.blocks)
		})
	}
	previous := q8_0PackedAsmOK
	q8_0PackedAsmOK = false
	t.Cleanup(func() { q8_0PackedAsmOK = previous })
	var packed []int8
	if prepareQ8_0PackedBatch(&packed, make([]int8, 1024), 256, 4) || packed != nil {
		t.Fatal("disabled packed kernel prepared a batch")
	}
	if batchQ8_0PackedRows4(Weight{Type: GGMLTypeQ8_0, Cols: 256}, make([][]float32, 4), nil, nil, nil, 0, 0) {
		t.Fatal("disabled packed kernel accepted a batch")
	}
}

func TestQ8_0PackedBatchDeclinesMalformedRanges(t *testing.T) {
	requireDotProd(t)
	weight := Weight{Type: GGMLTypeQ8_0, Cols: 256, Rows: 2, Raw: make([]byte, 544)}
	q8, packed, scales := make([]int8, 1024), make([]int8, 1024), make([]float32, 4)
	for _, tc := range []struct {
		name                       string
		start, end, rawLen, outLen int
	}{
		{"negative start", -1, 1, 544, 2},
		{"reversed", 2, 1, 544, 2},
		{"end", 0, 3, 544, 2},
		{"weights", 0, 2, 543, 2},
		{"outputs", 0, 2, 544, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := weight
			w.Raw = w.Raw[:tc.rawLen]
			outs := make([][]float32, 4)
			for token := range outs {
				outs[token] = make([]float32, tc.outLen)
				for row := range outs[token] {
					outs[token][row] = -999
				}
			}
			if batchQ8_0PackedRows4(w, outs, q8, packed, scales, tc.start, tc.end) {
				t.Fatal("malformed batch accepted")
			}
			for _, out := range outs {
				for _, value := range out {
					if value != -999 {
						t.Fatal("declined batch changed output")
					}
				}
			}
		})
	}
}

func TestQ8_0PackedMatvecAndFused3Integration(t *testing.T) {
	requireDotProd(t)
	if !q8_0PackedAsmOK {
		t.Fatal("packed Q8_0 startup self-check failed")
	}
	previousPacked, previousQ8 := q8_0PackedAsmOK, useQ8Activations.Load()
	t.Cleanup(func() {
		q8_0PackedAsmOK = previousPacked
		useQ8Activations.Store(previousQ8)
	})
	useQ8Activations.Store(true)
	const tokens, cols = 35, 512 // Eight packed groups plus three ordinary tails.
	rng := rand.New(rand.NewSource(9305))
	var weights [3]Weight
	for wi, rows := range []int{131, 17, 65} {
		raw := make([]byte, 0, rows*(cols/256)*272)
		for range rows {
			raw = append(raw, randomQ8_0Row(rng, cols)...)
		}
		weights[wi] = Weight{Type: GGMLTypeQ8_0, Raw: raw, Rows: rows, Cols: cols}
	}
	xs := make([][]float32, tokens)
	for token := range tokens {
		xs[token] = randomVec(rng, cols)
	}
	run := func(t *testing.T, fused bool) [3][][]float32 {
		t.Helper()
		var outs [3][][]float32
		for wi, weight := range weights {
			outs[wi] = make([][]float32, tokens)
			for token := range tokens {
				outs[wi][token] = make([]float32, weight.Rows)
			}
		}
		if fused {
			if !matvecBatchQ8Fused3(weights[0], weights[1], weights[2], xs, outs[0], outs[1], outs[2]) {
				t.Fatal("fused Q8_0 batch declined")
			}
		} else {
			for wi, weight := range weights {
				if !matvecBatchQ8(weight, xs, outs[wi]) {
					t.Fatal("Q8_0 batch declined")
				}
			}
		}
		return outs
	}
	for _, fused := range []bool{false, true} {
		t.Run(fmt.Sprintf("fused=%t", fused), func(t *testing.T) {
			q8_0PackedAsmOK = false
			want := run(t, fused)
			// Toggle repeatedly to exercise scratch and task reuse after a packed
			// invocation: disabling it must never reuse stale packed activations.
			for _, enabled := range []bool{true, false, true} {
				q8_0PackedAsmOK = enabled
				got := run(t, fused)
				for wi := range weights {
					for token := range tokens {
						for row, value := range got[wi][token] {
							if math.Float32bits(value) != math.Float32bits(want[wi][token][row]) {
								t.Fatalf("enabled=%t weight=%d token=%d row=%d: got %g, want %g", enabled, wi, token, row, value, want[wi][token][row])
							}
						}
					}
				}
			}
		})
	}
}

var q8_0PackedBenchmarkResult [4]float32

func BenchmarkQ8_0PackedRow(b *testing.B) {
	if !q8_0PackedAsmOK || !q8_0BatchAsmOK {
		b.Skip("SDOT unavailable")
	}
	for _, cols := range []int{2560, 4096, 9728} {
		rng := rand.New(rand.NewSource(9303))
		blocks := cols / 256
		row := randomQ8_0Row(rng, cols)
		q8 := make([]int8, 4*cols)
		scales := make([]float32, 4*blocks)
		for token := range 4 {
			q8kQuantize(randomVec(rng, cols), q8[token*cols:], scales[token*blocks:], blocks)
		}
		var packed []int8
		prepareQ8_0PackedBatch(&packed, q8, cols, 4)
		b.Run(fmt.Sprintf("cols%d/prior", cols), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				q8_0PackedBenchmarkResult = q8_0RowBatch4(row, q8, scales, cols, blocks)
			}
		})
		b.Run(fmt.Sprintf("cols%d/packed", cols), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				q8_0PackedBenchmarkResult = q8_0RowBatch4Packed(row, packed, scales, blocks)
			}
		})
	}
}

func BenchmarkQ8_0PackedMatrix(b *testing.B) {
	if !q8_0PackedAsmOK || !q8_0BatchAsmOK {
		b.Skip("SDOT unavailable")
	}
	const rows, cols, blocks = 512, 2560, 10
	for _, tokens := range []int{32, 128} {
		rng := rand.New(rand.NewSource(9304))
		raw := make([]byte, 0, rows*blocks*272)
		for range rows {
			raw = append(raw, randomQ8_0Row(rng, cols)...)
		}
		weight := Weight{Type: GGMLTypeQ8_0, Raw: raw, Rows: rows, Cols: cols}
		q8 := make([]int8, tokens*cols)
		scales := make([]float32, tokens*blocks)
		outs := make([][]float32, tokens)
		for token := range tokens {
			q8kQuantize(randomVec(rng, cols), q8[token*cols:], scales[token*blocks:], blocks)
			outs[token] = make([]float32, rows)
		}
		var packed []int8
		prepareQ8_0PackedBatch(&packed, q8, cols, tokens)
		b.Run(fmt.Sprintf("tokens%d/prior", tokens), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				batchQ8_0Rows4(weight, outs, q8, scales, 0, rows)
			}
		})
		b.Run(fmt.Sprintf("tokens%d/packed_with_prepare", tokens), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				prepareQ8_0PackedBatch(&packed, q8, cols, tokens)
				batchQ8_0PackedRows4(weight, outs, q8, packed, scales, 0, rows)
			}
		})
	}
}
