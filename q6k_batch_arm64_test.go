//go:build arm64

package gopherllm

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestQ6KBatch4BitExact(t *testing.T) {
	requireDotProd(t)
	if !q6kBatchAsmOK {
		t.Fatal("Q6_K batch startup self-check failed")
	}
	rng := rand.New(rand.NewSource(6401))
	for _, cols := range []int{256, 512, 3072, 9216} {
		blocks := cols / 256
		for trial := range 32 {
			row := randomQ6KRow(rng, cols)
			q8 := make([]int8, 4*cols)
			scales := make([]float32, 4*blocks)
			sums := make([]float32, 4*blocks*16)
			for i := range q8 {
				q8[i] = int8(rng.Intn(256) - 128)
				if trial == 9 {
					q8[i] = []int8{-128, 127}[i%2]
				}
			}
			for i := range scales {
				scales[i] = float32(rng.NormFloat64() * 0.03)
			}
			for i := range sums {
				sums[i] = float32(rng.NormFloat64() * 17)
			}
			for b := range blocks {
				h := uint16(rng.Intn(0x6000)) | uint16(rng.Intn(2))<<15
				if trial < 9 {
					h = [...]uint16{0, 0x8000, 1, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff, 0xfbff}[trial]
				} else if trial == 9 {
					// A nonzero weight scale keeps the integer extreme case
					// observable in the final output.
					h = 0x3c00
				}
				binary.LittleEndian.PutUint16(row[b*210+208:], h)
				if trial == 9 {
					for i := range 16 {
						row[b*210+192+i] = []byte{128, 127}[i%2]
					}
				}
			}
			got := q6kRowBatch4(row, q8, scales, sums, cols, blocks)
			for token := range 4 {
				want := q6kDotQ8KRowBlock(row, q8[token*cols:], scales[token*blocks:], sums[token*blocks*16:], blocks)
				if math.Float32bits(got[token]) != math.Float32bits(want) {
					t.Fatalf("cols=%d trial=%d token=%d: got %g (%08x), want %g (%08x)",
						cols, trial, token, got[token], math.Float32bits(got[token]), want, math.Float32bits(want))
				}
			}
		}
	}
}

func TestQ6KBatch4MixedProjectionsMatchFallback(t *testing.T) {
	requireDotProd(t)
	if !q6kBatchAsmOK || !q6kDotAsmOK {
		t.Fatal("Q6_K startup self-check failed")
	}
	previousBatch, previousQ8 := q6kBatchAsmOK, useQ8Activations.Swap(true)
	t.Cleanup(func() {
		q6kBatchAsmOK = previousBatch
		useQ8Activations.Store(previousQ8)
	})
	const cols, tokens = 512, 35 // Eight full groups and a three-token tail.
	rng := rand.New(rand.NewSource(6404))
	weights := [3]Weight{
		{Type: GGMLTypeQ6_K, Rows: 67, Cols: cols},
		{Type: GGMLTypeQ4_K, Rows: 31, Cols: cols},
		{Type: GGMLTypeQ6_K, Rows: 41, Cols: cols},
	}
	var want, got [3][][]float32
	for wi := range weights {
		w := &weights[wi]
		for range w.Rows {
			if w.Type == GGMLTypeQ6_K {
				w.Raw = append(w.Raw, randomQ6KRow(rng, cols)...)
			} else {
				w.Raw = append(w.Raw, randomQ4KRow(rng, cols)...)
			}
		}
		want[wi], got[wi] = make([][]float32, tokens), make([][]float32, tokens)
		for token := range tokens {
			want[wi][token], got[wi][token] = make([]float32, w.Rows), make([]float32, w.Rows)
		}
	}
	for pass := range 3 { // New inputs expose stale pooled activation/sum data.
		xs := make([][]float32, tokens)
		for token := range tokens {
			xs[token] = randomVec(rng, cols)
			if (token+pass)%3 == 0 {
				clear(xs[token][:256])
			}
		}
		q6kBatchAsmOK = false
		for wi, w := range weights {
			if !matvecBatchQ8(w, xs, want[wi]) {
				t.Fatal("separate reference projection declined")
			}
		}
		for _, mode := range []struct {
			name        string
			fast, fused bool
		}{{"separate", true, false}, {"fused_fallback", false, true}, {"fused", true, true}} {
			q6kBatchAsmOK = mode.fast
			for wi := range got {
				for token := range tokens {
					for row := range got[wi][token] {
						got[wi][token][row] = float32(math.NaN())
					}
				}
			}
			if mode.fused {
				if !matvecBatchQ8Fused3(weights[0], weights[1], weights[2], xs, got[0], got[1], got[2]) {
					t.Fatal("fused projection declined")
				}
			} else {
				for wi, w := range weights {
					if !matvecBatchQ8(w, xs, got[wi]) {
						t.Fatal("separate fast projection declined")
					}
				}
			}
			for wi := range got {
				for token := range tokens {
					for row, reference := range want[wi][token] {
						if value := got[wi][token][row]; math.Float32bits(value) != math.Float32bits(reference) {
							t.Fatalf("%s pass%d weight%d token%d row%d: got %08x, want %08x", mode.name, pass, wi, token, row,
								math.Float32bits(value), math.Float32bits(reference))
						}
					}
				}
			}
		}
	}
}

func TestQ6KBatch4RowsAndTails(t *testing.T) {
	requireDotProd(t)
	if !q6kBatchAsmOK {
		t.Fatal("Q6_K batch startup self-check failed")
	}
	rng := rand.New(rand.NewSource(6402))
	const rows = 9
	for _, cols := range []int{256, 3072} {
		blocks := cols / 256
		for _, tokens := range []int{4, 5, 7, 8, 9} {
			t.Run(fmt.Sprintf("cols%d/tokens%d", cols, tokens), func(t *testing.T) {
				raw := make([]byte, 0, rows*blocks*210)
				for range rows {
					raw = append(raw, randomQ6KRow(rng, cols)...)
				}
				q8 := make([]int8, tokens*cols)
				scales := make([]float32, tokens*blocks)
				sums := make([]float32, tokens*blocks*16)
				outs := make([][]float32, tokens)
				for token := range tokens {
					x := randomVec(rng, cols)
					if token%2 == 0 {
						clear(x[:256])
					}
					q8kQuantizePortable(x, q8[token*cols:], scales[token*blocks:], blocks)
					sub := sums[token*blocks*16 : (token+1)*blocks*16]
					fillQ6KXSums16(x, cols, &sub)
					ScaleF32(sub, 32)
					outs[token] = make([]float32, rows)
					for i := range outs[token] {
						outs[token][i] = -999
					}
				}
				w := Weight{Raw: raw, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols}
				if !batchQ6KRows4(w, outs, q8, scales, sums, 1, rows-1) {
					t.Fatal("valid Q6_K batch declined")
				}
				for token := range tokens {
					for row := range rows {
						want := float32(-999)
						if row >= 1 && row < rows-1 {
							want = q6kDotQ8KRowBlock(raw[row*blocks*210:], q8[token*cols:], scales[token*blocks:], sums[token*blocks*16:], blocks)
						}
						if math.Float32bits(outs[token][row]) != math.Float32bits(want) {
							t.Fatalf("token%d row%d: got %g (%08x), want %g (%08x)", token, row,
								outs[token][row], math.Float32bits(outs[token][row]), want, math.Float32bits(want))
						}
					}
				}
			})
		}
	}
}

func TestQ6KBatch4Bounds(t *testing.T) {
	if got := q6kRowBatch4(nil, nil, nil, nil, 0, 0); got != [4]float32{} {
		t.Fatalf("empty batch = %v", got)
	}
	for _, tc := range []struct {
		name                                string
		row, q8, scales, sums, cols, blocks int
	}{
		{"weights", 209, 1024, 4, 64, 256, 1},
		{"activations", 210, 1023, 4, 64, 256, 1},
		{"scales", 210, 1024, 3, 64, 256, 1},
		{"sums", 210, 1024, 4, 63, 256, 1},
		{"stride", 210, 1024, 4, 64, 257, 1},
		{"mismatched count", 210, 1024, 4, 64, 512, 1},
		{"overflow", 210, 1024, 4, 64, 256, int(^uint(0) >> 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid buffers reached assembly")
				}
			}()
			q6kRowBatch4(make([]byte, tc.row), make([]int8, tc.q8), make([]float32, tc.scales),
				make([]float32, tc.sums), tc.cols, tc.blocks)
		})
	}
}

func TestQ6KBatch4DeclinesUnsupported(t *testing.T) {
	if batchQ6KRows4(Weight{Type: GGMLTypeQ4_K}, make([][]float32, 4), nil, nil, nil, 0, 1) {
		t.Fatal("wrong weight format accepted")
	}
	if batchQ6KRows4(Weight{Type: GGMLTypeQ6_K}, make([][]float32, 3), nil, nil, nil, 0, 1) {
		t.Fatal("short token batch accepted")
	}
	previous := q6kBatchAsmOK
	q6kBatchAsmOK = false
	t.Cleanup(func() { q6kBatchAsmOK = previous })
	if batchQ6KRows4(Weight{Type: GGMLTypeQ6_K}, make([][]float32, 4), nil, nil, nil, 0, 1) {
		t.Fatal("disabled assembly accepted a batch")
	}
}

var q6kBatchBenchmarkSink [4]float32

func BenchmarkQ6KBatch4Row(b *testing.B) {
	if !q6kBatchAsmOK || !q6kDecodeAsmOK {
		b.Skip("Q6_K SDOT unavailable")
	}
	for _, cols := range []int{3072, 9216} {
		rng := rand.New(rand.NewSource(6403))
		blocks := cols / 256
		row := randomQ6KRow(rng, cols)
		q8 := make([]int8, 4*cols)
		scales := make([]float32, 4*blocks)
		sums := make([]float32, 4*blocks*16)
		for token := range 4 {
			x := randomVec(rng, cols)
			q8kQuantizePortable(x, q8[token*cols:], scales[token*blocks:], blocks)
			sub := sums[token*blocks*16 : (token+1)*blocks*16]
			fillQ6KXSums16(x, cols, &sub)
			ScaleF32(sub, 32)
		}
		b.Run(fmt.Sprintf("cols%d/separate", cols), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for token := range 4 {
					q6kBatchBenchmarkSink[token] = q6kRowDecode(row, q8[token*cols:], scales[token*blocks:], sums[token*blocks*16:], blocks)
				}
			}
		})
		b.Run(fmt.Sprintf("cols%d/batch4", cols), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				q6kBatchBenchmarkSink = q6kRowBatch4(row, q8, scales, sums, cols, blocks)
			}
		})
	}
}
