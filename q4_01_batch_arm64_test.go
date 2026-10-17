//go:build arm64

package gopherllm

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func legacyQ4TestRow(rng *rand.Rand, typ GGMLType, cols int) []byte {
	stride := 18
	if typ == GGMLTypeQ4_1 {
		stride = 20
	}
	row := make([]byte, cols/32*stride)
	rng.Read(row)
	for g := range cols / 32 {
		binary.LittleEndian.PutUint16(row[g*stride:], uint16(rng.Intn(0x6000))|uint16(rng.Intn(2))<<15)
		if stride == 20 {
			binary.LittleEndian.PutUint16(row[g*stride+2:], uint16(rng.Intn(0x6000))|uint16(rng.Intn(2))<<15)
		}
	}
	return row
}

func legacyQ4TestDot(typ GGMLType, row []byte, q8 []int8, scales, sums []float32, blocks int) float32 {
	if typ == GGMLTypeQ4_0 {
		return q4_0DotQ8KRow(row, q8, scales, sums, blocks)
	}
	return q4_1DotQ8KRow(row, q8, scales, sums, blocks)
}

func TestLegacyQ4Batch4BitExact(t *testing.T) {
	requireDotProd(t)
	if !q4_01BatchAsmOK {
		t.Fatal("legacy Q4 batch startup check failed")
	}
	rng := rand.New(rand.NewSource(40104))
	for _, typ := range []GGMLType{GGMLTypeQ4_0, GGMLTypeQ4_1} {
		for _, blocks := range []int{1, 2, 3, 8, 10, 12, 16, 27, 36, 64} {
			cols := blocks * 256
			q8 := make([]int8, 4*cols)
			scales, sums := make([]float32, 4*blocks), make([]float32, 32*blocks)
			for iter := range 60 {
				row := legacyQ4TestRow(rng, typ, cols)
				stride := 18
				if typ == GGMLTypeQ4_1 {
					stride = 20
				}
				if iter%17 == 0 {
					for g := range cols / 32 {
						binary.LittleEndian.PutUint16(row[g*stride:], uint16(g&1)<<15)
						if stride == 20 {
							binary.LittleEndian.PutUint16(row[g*stride+2:], uint16(g&1)<<15)
						}
					}
				}
				for i := range q8 {
					q8[i] = int8(rng.Intn(256) - 128)
				}
				for i := range scales {
					scales[i] = float32(rng.NormFloat64() * 0.17)
					if iter%19 == 0 {
						scales[i] = math.Float32frombits(uint32(i&1) << 31)
					}
				}
				for i := range sums {
					sums[i] = float32(rng.NormFloat64() * 17)
					if iter%23 == 0 {
						sums[i] = math.Float32frombits(uint32(i&1) << 31)
					}
				}
				got := q4_01RowBatch4(typ, row, q8, scales, sums, cols, blocks)
				for token := range 4 {
					want := legacyQ4TestDot(typ, row, q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
					if math.Float32bits(got[token]) != math.Float32bits(want) {
						t.Fatalf("%s blocks=%d iter=%d token=%d: got %g (%08x), want %g (%08x)", typ, blocks, iter, token,
							got[token], math.Float32bits(got[token]), want, math.Float32bits(want))
					}
				}
			}
		}
	}
}

func TestLegacyQ4Batch4MatchesSingleToken(t *testing.T) {
	requireDotProd(t)
	if !q4_01BatchAsmOK {
		t.Fatal("legacy Q4 batch startup check failed")
	}
	rng := rand.New(rand.NewSource(40105))
	for _, typ := range []GGMLType{GGMLTypeQ4_0, GGMLTypeQ4_1} {
		for _, cols := range []int{256, 2048, 2560, 6912} {
			for _, tokens := range []int{4, 5, 7, 8, 9} {
				t.Run(fmt.Sprintf("%s/cols%d/P%d", typ, cols, tokens), func(t *testing.T) {
					const rows = 11
					blocks := cols / 256
					var raw []byte
					for range rows {
						raw = append(raw, legacyQ4TestRow(rng, typ, cols)...)
					}
					rowBytes := len(raw) / rows
					w := Weight{Raw: raw, Type: typ, Rows: rows, Cols: cols}
					q8, scales, sums := make([]int8, tokens*cols), make([]float32, tokens*blocks), make([]float32, tokens*blocks*8)
					outs := make([][]float32, tokens)
					for token := range tokens {
						x := randomVec(rng, cols)
						if token%2 == 0 {
							clear(x[:256])
						}
						q8kQuantizePortable(x, q8[token*cols:], scales[token*blocks:], blocks)
						sub := sums[token*blocks*8 : (token+1)*blocks*8]
						fillQ4KXSums(x, cols, &sub)
						outs[token] = make([]float32, rows)
						for row := range rows {
							outs[token][row] = -999
						}
					}
					if !batchQ4_01Rows4(w, outs, q8, scales, sums, 3, rows-1) {
						t.Fatal("valid batch declined")
					}
					for token := range tokens {
						for row := range rows {
							want := float32(-999)
							if row >= 3 && row < rows-1 {
								want = legacyQ4TestDot(typ, raw[row*rowBytes:], q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
							}
							if math.Float32bits(outs[token][row]) != math.Float32bits(want) {
								t.Fatalf("token %d row %d: got %g (%08x), want %g (%08x)", token, row,
									outs[token][row], math.Float32bits(outs[token][row]), want, math.Float32bits(want))
							}
						}
					}
				})
			}
		}
	}
}

func TestLegacyQ4Batch4InputBounds(t *testing.T) {
	if got := q4_01RowBatch4(GGMLTypeQ4_0, nil, nil, nil, nil, 0, 0); got != [4]float32{} {
		t.Fatalf("empty row = %v", got)
	}
	for _, tc := range []struct {
		name                                string
		row, q8, scales, sums, cols, blocks int
	}{
		{"weights", 143, 1024, 4, 32, 256, 1},
		{"activations", 144, 1023, 4, 32, 256, 1},
		{"scales", 144, 1024, 3, 32, 256, 1},
		{"sums", 144, 1024, 4, 31, 256, 1},
		{"stride", 144, 1024, 4, 32, 257, 1},
		{"zero stride", 144, 1024, 4, 32, 0, 1},
		{"mismatched blocks", 144, 1024, 4, 32, 512, 1},
		{"overflow", 144, 1024, 4, 32, 256, int(^uint(0) >> 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid buffers reached assembly")
				}
			}()
			q4_01RowBatch4(GGMLTypeQ4_0, make([]byte, tc.row), make([]int8, tc.q8), make([]float32, tc.scales), make([]float32, tc.sums), tc.cols, tc.blocks)
		})
	}
	for _, w := range []Weight{
		{Type: GGMLTypeQ4_K}, {Type: GGMLTypeQ4_0}, {Type: GGMLTypeQ4_1},
		{Type: GGMLTypeQ4_0, Rows: 1, Cols: 256, Raw: make([]byte, 143)},
		{Type: GGMLTypeQ4_0, Rows: 1, Cols: 32, Raw: make([]byte, 18)},
		{Type: GGMLTypeQ4_0, Rows: 1, Cols: int(^uint(0)>>1) / 256 * 256},
	} {
		if batchQ4_01Rows4(w, make([][]float32, 4), nil, nil, nil, 0, 1) {
			t.Fatal("unsupported or incomplete weight accepted")
		}
	}
}

// Exercise both production dispatch sites with shared Q4_0/Q4_1 activation
// sums, unequal projection sizes and one/three trailing prompt tokens.
func TestLegacyQ4Batch4ProjectionDispatch(t *testing.T) {
	requireDotProd(t)
	if !q4_01BatchAsmOK {
		t.Fatal("legacy Q4 batch startup check failed")
	}
	rng := rand.New(rand.NewSource(40106))
	for _, shape := range []struct{ cols, tokens int }{{512, 35}, {2560, 33}} {
		t.Run(fmt.Sprintf("cols%d/P%d", shape.cols, shape.tokens), func(t *testing.T) {
			const count = 3
			weights := [count]Weight{
				{Type: GGMLTypeQ4_0, Rows: 9, Cols: shape.cols},
				{Type: GGMLTypeQ4_1, Rows: 7, Cols: shape.cols},
				{Type: GGMLTypeQ4_0, Rows: 11, Cols: shape.cols},
			}
			var separate, fused, want [count][][]float32
			for wi := range weights {
				w := &weights[wi]
				for range w.Rows {
					w.Raw = append(w.Raw, legacyQ4TestRow(rng, w.Type, w.Cols)...)
				}
				for _, outputs := range []*[][]float32{&separate[wi], &fused[wi], &want[wi]} {
					*outputs = make([][]float32, shape.tokens)
					for token := range *outputs {
						(*outputs)[token] = make([]float32, w.Rows)
					}
				}
			}
			xs := make([][]float32, shape.tokens)
			blocks := shape.cols / 256
			q8, scales, sums := make([]int8, shape.cols), make([]float32, blocks), make([]float32, blocks*8)
			for token := range xs {
				xs[token] = randomVec(rng, shape.cols)
				if token%3 == 0 {
					clear(xs[token][:256])
				}
				q8kQuantize(xs[token], q8, scales, blocks)
				fillQ4KXSums(xs[token], shape.cols, &sums)
				for wi, w := range weights {
					rowBytes := len(w.Raw) / w.Rows
					for row := range w.Rows {
						want[wi][token][row] = legacyQ4TestDot(w.Type, w.Raw[row*rowBytes:], q8, scales, sums, blocks)
					}
				}
			}
			withQ8Activations(true, func() {
				for wi, w := range weights {
					if !matvecBatchQ8(w, xs, separate[wi]) {
						t.Fatal("separate production batch declined")
					}
				}
				if !matvecBatchQ8Fused3(weights[0], weights[1], weights[2], xs, fused[0], fused[1], fused[2]) {
					t.Fatal("fused production batch declined")
				}
			})
			for wi := range weights {
				for token := range shape.tokens {
					for row, expected := range want[wi][token] {
						if math.Float32bits(separate[wi][token][row]) != math.Float32bits(expected) || math.Float32bits(fused[wi][token][row]) != math.Float32bits(expected) {
							t.Fatalf("weight %d token %d row %d: separate=%g fused=%g expected=%g", wi, token, row,
								separate[wi][token][row], fused[wi][token][row], expected)
						}
					}
				}
			}
		})
	}
}

func BenchmarkLegacyQ4Batch4Row(b *testing.B) {
	if !q4_01BatchAsmOK {
		b.Skip("SDOT unavailable")
	}
	for _, typ := range []GGMLType{GGMLTypeQ4_0, GGMLTypeQ4_1} {
		for _, cols := range []int{2048, 2560, 3072, 11008} {
			b.Run(fmt.Sprintf("%s/cols%d", typ, cols), func(b *testing.B) {
				blocks := cols / 256
				row := legacyQ4TestRow(rand.New(rand.NewSource(401)), typ, cols)
				q8, scales, sums := make([]int8, 4*cols), make([]float32, 4*blocks), make([]float32, 32*blocks)
				for i := range q8 {
					q8[i] = int8(i * 17)
				}
				for i := range scales {
					scales[i] = 0.017
				}
				b.Run("separate", func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						for token := range 4 {
							legacyQ4TestDot(typ, row, q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
						}
					}
				})
				b.Run("batch4", func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						q4_01RowBatch4(typ, row, q8, scales, sums, cols, blocks)
					}
				})
			})
		}
	}
}
