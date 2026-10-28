//go:build arm64

package engine

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestQ4KBatch4MatchesSingleToken(t *testing.T) {
	requireDotProd(t)
	if !q4kBatchAsmOK {
		t.Fatal("batch kernel startup check failed")
	}
	rng := rand.New(rand.NewSource(4404))
	for _, cols := range []int{256, 3072, 9216} {
		for _, tokens := range []int{4, 5, 7, 8, 9} {
			t.Run(fmt.Sprintf("cols=%d/tokens=%d", cols, tokens), func(t *testing.T) {
				const rows = 19
				blocks := cols / 256
				raw := make([]byte, 0, rows*blocks*144)
				for range rows {
					raw = append(raw, randomQ4KRow(rng, cols)...)
				}
				w := Weight{Raw: raw, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols}
				q8 := make([]int8, tokens*cols)
				scales := make([]float32, tokens*blocks)
				sums := make([]float32, tokens*blocks*8)
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
				if !batchQ4KRows4(w, outs, q8, scales, sums, 3, rows-1) {
					t.Fatal("valid batch declined")
				}
				for token := range tokens {
					for row := range rows {
						want := float32(-999)
						if row >= 3 && row < rows-1 {
							want = q4kDotQ8KRow(raw[row*blocks*144:], q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
						}
						if math.Float32bits(outs[token][row]) != math.Float32bits(want) {
							t.Fatalf("token %d row %d: got %v (%08x), want %v (%08x)", token, row, outs[token][row], math.Float32bits(outs[token][row]), want, math.Float32bits(want))
						}
					}
				}
			})
		}
	}
}

// The earlier single-token block kernel remains an independent oracle for
// signed/subnormal scales and cancellation-sensitive float reductions.
func TestQ4KBatch4BitExact(t *testing.T) {
	requireDotProd(t)
	if !q4kBatchAsmOK {
		t.Fatal("batch kernel startup check failed")
	}
	rng := rand.New(rand.NewSource(4405))
	for _, blocks := range []int{1, 2, 3, 8, 12, 16, 36, 64} {
		cols := blocks * 256
		row := make([]byte, blocks*144)
		q8 := make([]int8, 4*cols)
		scales := make([]float32, 4*blocks)
		sums := make([]float32, 32*blocks)
		for iter := range 100 {
			rng.Read(row)
			for b := range blocks {
				for _, offset := range []int{0, 2} {
					h := uint16(rng.Intn(0x6000)) | uint16(rng.Intn(2))<<15
					if iter%17 == 0 {
						h &= 0x8000
					}
					binary.LittleEndian.PutUint16(row[b*144+offset:], h)
				}
			}
			for i := range q8 {
				q8[i] = int8(rng.Intn(256) - 128)
			}
			for i := range scales {
				scales[i] = float32(rng.NormFloat64() * .17)
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
			got := q4kRowBatch4(row, q8, scales, sums, cols, blocks)
			for token := range 4 {
				want := q4kDecodePriorRow(row, q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
				if math.Float32bits(got[token]) != math.Float32bits(want) {
					t.Fatalf("blocks=%d iter=%d token=%d: got %g (%08x), want %g (%08x)", blocks, iter, token,
						got[token], math.Float32bits(got[token]), want, math.Float32bits(want))
				}
			}
		}
	}
}

func TestQ4KBatch4InputBounds(t *testing.T) {
	if got := q4kRowBatch4(nil, nil, nil, nil, 0, 0); got != [4]float32{} {
		t.Fatalf("empty batch = %v", got)
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
		{"mismatched count", 144, 1024, 4, 32, 512, 1},
		{"overflow", 144, 1024, 4, 32, 256, int(^uint(0) >> 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid buffers reached assembly")
				}
			}()
			q4kRowBatch4(make([]byte, tc.row), make([]int8, tc.q8), make([]float32, tc.scales),
				make([]float32, tc.sums), tc.cols, tc.blocks)
		})
	}
}

func TestQ4KBatch4DeclinesUnsupportedInputs(t *testing.T) {
	for _, w := range []Weight{{Type: GGMLTypeQ6_K}, {Type: GGMLTypeQ4_K}} {
		if batchQ4KRows4(w, make([][]float32, 3), nil, nil, nil, 0, 1) {
			t.Fatal("short batch accepted")
		}
	}
	if batchQ4KRows4(Weight{Type: GGMLTypeQ6_K}, make([][]float32, 4), nil, nil, nil, 0, 1) {
		t.Fatal("Q6_K accepted")
	}
}

// Track the Q4_K projections alongside Q6_K when tuning mixed-quant Mistral
// models; improvements to one format must not hide regressions in the other.
func BenchmarkQ4KBatch4Row(b *testing.B) {
	if !q4kBatchAsmOK {
		b.Skip("SDOT unavailable")
	}
	for _, cols := range []int{3072, 9216} {
		b.Run(fmt.Sprint(cols), func(b *testing.B) {
			blocks := cols / 256
			row := randomQ4KRow(rand.New(rand.NewSource(44)), cols)
			q8, scales, sums := make([]int8, 4*cols), make([]float32, 4*blocks), make([]float32, 32*blocks)
			for i := range q8 {
				q8[i] = int8(i * 17)
			}
			for i := range scales {
				scales[i] = .017
			}
			b.ReportAllocs()
			for b.Loop() {
				q4kRowBatch4(row, q8, scales, sums, cols, blocks)
			}
		})
	}
}
