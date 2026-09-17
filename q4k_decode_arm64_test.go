//go:build arm64

package gopherllm

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Retain the former per-block call and combine sequence as a bit-exact oracle
// even when q4kDotQ8KRow dispatches to the new full-row implementation.
func q4kDecodePriorRow(row []byte, q8 []int8, scales, sums []float32, blocks int) float32 {
	var dots [8]int32
	var total float32
	for b := range blocks {
		block := row[b*144 : (b+1)*144]
		q4kQ8Dots8Asm(&block[16], &q8[b*256], &dots[0])
		total += combineQ4KStyle(block[4:16], &dots, sums[b*8:],
			F16ToF32(binaryLE16(block)), F16ToF32(binaryLE16(block[2:])), scales[b])
	}
	return total
}

func TestQ4KDecodeBitExact(t *testing.T) {
	requireDotProd(t)
	if !q4kDecodeAsmOK {
		t.Fatal("Q4_K full-row startup self-check failed")
	}
	rng := rand.New(rand.NewSource(7401))
	for _, blocks := range []int{1, 2, 3, 8, 12, 16, 36, 64} {
		row := make([]byte, blocks*144)
		q8 := make([]int8, blocks*256)
		scales := make([]float32, blocks)
		sums := make([]float32, blocks*8)
		for iter := range 100 {
			rng.Read(row)
			for b := range blocks {
				// Signed and zero half scales, including the subnormal range.
				for _, offset := range []int{0, 2} {
					h := uint16(rng.Intn(0x6000)) | uint16(rng.Intn(2))<<15
					if iter%17 == 0 {
						h = 0
					}
					binary.LittleEndian.PutUint16(row[b*144+offset:], h)
				}
				scales[b] = float32(rng.Float64() * 0.17)
			}
			for i := range q8 {
				q8[i] = int8(rng.Intn(256) - 128)
			}
			for i := range sums {
				sums[i] = float32(rng.NormFloat64() * 17)
			}
			got := q4kRowDecode(row, q8, scales, sums, blocks)
			want := q4kDecodePriorRow(row, q8, scales, sums, blocks)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("blocks=%d iter=%d: got %g (%08x), want %g (%08x)", blocks, iter,
					got, math.Float32bits(got), want, math.Float32bits(want))
			}
		}
	}
}

func TestQ4KDecodeInputBounds(t *testing.T) {
	if got := q4kRowDecode(nil, nil, nil, nil, 0); got != 0 {
		t.Fatalf("empty row = %g", got)
	}
	for _, tc := range []struct {
		name                     string
		row, q8, scales, sums, n int
	}{
		{"weights", 143, 256, 1, 8, 1},
		{"activations", 144, 255, 1, 8, 1},
		{"scales", 144, 256, 0, 8, 1},
		{"sums", 144, 256, 1, 7, 1},
		{"overflow", 144, 256, 1, 8, int(^uint(0) >> 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("incomplete row did not panic before entering assembly")
				}
			}()
			q4kRowDecode(make([]byte, tc.row), make([]int8, tc.q8),
				make([]float32, tc.scales), make([]float32, tc.sums), tc.n)
		})
	}
}

func TestQ4KDecodeDispatchAndFallback(t *testing.T) {
	requireDotProd(t)
	if !q4kDecodeAsmOK || !q4kDotAsmOK {
		t.Fatal("Q4_K assembly startup self-check failed")
	}
	const cols = 3072
	const blocks = cols / 256
	rng := rand.New(rand.NewSource(7403))
	row := randomQ4KRow(rng, cols)
	x := randomVec(rng, cols)
	q8 := make([]int8, cols)
	scales := make([]float32, blocks)
	q8kQuantizePortable(x, q8, scales, blocks)
	var scratch []float32
	sums := fillQ4KXSums(x, cols, &scratch)
	want := q4kDecodePriorRow(row, q8, scales, sums, blocks)

	previous := q4kDecodeAsmOK
	t.Cleanup(func() { q4kDecodeAsmOK = previous })
	for _, enabled := range []bool{true, false} {
		q4kDecodeAsmOK = enabled
		got := q4kDotQ8KRow(row, q8, scales, sums, blocks)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("full-row enabled=%t: got %g (%08x), want %g (%08x)", enabled,
				got, math.Float32bits(got), want, math.Float32bits(want))
		}
	}
}

var q4kDecodeBenchmarkSink float32

func BenchmarkQ4KRowDecode(b *testing.B) {
	if !q4kDecodeAsmOK || !q4kDotAsmOK {
		b.Skip("Q4_K SDOT unavailable")
	}
	for _, cols := range []int{2048, 3072, 4096, 9216} {
		rng := rand.New(rand.NewSource(7402))
		blocks := cols / 256
		row := randomQ4KRow(rng, cols)
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		scales := make([]float32, blocks)
		q8kQuantizePortable(x, q8, scales, blocks)
		var scratch []float32
		sums := fillQ4KXSums(x, cols, &scratch)
		for _, kernel := range []struct {
			name string
			fn   func([]byte, []int8, []float32, []float32, int) float32
		}{{"block", q4kDecodePriorRow}, {"full", q4kRowDecode}} {
			b.Run(fmt.Sprintf("%d/%s", cols, kernel.name), func(b *testing.B) {
				b.SetBytes(int64(len(row)))
				b.ReportAllocs()
				for range b.N {
					q4kDecodeBenchmarkSink = kernel.fn(row, q8, scales, sums, blocks)
				}
			})
		}
	}
}
