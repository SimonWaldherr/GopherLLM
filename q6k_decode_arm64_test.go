//go:build arm64

package gopherllm

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

func TestQ6KDecodeMatchesBlock(t *testing.T) {
	requireDotProd(t)
	if !q6kDecodeAsmOK {
		t.Fatal("Q6_K full-row startup check failed")
	}
	rng := rand.New(rand.NewSource(6107))
	for _, cols := range []int{256, 512, 2560, 3072, 4096, 9216, 16384} {
		blocks := cols / 256
		for trial := range 32 {
			row := randomQ6KRow(rng, cols)
			x := randomVec(rng, cols)
			if trial == 0 {
				clear(x)
			}
			q8, scales := make([]int8, cols), make([]float32, blocks)
			q8kQuantize(x, q8, scales, blocks)
			var sums []float32
			fillQ6KXSums16(x, cols, &sums)
			ScaleF32(sums, 32)
			if trial == 1 {
				// Extreme signs test int32 products and signed scale loads;
				// cancellation makes changes in float accumulation visible.
				for i := range q8 {
					q8[i] = []int8{-128, 127}[i%2]
				}
				for b := range blocks {
					for i := range 16 {
						row[b*210+192+i] = []byte{128, 127}[i%2]
					}
				}
			}
			want := q6kDotQ8KRowBlock(row, q8, scales, sums, blocks)
			for name, got := range map[string]float32{
				"assembly": q6kRowDecode(row, q8, scales, sums, blocks),
				"dispatch": q6kDotQ8KRow(row, q8, scales, sums, blocks),
			} {
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("%s cols=%d trial=%d: got %g (%08x), want %g (%08x)",
						name, cols, trial, got, math.Float32bits(got), want, math.Float32bits(want))
				}
			}
		}
	}
}

func TestQ6KDecodeEmptyAndBounds(t *testing.T) {
	requireDotProd(t)
	if got := q6kRowDecode(nil, nil, nil, nil, 0); got != 0 {
		t.Fatalf("empty row = %v", got)
	}
	for _, short := range []string{"row", "q8", "scales", "sums", "overflow"} {
		t.Run(short, func(t *testing.T) {
			row, q8 := make([]byte, 420), make([]int8, 512)
			scales, sums := make([]float32, 2), make([]float32, 32)
			blocks := 2
			switch short {
			case "row":
				row = row[:419]
			case "q8":
				q8 = q8[:511]
			case "scales":
				scales = scales[:1]
			case "sums":
				sums = sums[:31]
			case "overflow":
				blocks = int(^uint(0) >> 1)
			}
			defer func() {
				if recover() == nil {
					t.Fatal("expected a bounds panic before entering assembly")
				}
			}()
			q6kRowDecode(row, q8, scales, sums, blocks)
		})
	}
}

func TestQ6KDecodeScaleEdges(t *testing.T) {
	requireDotProd(t)
	rng := rand.New(rand.NewSource(6108))
	scaleBits := []uint16{0, 0x8000, 1, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff, 0xfbff}
	blocks := len(scaleBits)
	row := randomQ6KRow(rng, blocks*256)
	q8 := make([]int8, blocks*256)
	scales, sums := make([]float32, blocks), make([]float32, blocks*16)
	for i := range q8 {
		q8[i] = int8(i)
	}
	for i := range sums {
		sums[i] = float32(i%23-11) * 0.0625
	}
	for i, bits := range scaleBits {
		row[i*210+208], row[i*210+209] = byte(bits), byte(bits>>8)
		scales[i] = float32(i%5-2) * 0.01
	}
	got := q6kRowDecode(row, q8, scales, sums, blocks)
	want := q6kDotQ8KRowBlock(row, q8, scales, sums, blocks)
	if math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("edge scales got %g (%08x), want %g (%08x)", got, math.Float32bits(got), want, math.Float32bits(want))
	}
}

func TestQ6KDecodeFallback(t *testing.T) {
	requireDotProd(t)
	old := q6kDecodeAsmOK
	q6kDecodeAsmOK = false
	t.Cleanup(func() { q6kDecodeAsmOK = old })
	rng := rand.New(rand.NewSource(39))
	row := randomQ6KRow(rng, 512)
	q8, scales, sums := make([]int8, 512), []float32{0.03, 0.04}, make([]float32, 32)
	for i := range q8 {
		q8[i] = int8(i)
	}
	got := q6kDotQ8KRow(row, q8, scales, sums, 2)
	want := q6kDotQ8KRowBlock(row, q8, scales, sums, 2)
	if math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("fallback got %v want %v", got, want)
	}
}

func BenchmarkQ6KDecodeRow(b *testing.B) {
	if !q6kDecodeAsmOK {
		b.Skip("full-row SDOT unavailable")
	}
	for _, cols := range []int{3072, 9216} {
		b.Run(strconv.Itoa(cols), func(b *testing.B) {
			rng := rand.New(rand.NewSource(103))
			row, x := randomQ6KRow(rng, cols), randomVec(rng, cols)
			q8, scales := make([]int8, cols), make([]float32, cols/256)
			q8kQuantize(x, q8, scales, cols/256)
			var sums []float32
			fillQ6KXSums16(x, cols, &sums)
			ScaleF32(sums, 32)
			b.Run("block", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					q6kDotQ8KRowBlock(row, q8, scales, sums, cols/256)
				}
			})
			b.Run("row", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					q6kRowDecode(row, q8, scales, sums, cols/256)
				}
			})
		})
	}
}
