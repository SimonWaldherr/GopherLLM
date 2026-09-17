//go:build arm64

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Keep the previous block-at-a-time kernel as an independent compatibility
// oracle: decode optimizations must preserve every float32 output bit.
func q8_0DecodeLegacyRow(row []byte, q8 []int8, scales []float32, blocks int) float32 {
	var dots [8]int32
	var sum float32
	for b := range blocks {
		q8_0Q8Dots8Asm(&row[b*272], &q8[b*256], &dots[0])
		var blockSum float32
		for j := range 8 {
			blockSum += F16ToF32(binaryLE16(row[b*272+j*34:])) * float32(dots[j])
		}
		sum += scales[b] * blockSum
	}
	return sum
}

func TestQ8_0DecodeMatchesLegacy(t *testing.T) {
	requireDotProd(t)
	if !q8_0DecodeAsmOK {
		t.Fatal("decode self-check failed")
	}
	rng := rand.New(rand.NewSource(9102))
	for _, cols := range []int{256, 512, 2560, 3072, 4096, 9728} {
		blocks := cols / 256
		for trial := range 12 {
			row := randomQ8_0Row(rng, cols)
			q8 := make([]int8, cols)
			scales := make([]float32, blocks)
			q8kQuantize(randomVec(rng, cols), q8, scales, blocks)
			if trial == 0 {
				clear(q8)
				clear(scales)
			}
			got := q8_0RowDecode(row, q8, scales, blocks)
			want := q8_0DecodeLegacyRow(row, q8, scales, blocks)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("cols=%d trial=%d: got %g (%08x), want %g (%08x)", cols, trial, got, math.Float32bits(got), want, math.Float32bits(want))
			}
		}
	}
}

func TestQ8_0DecodeEmpty(t *testing.T) {
	for _, blocks := range []int{0, -1} {
		if got := q8_0RowDecode(nil, nil, nil, blocks); got != 0 {
			t.Fatalf("blocks=%d: empty row = %g, want 0", blocks, got)
		}
	}
}

func TestQ8_0DecodeRejectsShortBuffers(t *testing.T) {
	for _, tc := range []struct {
		name                            string
		rowLen, q8Len, scaleLen, blocks int
	}{
		{"empty", 0, 0, 0, 1},
		{"row", 543, 512, 2, 2},
		{"activations", 544, 511, 2, 2},
		{"scales", 544, 512, 1, 2},
		{"maximum count", 272, 256, 1, int(^uint(0) >> 1)},
		{"wrapped byte count", 272, 256, 1, 1<<60 + 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("invalid buffers reached assembly without a panic")
				}
			}()
			q8_0RowDecode(make([]byte, tc.rowLen), make([]int8, tc.q8Len), make([]float32, tc.scaleLen), tc.blocks)
		})
	}
}

func TestQ8_0DecodeScaleAndInt8Edges(t *testing.T) {
	requireDotProd(t)
	// Cover the LUT's full index range, signed zero, subnormal scales, negative
	// scales, and both int8 endpoints rather than only ordinary model weights.
	const blocks = 3
	scaleBits := [...]uint16{0, 0x8000, 1, 0x03ff, 0x0400, 0x3555, 0xbc00, 0x7bff, 0xfbff}
	row := make([]byte, blocks*272)
	q8 := make([]int8, blocks*256)
	for j := range blocks * 8 {
		scale := scaleBits[j%len(scaleBits)]
		row[j*34], row[j*34+1] = byte(scale), byte(scale>>8)
		for i := range 32 {
			row[j*34+2+i] = byte(int8((j+i)%256 - 128))
			if i%2 == 0 {
				q8[j*32+i] = -128
			} else {
				q8[j*32+i] = 127
			}
		}
	}
	scales := []float32{0.03, 0, -0.025}
	got := q8_0RowDecode(row, q8, scales, blocks)
	want := q8_0DecodeLegacyRow(row, q8, scales, blocks)
	if math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("edge scales: got %g (%08x), want %g (%08x)", got, math.Float32bits(got), want, math.Float32bits(want))
	}
}

func TestQ8_0DecodeFallback(t *testing.T) {
	requireDotProd(t)
	if !q8_0DotAsmOK {
		t.Fatal("block dot self-check failed")
	}
	const cols = 512
	rng := rand.New(rand.NewSource(9104))
	row := randomQ8_0Row(rng, cols)
	q8 := make([]int8, cols)
	scales := make([]float32, cols/256)
	q8kQuantize(randomVec(rng, cols), q8, scales, cols/256)
	want := q8_0DecodeLegacyRow(row, q8, scales, cols/256)
	previous := q8_0DecodeAsmOK
	q8_0DecodeAsmOK = false
	t.Cleanup(func() { q8_0DecodeAsmOK = previous })
	got := q8_0DotQ8KRow(row, q8, scales, cols/256)
	if math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("fallback = %g, want %g", got, want)
	}
}

var q8_0DecodeBenchResult float32

func BenchmarkQ8_0DecodeRow(b *testing.B) {
	if !q8_0DecodeAsmOK || !q8_0DotAsmOK {
		b.Skip("SDOT unavailable")
	}
	for _, cols := range []int{2560, 4096, 9728} {
		rng := rand.New(rand.NewSource(9103))
		blocks := cols / 256
		row := randomQ8_0Row(rng, cols)
		q8 := make([]int8, cols)
		scales := make([]float32, blocks)
		q8kQuantize(randomVec(rng, cols), q8, scales, blocks)
		b.Run(fmt.Sprintf("cols%d/legacy", cols), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				q8_0DecodeBenchResult = q8_0DecodeLegacyRow(row, q8, scales, blocks)
			}
		})
		b.Run(fmt.Sprintf("cols%d/fullrow", cols), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				q8_0DecodeBenchResult = q8_0RowDecode(row, q8, scales, blocks)
			}
		})
	}
}
