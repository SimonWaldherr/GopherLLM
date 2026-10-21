package gopherllm

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestQuantizeLegacy5BitPlanes(t *testing.T) {
	// Exact integer steps exercise both halves of the split fifth-bit plane.
	row := make([]float32, 64)
	for i := range row {
		row[i] = float32(i%32 - 16)
	}
	for _, tc := range []struct {
		name                              string
		encode                            func([]float32, int) []byte
		decode                            func([]byte, int) []float32
		blockBytes, highOffset, lowOffset int
	}{
		{"Q5_0", QuantizeRowQ5_0, DequantRowQ5_0, 22, 2, 6},
		{"Q5_1", QuantizeRowQ5_1, DequantRowQ5_1, 24, 4, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			encoded := tc.encode(row, len(row))
			if len(encoded) != 2*tc.blockBytes {
				t.Fatalf("encoded size %d", len(encoded))
			}
			for block := 0; block < 2; block++ {
				b := encoded[block*tc.blockBytes:]
				if high := binary.LittleEndian.Uint32(b[tc.highOffset:]); high != 0xffff0000 {
					t.Fatalf("fifth-bit plane = %#x", high)
				}
				for i := range 16 {
					if got, want := b[tc.lowOffset+i], byte(i|i<<4); got != want {
						t.Fatalf("nibbles[%d] = %#x, want %#x", i, got, want)
					}
				}
			}
			decoded := tc.decode(encoded, len(row))
			for i, want := range row {
				if decoded[i] != want {
					t.Fatalf("decoded[%d] = %g, want %g", i, decoded[i], want)
				}
			}
		})
	}
}

func TestQuantizeLegacy5RoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name   string
		encode func([]float32, int) []byte
		decode func([]byte, int) []float32
	}{
		{"Q5_0", QuantizeRowQ5_0, DequantRowQ5_0},
		{"Q5_1", QuantizeRowQ5_1, DequantRowQ5_1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, constant := range []float32{0, -0.25, 0.5} {
				row := make([]float32, 96)
				for i := range row {
					row[i] = constant
				}
				for i, got := range tc.decode(tc.encode(row, len(row)), len(row)) {
					if math.IsNaN(float64(got)) || abs32(got-constant) > 0.02 {
						t.Fatalf("constant %g at %d became %g", constant, i, got)
					}
				}
			}
			row := smallWeights(96, 103)
			decoded := tc.decode(tc.encode(row, len(row)), len(row))
			for i, want := range row {
				var maximum float32
				for _, v := range row[i/32*32 : (i/32+1)*32] {
					maximum = max(maximum, abs32(v))
				}
				// Signed Q5_0 has one fewer positive level than negative levels;
				// its clipped endpoint can be one step from the source value.
				if bound := maximum/16 + maximum/512; abs32(decoded[i]-want) > bound {
					t.Fatalf("error[%d] = %g exceeds %g", i, decoded[i]-want, bound)
				}
			}
		})
	}
}

func TestQuantizeF16Fallback(t *testing.T) {
	row := []float32{-1.25, 0, 0.125, 2.5, 0.001}
	encoded := quantizeRowF16(row, len(row))
	if len(encoded) != len(row)*2 {
		t.Fatal("wrong F16 byte count")
	}
	for i, want := range row {
		got := F16ToF32(binary.LittleEndian.Uint16(encoded[2*i:]))
		if abs32(got-want) > 1e-6 {
			t.Fatalf("F16[%d] = %g, want %g", i, got, want)
		}
	}
}
