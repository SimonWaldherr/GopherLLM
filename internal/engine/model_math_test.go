package engine

import (
	"math"
	"math/rand"
	"testing"
)

func TestFinite32MatchesFloatClassification(t *testing.T) {
	bits := []uint32{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
		0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
		0x7fc00000, 0xffc00000, 0x7fffffff, 0xffffffff}
	rng := rand.New(rand.NewSource(1))
	for range 1024 {
		bits = append(bits, rng.Uint32())
	}
	for _, word := range bits {
		value := math.Float32frombits(word)
		want := !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
		if got := finite32(value); got != want {
			t.Fatalf("finite32(%#08x) = %v, want %v", word, got, want)
		}
	}
}

func TestExactSiLURounding(t *testing.T) {
	// Fixed float32 results from the original division formula. These catch
	// substitution of x*sigmoid(x), including signed-zero/overflow behavior.
	for _, tc := range []struct {
		input float32
		bits  uint32
	}{
		{0, 0}, {math.Float32frombits(0x80000000), 0x80000000},
		{1, 0x3f3b26a8}, {-1, 0xbe89b2b1}, {5, 0x409eeddc}, {-5, 0xbd0911d0},
		{20, 0x41a00000}, {-20, 0xb3310d40}, {80, 0x42a00000}, {-80, 0x88efe7e9},
		{100, 0x42c80000}, {-100, 0x80000000},
		{1e-30, 0x0d224260}, {-1e-30, 0x8d224260},
		{float32(math.Inf(1)), 0x7f800000},
	} {
		if got := math.Float32bits(siluExact(tc.input)); got != tc.bits {
			t.Fatalf("silu(%g) = %#08x, want %#08x", tc.input, got, tc.bits)
		}
	}
	for _, value := range []float32{float32(math.Inf(-1)), float32(math.NaN())} {
		if !math.IsNaN(float64(siluExact(value))) {
			t.Fatalf("silu(%g) must preserve non-finite result", value)
		}
	}
}

func TestSoftmaxF64Rounding(t *testing.T) {
	// A long row distinguishes float64 accumulation from float32 accumulation:
	// its denominator rounds to 1805.0516357421875 rather than 1805.0693359375.
	want := [...]uint32{
		0x389d3c5a, 0x38b22be5, 0x38c9e50c, 0x38e4c6d4, 0x39019e75,
		0x3912e0a6, 0x39266f1e, 0x393c9830, 0x3955b49f, 0x397228fd,
		0x398933a1, 0x399b784a, 0x39b02ba2, 0x39c7a095, 0x39e23513,
		0x3a0029ca, 0x3a113a5d,
	}
	for _, useLayaMaximum := range []bool{false, true} {
		values := make([]float32, 4096)
		for i := range values {
			values[i] = float32(i%17)*.125 - 1
		}
		if useLayaMaximum {
			layaSoftmax(values)
		} else {
			softmaxF64InPlace(values, 1)
		}
		for i, value := range values {
			if got := math.Float32bits(value); got != want[i%len(want)] {
				t.Fatalf("Laya maximum=%v: output[%d] = %#08x, want %#08x", useLayaMaximum, i, got, want[i%len(want)])
			}
		}
	}
}

func TestSoftmaxF64Edges(t *testing.T) {
	softmaxF64InPlace(nil, 0)
	for _, values := range [][]float32{
		{float32(math.Inf(1)), 1},
		{float32(math.Inf(-1)), float32(math.Inf(-1))},
		{float32(math.NaN()), 0},
	} {
		layaSoftmax(values)
		for _, value := range values {
			if !math.IsNaN(float64(value)) {
				t.Fatal("non-finite softmax behavior changed")
			}
		}
	}
	values := []float32{-1000, 0, 1000}
	softmaxF64InPlace(values, 1000)
	for i, want := range []uint32{0, 0, 0x3f800000} {
		if math.Float32bits(values[i]) != want {
			t.Fatal("softmax underflow or normalization changed")
		}
	}
}
