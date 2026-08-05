package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// TestPackScaleMinK4AllRoundTrip exhaustively checks packScaleMinK4All
// against the real decoder's getScaleMinK4 (simd.go:1658-1663) — this is
// the single most bug-prone piece of the Q4_K encoder (joint 6-bit packing
// across all 8 sub-blocks), so it gets its own focused test before being
// wired into QuantizeRowQ4K.
func TestPackScaleMinK4AllRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		var sc, m [8]byte
		for i := range sc {
			sc[i] = byte(rng.Intn(64))
			m[i] = byte(rng.Intn(64))
		}
		packed := packScaleMinK4All(sc, m)
		for j := 0; j < 8; j++ {
			gotSc, gotM := getScaleMinK4(j, packed[:])
			if gotSc != sc[j] || gotM != m[j] {
				t.Fatalf("trial %d, j=%d: got (sc=%d,m=%d), want (sc=%d,m=%d); sc=%v m=%v packed=%v",
					trial, j, gotSc, gotM, sc[j], m[j], sc, m, packed)
			}
		}
	}
}

// TestPackScaleMinK4AllBoundaryValues checks the 0 and 63 extremes
// specifically, since off-by-one shift/mask errors tend to show up at the
// edges of a value range even when random-value fuzzing passes.
func TestPackScaleMinK4AllBoundaryValues(t *testing.T) {
	cases := [][8]byte{
		{0, 0, 0, 0, 0, 0, 0, 0},
		{63, 63, 63, 63, 63, 63, 63, 63},
		{63, 0, 63, 0, 63, 0, 63, 0},
		{0, 63, 0, 63, 0, 63, 0, 63},
	}
	for _, sc := range cases {
		for _, m := range cases {
			packed := packScaleMinK4All(sc, m)
			for j := 0; j < 8; j++ {
				gotSc, gotM := getScaleMinK4(j, packed[:])
				if gotSc != sc[j] || gotM != m[j] {
					t.Fatalf("j=%d: got (sc=%d,m=%d), want (sc=%d,m=%d); sc=%v m=%v", j, gotSc, gotM, sc[j], m[j], sc, m)
				}
			}
		}
	}
}

// quantDequantMaxErr quantizes x with quantize, dequantizes the result with
// the real, trusted DequantRow* decoder, and returns the max absolute
// error alongside the dequantized values for inspection.
func quantDequantMaxErr(t *testing.T, x []float32, quantize func([]float32, int) []byte, dequant func([]byte, int) []float32) (float32, []float32) {
	t.Helper()
	cols := len(x)
	packed := quantize(x, cols)
	deq := dequant(packed, cols)
	if len(deq) != cols {
		t.Fatalf("dequantized length %d, want %d", len(deq), cols)
	}
	var maxErr float32
	for i, v := range x {
		e := abs32(v - deq[i])
		if e > maxErr {
			maxErr = e
		}
	}
	return maxErr, deq
}

func randomRow(rng *rand.Rand, n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = (rng.Float32()*2 - 1) * scale
	}
	return out
}

func TestQuantizeRowQ8_0RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 20; trial++ {
		x := randomRow(rng, 256, 1.0+float32(trial))
		maxErr, _ := quantDequantMaxErr(t, x, QuantizeRowQ8_0, DequantRowQ8_0)
		// Symmetric int8 over 32-block absmax: max quantization step is
		// amax/127, error must be within half a step plus a small margin.
		bound := (1.0 + float32(trial)) / 127 * 0.6
		if maxErr > bound {
			t.Fatalf("trial %d: max error %v exceeds bound %v", trial, maxErr, bound)
		}
	}
}

func TestQuantizeRowQ8_0KnownValues(t *testing.T) {
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32(i) - 16 // -16..15
	}
	packed := QuantizeRowQ8_0(x, 32)
	deq := DequantRowQ8_0(packed, 32)
	// amax=16, so one quantization step is 16/127 =~ 0.126; half a step plus
	// a little slack for f16-rounding the scale itself comfortably covers it.
	for i := range x {
		if abs32(deq[i]-x[i]) > 0.08 {
			t.Fatalf("i=%d: got %v, want ~%v", i, deq[i], x[i])
		}
	}
}

func TestQuantizeRowQ4_0RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 20; trial++ {
		x := randomRow(rng, 256, 1.0+float32(trial))
		maxErr, _ := quantDequantMaxErr(t, x, QuantizeRowQ4_0, DequantRowQ4_0)
		// Q4_0's code-8 range is asymmetric (-8..+7): the signed-scale trick
		// (see QuantizeRowQ4_0's doc comment) guarantees only the block's
		// single most extreme value avoids clipping. A block with both a
		// near-max-magnitude positive AND negative value (common in random
		// data) clips the other one by up to a full step, amax/8 — this is
		// an inherent format property, not an encoder bug, so the bound
		// must cover a full step, not half.
		bound := (1.0+float32(trial))/8 + 0.02
		if maxErr > bound {
			t.Fatalf("trial %d: max error %v exceeds bound %v", trial, maxErr, bound)
		}
	}
}

func TestQuantizeRowQ4KRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for trial := 0; trial < 20; trial++ {
		x := randomRow(rng, 256, 1.0+float32(trial)*0.3)
		maxErr, deq := quantDequantMaxErr(t, x, QuantizeRowQ4K, DequantRowQ4K)
		for i, v := range deq {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("trial %d, i=%d: dequantized to non-finite %v", trial, i, v)
			}
		}
		// 4-bit code over a per-32-sub-block range further coarsened by
		// 6-bit scale/min quantization: generous but not unbounded.
		bound := (1.0 + float32(trial)*0.3) * 0.35
		if maxErr > bound {
			t.Fatalf("trial %d: max error %v exceeds bound %v", trial, maxErr, bound)
		}
	}
}

func TestQuantizeRowQ4KAllSameValue(t *testing.T) {
	// A degenerate all-equal block (max==min for every sub-block, so raw
	// scale is 0 for all of them) must not divide by zero or NaN out.
	x := make([]float32, 256)
	for i := range x {
		x[i] = 0.5
	}
	packed := QuantizeRowQ4K(x, 256)
	deq := DequantRowQ4K(packed, 256)
	for i, v := range deq {
		if math.IsNaN(float64(v)) {
			t.Fatalf("i=%d: got NaN", i)
		}
		if abs32(v-0.5) > 0.05 {
			t.Fatalf("i=%d: got %v, want ~0.5", i, v)
		}
	}
}

func TestQuantizeRowQ4KAllZero(t *testing.T) {
	x := make([]float32, 256)
	packed := QuantizeRowQ4K(x, 256)
	deq := DequantRowQ4K(packed, 256)
	for i, v := range deq {
		if math.IsNaN(float64(v)) || v != 0 {
			t.Fatalf("i=%d: got %v, want 0", i, v)
		}
	}
}

func TestQuantizeRowQ5KRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for trial := 0; trial < 20; trial++ {
		x := randomRow(rng, 256, 1.0+float32(trial)*0.3)
		maxErr, deq := quantDequantMaxErr(t, x, QuantizeRowQ5K, DequantRowQ5K)
		for i, v := range deq {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("trial %d, i=%d: dequantized to non-finite %v", trial, i, v)
			}
		}
		// 5-bit code (finer than Q4_K's 4-bit) over the same double-quantized
		// (superblock d/dmin, then 6-bit sub-block scale/min) scheme.
		bound := (1.0+float32(trial)*0.3)*0.2 + 0.02
		if maxErr > bound {
			t.Fatalf("trial %d: max error %v exceeds bound %v", trial, maxErr, bound)
		}
	}
}

func TestQuantizeRowQ5KAllSameValue(t *testing.T) {
	x := make([]float32, 256)
	for i := range x {
		x[i] = 0.5
	}
	packed := QuantizeRowQ5K(x, 256)
	deq := DequantRowQ5K(packed, 256)
	for i, v := range deq {
		if math.IsNaN(float64(v)) {
			t.Fatalf("i=%d: got NaN", i)
		}
		if abs32(v-0.5) > 0.05 {
			t.Fatalf("i=%d: got %v, want ~0.5", i, v)
		}
	}
}

func TestQuantizeRowQ5KAllZero(t *testing.T) {
	x := make([]float32, 256)
	packed := QuantizeRowQ5K(x, 256)
	deq := DequantRowQ5K(packed, 256)
	for i, v := range deq {
		if math.IsNaN(float64(v)) || v != 0 {
			t.Fatalf("i=%d: got %v, want 0", i, v)
		}
	}
}

func TestQuantizeRowQ6KRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := 0; trial < 20; trial++ {
		x := randomRow(rng, 256, 1.0+float32(trial)*0.3)
		maxErr, deq := quantDequantMaxErr(t, x, QuantizeRowQ6K, DequantRowQ6K)
		for i, v := range deq {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("trial %d, i=%d: dequantized to non-finite %v", trial, i, v)
			}
		}
		// Two stacked quantization steps here (superblock-wide d, then each
		// sub-block's own int8 multiplier), so the effective step is
		// coarser than a single-scale format's amax/62 — generous margin.
		bound := (1.0+float32(trial)*0.3)*0.25 + 0.02
		if maxErr > bound {
			t.Fatalf("trial %d: max error %v exceeds bound %v", trial, maxErr, bound)
		}
	}
}

func TestQuantizeRowQ6KAllZero(t *testing.T) {
	x := make([]float32, 256)
	packed := QuantizeRowQ6K(x, 256)
	deq := DequantRowQ6K(packed, 256)
	for i, v := range deq {
		if math.IsNaN(float64(v)) || v != 0 {
			t.Fatalf("i=%d: got %v, want 0", i, v)
		}
	}
}

// TestQuantizeRowSizes confirms every encoder's output length matches
// GGMLType.DataSize, which the GGUF writer relies on for the tensor
// descriptor table's cumulative offsets.
func TestQuantizeRowSizes(t *testing.T) {
	cases := []struct {
		name      string
		cols      int
		dtype     GGMLType
		quantize  func([]float32, int) []byte
	}{
		{"Q8_0", 256, GGMLTypeQ8_0, QuantizeRowQ8_0},
		{"Q4_0", 256, GGMLTypeQ4_0, QuantizeRowQ4_0},
		{"Q4_K", 512, GGMLTypeQ4_K, QuantizeRowQ4K},
		{"Q5_K", 512, GGMLTypeQ5_K, QuantizeRowQ5K},
		{"Q6_K", 512, GGMLTypeQ6_K, QuantizeRowQ6K},
	}
	for _, c := range cases {
		want, ok := c.dtype.DataSize(c.cols)
		if !ok {
			t.Fatalf("%s: DataSize(%d) not ok", c.name, c.cols)
		}
		got := len(c.quantize(make([]float32, c.cols), c.cols))
		if got != want {
			t.Fatalf("%s: encoder produced %d bytes, DataSize says %d", c.name, got, want)
		}
	}
}
