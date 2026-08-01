//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// The portable Q8K kernels in quant_q8k_portable.go are the fallback on every
// architecture without a hand-written int8 kernel, and the oracle that such a
// kernel is differentially tested against. That makes their arithmetic
// load-bearing on targets this test cannot itself run on, so it is pinned here
// against both the scalar references and the AVX2 assembly that already guard
// each other on amd64.
//
// Data comes from the same randomQ*Row / randomVec builders and the same
// fillQ*XSums helpers the pre-existing kernel suites use, so rows carry
// realistic f16 scales. Feeding uniformly random bytes instead produces NaN and
// Inf block scales, which makes every comparison vacuously "equal" under an
// exact check and vacuously unequal under NaN != NaN.
//
// Tolerance matches the house rule in q4k_q8_amd64_test.go: relative 1e-3. The
// portable kernels and the vector kernels accumulate in a different order, so
// exact equality is not the contract — agreement well inside f32 rounding is.
func closeEnoughQ8K(t *testing.T, label string, got, want float32) {
	t.Helper()
	if math.IsNaN(float64(got)) || math.IsNaN(float64(want)) {
		t.Fatalf("%s: got %v, want %v — NaN means the test data is degenerate, not that the kernels agree", label, got, want)
	}
	if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
		t.Fatalf("%s: got %v, want %v (diff %v)", label, got, want, diff)
	}
}

// q8kTestInputs builds activations, their Q8K quantization, and both xsums
// flavours the row kernels need.
func q8kTestInputs(rng *rand.Rand, cols int) (q8 []int8, xsc, xsums32, xsums16 []float32) {
	x := randomVec(rng, cols)
	// Exercise the zero-block branch the way the existing suites do.
	for i := 256; i < min(512, cols); i++ {
		x[i] = 0
	}
	q8, xsc = quantizeQ8KRef(x, cols)
	var s1, s2 []float32
	xsums32 = fillQ4KXSums(x, cols, &s1)
	xsums16 = fillQ6KXSums16(x, cols, &s2)
	// Q6_K folds its -32 quant offset in by pre-scaling the per-16 sums.
	xsums16 = append([]float32(nil), xsums16...)
	ScaleF32(xsums16, 32)
	return q8, xsc, xsums32, xsums16
}

var q8kTestCols = []int{256, 1024, 3072, 4096}

// absF32 and roundToEvenF32 replace float64 libm calls in the quantizer's inner
// loop, so they have to be bit-identical to what they replace — not merely
// close. A discrepancy would shift a quantized activation by one step, which is
// invisible in aggregate and impossible to debug later.
func TestFloat32HelpersMatchFloat64Forms(t *testing.T) {
	check := func(v float32) {
		t.Helper()
		if got, want := absF32(v), float32(math.Abs(float64(v))); math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("absF32(%v) = %v, want %v", v, got, want)
		}
		// Only defined for the magnitudes the quantizer actually produces.
		if a := math.Abs(float64(v)); a < 4194304 {
			if got, want := roundToEvenF32(v), float32(math.RoundToEven(float64(v))); math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("roundToEvenF32(%v) = %v, want %v", v, got, want)
			}
		}
	}
	// Exact ties in both directions are the whole point of round-to-EVEN.
	for _, v := range []float32{
		0, -0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5,
		126.5, -126.5, 127, -127, 127.4999, -127.4999,
		1, -1, 0.4999999, -0.4999999,
	} {
		check(v)
	}
	// Then a dense sweep across the quantizer's output range.
	rng := rand.New(rand.NewSource(4711))
	for range 200000 {
		check(rng.Float32()*254 - 127)
	}
	// And every representable eighth, to hit tie cases systematically.
	for i := -127 * 8; i <= 127*8; i++ {
		check(float32(i) / 8)
	}
}

func BenchmarkQ8KQuantizePortable(b *testing.B) {
	const cols = 4096
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%977)/311 - 1.5
	}
	q8 := make([]int8, cols)
	sc := make([]float32, cols/256)
	b.Run("float32_helpers", func(b *testing.B) {
		for range b.N {
			q8kQuantizePortable(x, q8, sc, cols/256)
		}
	})
	b.Run("float64_libm", func(b *testing.B) {
		blocks := cols / 256
		for range b.N {
			for bl := range blocks {
				xb := x[bl*256 : bl*256+256]
				var amax float32
				for _, v := range xb {
					if a := float32(math.Abs(float64(v))); a > amax {
						amax = a
					}
				}
				if amax == 0 || amax != amax {
					continue
				}
				sc[bl] = amax / 127
				inv := 127 / amax
				qb := q8[bl*256 : bl*256+256]
				for i, v := range xb {
					qb[i] = int8(math.RoundToEven(float64(v * inv)))
				}
			}
		}
	})
}

func TestQ8KQuantizePortableMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(9001))
	for _, cols := range q8kTestCols {
		x := randomVec(rng, cols)
		for i := 256; i < min(512, cols); i++ {
			x[i] = 0
		}
		blocks := cols / 256
		gotQ8 := make([]int8, cols)
		gotScales := make([]float32, blocks)
		q8kQuantizePortable(x, gotQ8, gotScales, blocks)
		wantQ8, wantScales := quantizeQ8KRef(x, cols)
		// The quantizer is pure integer rounding of a scaled float, so this one
		// genuinely is exact on both sides.
		for b := range blocks {
			if gotScales[b] != wantScales[b] {
				t.Fatalf("cols=%d block %d scale = %v, want %v", cols, b, gotScales[b], wantScales[b])
			}
		}
		for i := range cols {
			if gotQ8[i] != wantQ8[i] {
				t.Fatalf("cols=%d q8[%d] = %d, want %d", cols, i, gotQ8[i], wantQ8[i])
			}
		}
	}
}

// A zero block must yield a zero scale rather than dividing by zero, and an
// all-NaN block must not poison the scale either: both occur on real models
// (padding rows, dead MoE experts).
func TestQ8KQuantizePortableDegenerateBlocks(t *testing.T) {
	allNaN := make([]float32, 256)
	for i := range allNaN {
		allNaN[i] = float32(math.NaN())
	}
	for name, x := range map[string][]float32{
		"zeros":   make([]float32, 256),
		"all NaN": allNaN,
	} {
		q8 := make([]int8, 256)
		for i := range q8 {
			q8[i] = 127 // poison, so a missing write is visible
		}
		scales := []float32{-1}
		q8kQuantizePortable(x, q8, scales, 1)
		if scales[0] != 0 {
			t.Errorf("%s: scale = %v, want 0", name, scales[0])
		}
		for i, v := range q8 {
			if v != 0 {
				t.Fatalf("%s: q8[%d] = %d, want 0", name, i, v)
			}
		}
	}
}

// Each case asserts portable == scalar reference AND portable == vector
// assembly, over the same inputs. The pre-existing suites already assert
// assembly == reference; checking both legs here means a future change that
// keeps any two of the three self-consistent still trips.
func TestPortableQ8KRowKernels(t *testing.T) {
	asm := hasAVX2 && hasF16C
	if !asm {
		t.Log("AVX2+F16C absent: checking portable against the scalar references only")
	}
	rng := rand.New(rand.NewSource(9002))
	for _, cols := range q8kTestCols {
		blocks := cols / 256
		q8, xsc, xsums32, xsums16 := q8kTestInputs(rng, cols)
		p8, ps, p32, p16 := &q8[0], &xsc[0], &xsums32[0], &xsums16[0]

		t.Run("Q4_K", func(t *testing.T) {
			row := randomQ4KRow(rng, cols)
			got := q4kDotQ8KRowPortable(row, q8, xsc, xsums32, blocks)
			closeEnoughQ8K(t, "portable vs ref", got, dotQ4KQ8KRowRef(row, q8, xsc, xsums32, blocks))
			if asm {
				closeEnoughQ8K(t, "portable vs asm", got, q4kDotQ8KRow(&row[0], p8, ps, p32, blocks))
			}
		})
		t.Run("Q5_K", func(t *testing.T) {
			row := randomQ5KRow(rng, cols)
			got := q5kDotQ8KRowPortable(row, q8, xsc, xsums32, blocks)
			closeEnoughQ8K(t, "portable vs ref", got, dotQ5KQ8KRowRef(row, q8, xsc, xsums32, blocks))
			if asm {
				closeEnoughQ8K(t, "portable vs asm", got, q5kDotQ8KRow(&row[0], p8, ps, p32, blocks))
			}
		})
		t.Run("Q6_K", func(t *testing.T) {
			row := randomQ6KRow(rng, cols)
			got := q6kDotQ8KRowPortable(row, q8, xsc, xsums16, blocks)
			closeEnoughQ8K(t, "portable vs ref", got, dotQ6KQ8KRowRef(row, q8, xsc, xsums16, blocks))
			if asm {
				closeEnoughQ8K(t, "portable vs asm", got, q6kDotQ8KRow(&row[0], p8, ps, p16, blocks))
			}
		})
		t.Run("Q8_0", func(t *testing.T) {
			row := randomQ8_0Row(rng, cols)
			got := q8_0DotQ8KRowPortable(row, q8, xsc, blocks)
			closeEnoughQ8K(t, "portable vs ref", got, dotQ8_0Q8KRowRef(row, q8, xsc, blocks))
			if asm {
				closeEnoughQ8K(t, "portable vs asm", got, q8_0DotQ8KRow(&row[0], p8, ps, blocks))
			}
		})
		t.Run("Q4_0", func(t *testing.T) {
			row := randomQ4_0Row(rng, cols)
			got := q4_0DotQ8KRowPortable(row, q8, xsc, xsums32, blocks)
			closeEnoughQ8K(t, "portable vs ref", got, dotQ4_0Q8KRowRef(row, q8, xsc, xsums32, blocks))
			if asm {
				closeEnoughQ8K(t, "portable vs asm", got, q4_0DotQ8KRow(&row[0], p8, ps, p32, blocks))
			}
		})
		t.Run("Q4_1", func(t *testing.T) {
			row := randomQ4_1Row(rng, cols)
			got := q4_1DotQ8KRowPortable(row, q8, xsc, xsums32, blocks)
			closeEnoughQ8K(t, "portable vs ref", got, dotQ4_1Q8KRowRef(row, q8, xsc, xsums32, blocks))
			if asm {
				closeEnoughQ8K(t, "portable vs asm", got, q4_1DotQ8KRow(&row[0], p8, ps, p32, blocks))
			}
		})
		t.Run("MXFP4", func(t *testing.T) {
			row := randomMXFP4Row(rng, cols)
			got := mxfp4DotQ8KRowPortable(row, q8, xsc, blocks)
			closeEnoughQ8K(t, "portable vs ref", got, dotMXFP4Q8KRowRef(row, q8, xsc, blocks))
			if asm {
				closeEnoughQ8K(t, "portable vs asm", got, mxfp4DotQ8KRow(&row[0], p8, ps, blocks))
			}
		})
	}
}
