package gopherllm

import (
	"math"
	"testing"
)

// These measure the error of the fast float32 transcendentals against the
// float64 libm functions they replace, rather than asserting a bound and hoping.
// The thresholds are the repo's existing activation tolerance (1e-5) with a lot
// of headroom; the t.Logf lines report what the error actually is so a
// regression shows up as a number, not just a pass.

func TestFastExpF32Accuracy(t *testing.T) {
	var maxRel float64
	var worstAt float64
	// Dense sweep over the range activations actually reach, plus the tails.
	for x := -40.0; x <= 40.0; x += 0.00037 {
		xf := float32(x)
		got := float64(fastExpF32(xf))
		// exp of the value the function actually receives. Comparing against
		// math.Exp(x) instead measures the float32 rounding of the INPUT, which
		// at x=33 is a 1.9e-6 relative shift in exp and swamps the kernel error.
		want := math.Exp(float64(xf))
		rel := math.Abs(got-want) / want
		if rel > maxRel {
			maxRel, worstAt = rel, x
		}
	}
	t.Logf("fastExpF32 max relative error %.3g (at x=%.4f)", maxRel, worstAt)
	// One float32 ulp is 1.2e-7; allow a small multiple.
	if maxRel > 1e-6 {
		t.Errorf("fastExpF32 max relative error %g at x=%v exceeds 1e-6", maxRel, worstAt)
	}
}

func TestFastExpF32Extremes(t *testing.T) {
	if got := fastExpF32(0); got != 1 {
		t.Errorf("fastExpF32(0) = %v, want exactly 1", got)
	}
	if got := fastExpF32(200); !math.IsInf(float64(got), 1) {
		t.Errorf("fastExpF32(200) = %v, want +Inf", got)
	}
	if got := fastExpF32(-200); got != 0 {
		t.Errorf("fastExpF32(-200) = %v, want 0", got)
	}
	// Must never produce NaN anywhere in the clamped band.
	for x := -90.0; x <= 89.0; x += 0.01 {
		if v := fastExpF32(float32(x)); math.IsNaN(float64(v)) {
			t.Fatalf("fastExpF32(%v) = NaN", x)
		}
	}
}

func TestFastSigmoidF32Accuracy(t *testing.T) {
	var maxAbs float64
	var worstAt float64
	for x := -30.0; x <= 30.0; x += 0.00029 {
		xf := float32(x)
		got := float64(fastSigmoidF32(xf))
		want := 1 / (1 + math.Exp(-float64(xf)))
		if d := math.Abs(got - want); d > maxAbs {
			maxAbs, worstAt = d, x
		}
	}
	t.Logf("fastSigmoidF32 max absolute error %.3g (at x=%.4f)", maxAbs, worstAt)
	if maxAbs > 1e-7 {
		t.Errorf("fastSigmoidF32 max absolute error %g at x=%v exceeds 1e-7", maxAbs, worstAt)
	}
}

func TestFastTanhF32Accuracy(t *testing.T) {
	var maxAbs, maxRel float64
	var worstAt float64
	for x := -12.0; x <= 12.0; x += 0.00017 {
		xf := float32(x)
		got := float64(fastTanhF32(xf))
		want := math.Tanh(float64(xf))
		if d := math.Abs(got - want); d > maxAbs {
			maxAbs, worstAt = d, x
		}
		if want != 0 {
			if r := math.Abs(got-want) / math.Abs(want); r > maxRel {
				maxRel = r
			}
		}
	}
	t.Logf("fastTanhF32 max absolute error %.3g, max relative %.3g (at x=%.4f)", maxAbs, maxRel, worstAt)
	// tanh's output lives in [-1, 1], where one float32 ulp is 1.19e-7, so a
	// bound below that would be asking for more precision than the return type
	// carries. Two ulp is the meaningful limit here.
	if maxAbs > 2.4e-7 {
		t.Errorf("fastTanhF32 max absolute error %g at x=%v exceeds two float32 ulp", maxAbs, worstAt)
	}
	// Relative accuracy near zero is what the naive 2*sigmoid(2x)-1 form loses;
	// the small-|x| series is there to keep it.
	if maxRel > 1e-6 {
		t.Errorf("fastTanhF32 max relative error %g exceeds 1e-6", maxRel)
	}
	if fastTanhF32(0) != 0 {
		t.Errorf("fastTanhF32(0) = %v, want exactly 0", fastTanhF32(0))
	}
	// Odd symmetry, which softcapF32's test depends on.
	for _, x := range []float32{0.1, 0.5, 1, 3, 8, 20} {
		if p, n := fastTanhF32(x), fastTanhF32(-x); p != -n {
			t.Errorf("tanh not odd at %v: %v vs %v", x, p, n)
		}
	}
}

// geluTanhScalar is a rewrite of 0.5x(1+tanh(u)) into x*sigmoid(2u); this pins
// that the algebra is right against the float64 form it replaces.
func TestGeluTanhScalarMatchesReferenceForm(t *testing.T) {
	reference := func(x float64) float64 {
		const sqrt2OverPi = 0.7978845608028654
		return 0.5 * x * (1 + math.Tanh(sqrt2OverPi*(x+0.044715*x*x*x)))
	}
	var maxAbs float64
	var worstAt float64
	for x := -20.0; x <= 20.0; x += 0.00023 {
		xf := float32(x)
		got := float64(geluTanhScalar(xf))
		want := reference(float64(xf))
		if d := math.Abs(got - want); d > maxAbs {
			maxAbs, worstAt = d, x
		}
	}
	t.Logf("geluTanhScalar max absolute error %.3g (at x=%.4f)", maxAbs, worstAt)
	if maxAbs > 1e-5 {
		t.Errorf("geluTanhScalar max absolute error %g at x=%v exceeds 1e-5", maxAbs, worstAt)
	}
}

func TestGeluMulF32MatchesScalarLoop(t *testing.T) {
	const n = 1024
	gate := make([]float32, n)
	up := make([]float32, n)
	for i := range gate {
		gate[i] = float32(i%211)/13 - 8
		up[i] = float32(i%97)/11 - 4
	}
	got := make([]float32, n)
	geluMulF32(gate, up, got)
	for i := range n {
		want := geluTanhScalar(gate[i]) * up[i]
		if got[i] != want {
			t.Fatalf("i=%d: geluMulF32 %v != %v", i, got[i], want)
		}
	}
	// Short output slice must bound the work, not panic.
	short := make([]float32, 7)
	geluMulF32(gate, up, short)
}

func BenchmarkExpF32(b *testing.B) {
	xs := make([]float32, 4096)
	for i := range xs {
		xs[i] = float32(i%211)/13 - 8
	}
	b.Run("mathExp_float64", func(b *testing.B) {
		var s float32
		for range b.N {
			for _, x := range xs {
				s += float32(math.Exp(float64(x)))
			}
		}
		_ = s
	})
	b.Run("fastExpF32", func(b *testing.B) {
		var s float32
		for range b.N {
			for _, x := range xs {
				s += fastExpF32(x)
			}
		}
		_ = s
	})
}

func BenchmarkTanhF32(b *testing.B) {
	xs := make([]float32, 4096)
	for i := range xs {
		xs[i] = float32(i%211)/29 - 3.6
	}
	b.Run("mathTanh_float64", func(b *testing.B) {
		var s float32
		for range b.N {
			for _, x := range xs {
				s += float32(math.Tanh(float64(x)))
			}
		}
		_ = s
	})
	b.Run("fastTanhF32", func(b *testing.B) {
		var s float32
		for range b.N {
			for _, x := range xs {
				s += fastTanhF32(x)
			}
		}
		_ = s
	})
}

// The shape that actually runs per token: a d_ff-sized GeGLU activation.
func BenchmarkGeluMul(b *testing.B) {
	const n = 8192
	gate := make([]float32, n)
	up := make([]float32, n)
	out := make([]float32, n)
	for i := range gate {
		gate[i] = float32(i%211)/13 - 8
		up[i] = float32(i%97)/11 - 4
	}
	b.Run("scalar_mathTanh", func(b *testing.B) {
		for range b.N {
			for i := range n {
				const sqrt2OverPi = 0.7978845608028654
				x := gate[i]
				out[i] = 0.5 * x * (1 + float32(math.Tanh(sqrt2OverPi*float64(x+0.044715*x*x*x)))) * up[i]
			}
		}
	})
	b.Run("geluMulF32", func(b *testing.B) {
		for range b.N {
			geluMulF32(gate, up, out)
		}
	})
}

func BenchmarkSiluMulScalar(b *testing.B) {
	const n = 8192
	gate := make([]float32, n)
	up := make([]float32, n)
	out := make([]float32, n)
	for i := range gate {
		gate[i] = float32(i%211)/13 - 8
		up[i] = float32(i%97)/11 - 4
	}
	b.Run("mathExp_float64", func(b *testing.B) {
		for range b.N {
			for i := range n {
				g := gate[i]
				out[i] = (g / (1 + float32(math.Exp(float64(-g))))) * up[i]
			}
		}
	})
	b.Run("fastSigmoid", func(b *testing.B) {
		for range b.N {
			siluMulF32Scalar(gate, up, out, 0, n)
		}
	})
}
