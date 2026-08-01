package gopherllm

import "math"

// Fast float32 transcendentals for the activation paths.
//
// Every activation in this engine used to route through float64 libm: SwiGLU
// called math.Exp(float64(-g)) per element, Gemma's GeGLU called
// math.Tanh(float64(...)) per element, and Gemma's logit softcapping called
// math.Tanh once per vocabulary entry — 256k times per token on Gemma. Each of
// those is a float32 -> float64 widening, a branchy float64 routine, and a
// narrowing back, to produce a result that only ever needed float32 precision.
//
// These replacements work natively in float32 and are accurate to roughly one
// float32 ulp, which is far inside the 1e-5 tolerance the existing activation
// tests use and far inside the error the int8 activation path already
// introduces. fastmath_test.go measures the actual error against math.Exp and
// math.Tanh rather than asserting it.
//
// The shared observation that makes one kernel cover both model families:
//
//	silu(x)     = x * sigmoid(x)
//	geluTanh(x) = x * sigmoid(2*sqrt(2/pi) * (x + 0.044715*x^3))
//
// so SwiGLU (Mistral/Ministral/Qwen/SmolLM) and GeGLU (Gemma) are the same
// shape — an elementwise product with a sigmoid — and both reduce to one
// fastExpF32 call per element.

const (
	// invLn2 converts to units of 2^k.
	fmInvLn2 = 1.4426950408889634
	fmLn2    = 0.6931471805599453

	// Clamps that keep the 2^k exponent-field construction in range:
	// 127*ln2 and -126*ln2. Beyond them the result is Inf or flushes to zero
	// anyway.
	fmExpHi = 88.02969
	fmExpLo = -87.33654
)

// fastExpF32 returns e**x for float32, accurate to about one ulp.
//
// Standard range reduction: write x = k*ln2 + r with |r| <= ln2/2, evaluate
// e**r by a degree-6 series (its first dropped term is r^7/5040 <= 1.2e-7 over
// that interval, i.e. float32 epsilon), then scale by 2^k by writing the
// exponent field directly instead of calling a power function.
func fastExpF32(x float32) float32 {
	if x > fmExpHi {
		return float32(math.Inf(1))
	}
	if x < fmExpLo {
		return 0
	}
	// Round to nearest away from zero. Truncation would be off by one for
	// negative x, which would widen |r| to ln2 and cost two decimal digits.
	t := x * fmInvLn2
	var k int32
	if t >= 0 {
		k = int32(t + 0.5)
	} else {
		k = int32(t - 0.5)
	}

	// The subtraction runs in float64. Doing it in float32 — even with a hi/lo
	// split of ln2 — caps accuracy at about 2e-6 relative near the top of the
	// range, because x itself has a 2e-6 ulp at x=33 and r inherits that
	// absolute error while being ~100x smaller. One float64 multiply-subtract
	// buys back the full float32 precision and is still nothing next to a libm
	// call.
	r := float32(float64(x) - float64(k)*fmLn2)

	p := 1 + r*(1+r*(0.5+r*(1.0/6+r*(1.0/24+r*(1.0/120+r*(1.0/720))))))

	// 2^k as a bare exponent field. k is within [-126, 127] because of the
	// clamps above, so 127+k is a valid biased exponent.
	return p * math.Float32frombits(uint32(127+k)<<23)
}

// fastSigmoidF32 returns 1/(1+e**-x).
func fastSigmoidF32(x float32) float32 {
	// sigmoid(17) = 1 - 4.1e-8, which rounds to 1 in float32 (eps at 1.0 is
	// 6e-8), so saturating here is exact rather than approximate.
	if x >= 17 {
		return 1
	}
	if x <= fmExpLo {
		return 0
	}
	return 1 / (1 + fastExpF32(-x))
}

// fastTanhF32 returns tanh(x).
//
// The magnitude is computed from |x| and the sign reapplied at the end, which
// makes the result exactly odd. Evaluating 2*sigmoid(2x)-1 directly at both
// signs does NOT give exact oddness: 1/(1+e**y) and 1-1/(1+e**-y) are the same
// number in exact arithmetic but round differently, which showed up as
// tanh(1) = 0.76159406 against tanh(-1) = -0.7615942.
func fastTanhF32(x float32) float32 {
	a := x
	if a < 0 {
		a = -a
	}
	var y float32
	switch {
	case a > 9:
		// tanh(9) = 1 - 3e-8, below float32 resolution at 1.0.
		y = 1
	case a < 0.25:
		// Near zero, 2*sigmoid(2a)-1 subtracts two nearly equal numbers and
		// loses the leading digits, so use the odd series. Its first dropped
		// term at a = 0.25 is 8e-9.
		a2 := a * a
		y = a * (1 - a2*(1.0/3-a2*(2.0/15-a2*(17.0/315))))
	default:
		// (1-u)/(1+u) with u = e**-2a, not 2*sigmoid(2a)-1. Both are tanh in
		// exact arithmetic, but the sigmoid form subtracts 1 from a value near
		// 0.85 and doubles the residual, which measured 1.8e-7 of absolute
		// error. Here u is bounded well away from 1 for a >= 0.25, so neither
		// 1-u nor 1+u cancels, and the error drops to well under one ulp.
		u := fastExpF32(-2 * a)
		y = (1 - u) / (1 + u)
	}
	if x < 0 {
		return -y
	}
	return y
}

// geluTanhScalar is the tanh-approximation GELU (HuggingFace's
// gelu_pytorch_tanh), rewritten as x*sigmoid(...) so it costs one exponential
// instead of a tanh.
//
//	0.5*x*(1 + tanh(u)) == x*sigmoid(2u)
//
// with u = sqrt(2/pi)*(x + 0.044715x^3), so the folded constant is
// 2*sqrt(2/pi).
func geluTanhScalar(x float32) float32 {
	const twoSqrt2OverPi = 1.5957691216057308
	return x * fastSigmoidF32(twoSqrt2OverPi*(x+0.044715*x*x*x))
}

// geluMulF32 computes out[i] = geluTanh(gate[i]) * up[i], the GeGLU feed-forward
// activation Gemma uses where Llama-family models use SwiGLU.
//
// This existed only as an open-coded scalar loop repeated at five call sites,
// with no vectorized path on any architecture — unlike SwiGLU, which at least
// had siluMulF32AVX2. Routing them all through one kernel means the arithmetic
// is defined and tested once, and gives a single place for a future NEON or
// AVX2 implementation to hook into.
func geluMulF32(gate, up, out []float32) {
	n := min(len(gate), len(up), len(out))
	for i := range n {
		out[i] = geluTanhScalar(gate[i]) * up[i]
	}
}
