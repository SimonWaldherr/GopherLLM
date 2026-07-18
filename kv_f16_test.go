package gopherllm

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

// TestF32ToF16RoundTripsAllHalfValues: every finite f16 bit pattern must
// survive f16 -> f32 -> f16 exactly (conversion to the wider type is exact,
// and rounding back must land on the same value).
func TestF32ToF16RoundTripsAllHalfValues(t *testing.T) {
	for h := 0; h < 1<<16; h++ {
		bits := uint16(h)
		f := F16ToF32(bits)
		if math.IsNaN(float64(f)) {
			if got := F32ToF16(f); got&0x7c00 != 0x7c00 || got&0x03ff == 0 {
				t.Fatalf("NaN %#04x -> %v -> %#04x is not a NaN", bits, f, got)
			}
			continue
		}
		if got := F32ToF16(f); got != bits {
			// -0 and +0 compare equal as floats but differ in bits; the
			// round-trip must still preserve the sign bit exactly.
			t.Fatalf("%#04x -> %v -> %#04x", bits, f, got)
		}
	}
}

func TestF32ToF16RoundsToNearestEven(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{1, 0x3c00},
		{-2, 0xc000},
		{65504, 0x7bff},           // f16 max
		{65520, 0x7c00},           // rounds up to +inf
		{1e9, 0x7c00},             // overflow -> +inf
		{float32(math.Inf(-1)), 0xfc00},
		{5.9604645e-8, 0x0001},    // smallest subnormal
		{1e-10, 0x0000},           // underflow -> zero
		{1.0009765625, 0x3c01},    // 1 + 1ulp(f16)
		{1.00048828125, 0x3c00},   // exact tie 1 + 0.5ulp -> even (down)
		{1.00146484375, 0x3c02},   // exact tie 3rd -> even (up)
	}
	for _, c := range cases {
		if got := F32ToF16(c.in); got != c.want {
			t.Fatalf("F32ToF16(%v) = %#04x, want %#04x", c.in, got, c.want)
		}
	}
}

func TestF16KernelsMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	for _, n := range []int{8, 64, 128, 96, 7, 33, 121} {
		a := randomVec(rng, n)
		h := make([]uint16, n)
		for i := range h {
			h[i] = F32ToF16(rng.Float32()*4 - 2)
		}

		gotDot := dotF32F16(a, h)
		wantDot := dotF32F16Scalar(a, h, 0)
		if diff := math.Abs(float64(gotDot - wantDot)); diff > 1e-4*(1+math.Abs(float64(wantDot))) {
			t.Fatalf("n=%d dot: %v != %v", n, gotDot, wantDot)
		}

		out1 := append([]float32(nil), a...)
		out2 := append([]float32(nil), a...)
		axpyF16(out1, 0.7, h)
		axpyF16Scalar(out2, 0.7, h, 0)
		for i := range out1 {
			if diff := math.Abs(float64(out1[i] - out2[i])); diff > 1e-5 {
				t.Fatalf("n=%d axpy[%d]: %v != %v", n, i, out1[i], out2[i])
			}
		}

		out1 = append([]float32(nil), a...)
		out2 = append([]float32(nil), a...)
		scaleAddF16(out1, 0.3, h)
		scaleAddF16Scalar(out2, 0.3, h, 0)
		for i := range out1 {
			if diff := math.Abs(float64(out1[i] - out2[i])); diff > 1e-5 {
				t.Fatalf("n=%d scaleAdd[%d]: %v != %v", n, i, out1[i], out2[i])
			}
		}

		dst1 := make([]uint16, n)
		dst2 := make([]uint16, n)
		f32ToF16Row(dst1, a)
		f32ToF16RowScalar(dst2, a, 0)
		for i := range dst1 {
			if dst1[i] != dst2[i] {
				t.Fatalf("n=%d cvt[%d]: %#04x != %#04x (src %v)", n, i, dst1[i], dst2[i], a[i])
			}
		}
	}
}

// TestOnlineAttentionF16MatchesF32 drives both attention variants over the
// same K/V content (f16-rounded in both so only the kernel paths differ).
func TestOnlineAttentionF16MatchesF32(t *testing.T) {
	rng := rand.New(rand.NewSource(62))
	const headDim, ctx = 64, 96
	q := randomVec(rng, headDim)
	k16 := make([]uint16, ctx*headDim)
	v16 := make([]uint16, ctx*headDim)
	kf := make([]float32, ctx*headDim)
	vf := make([]float32, ctx*headDim)
	for i := range k16 {
		k16[i] = F32ToF16(rng.Float32()*2 - 1)
		v16[i] = F32ToF16(rng.Float32()*2 - 1)
		kf[i] = F16ToF32(k16[i])
		vf[i] = F16ToF32(v16[i])
	}
	scale := float32(1 / math.Sqrt(headDim))
	outF32 := make([]float32, headDim)
	outF16 := make([]float32, headDim)
	onlineAttention(q, kf, vf, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, outF32)
	onlineAttentionF16(q, k16, v16, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 0, outF16)
	for i := range outF32 {
		if diff := math.Abs(float64(outF32[i] - outF16[i])); diff > 1e-4 {
			t.Fatalf("out[%d]: f32 %v != f16 %v", i, outF32[i], outF16[i])
		}
	}
	// Softcap branch too (Gemma-style).
	clear(outF32)
	clear(outF16)
	onlineAttention(q, kf, vf, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 50, outF32)
	onlineAttentionF16(q, k16, v16, headDim, headDim, headDim, headDim, 0, ctx-1, scale, 50, outF16)
	for i := range outF32 {
		if diff := math.Abs(float64(outF32[i] - outF16[i])); diff > 1e-4 {
			t.Fatalf("softcap out[%d]: f32 %v != f16 %v", i, outF32[i], outF16[i])
		}
	}
}

// naiveAttentionRef is the textbook two-loop softmax attention reference the
// production two-pass implementation must match.
func naiveAttentionRef(q, keys, values []float32, stride, headDim, startT, endT int, scale, softcap float32) []float32 {
	type sw struct {
		t int
		s float64
	}
	var scored []sw
	for t := startT; t <= endT; t++ {
		if (t+1)*stride > len(keys) {
			break
		}
		var dot float64
		for i := 0; i < headDim; i++ {
			dot += float64(q[i]) * float64(keys[t*stride+i])
		}
		s := dot * float64(scale)
		if softcap > 0 {
			s = float64(softcap) * math.Tanh(s/float64(softcap))
		}
		scored = append(scored, sw{t, s})
	}
	out := make([]float32, headDim)
	if len(scored) == 0 {
		return out
	}
	maxS := scored[0].s
	for _, e := range scored[1:] {
		maxS = math.Max(maxS, e.s)
	}
	var denom float64
	for _, e := range scored {
		denom += math.Exp(e.s - maxS)
	}
	for _, e := range scored {
		w := math.Exp(e.s-maxS) / denom
		for i := 0; i < headDim; i++ {
			out[i] += float32(w * float64(values[e.t*stride+i]))
		}
	}
	return out
}

// TestOnlineAttentionMatchesNaiveReference pins the two-pass production
// implementation to the textbook softmax, including the sliding-window
// (startT > 0) and softcap branches.
func TestOnlineAttentionMatchesNaiveReference(t *testing.T) {
	rng := rand.New(rand.NewSource(63))
	const headDim, ctx = 64, 200
	q := randomVec(rng, headDim)
	keys := randomVec(rng, ctx*headDim)
	values := randomVec(rng, ctx*headDim)
	scale := float32(1 / math.Sqrt(headDim))
	for _, tc := range []struct {
		name           string
		startT, endT   int
		softcap        float32
	}{
		{"full", 0, ctx - 1, 0},
		{"slidingWindow", 64, ctx - 1, 0},
		{"softcap", 0, ctx - 1, 30},
		{"singlePos", 5, 5, 0},
	} {
		out := make([]float32, headDim)
		onlineAttention(q, keys, values, headDim, headDim, headDim, headDim, tc.startT, tc.endT, scale, tc.softcap, out)
		want := naiveAttentionRef(q, keys, values, headDim, headDim, tc.startT, tc.endT, scale, tc.softcap)
		for i := range want {
			if diff := math.Abs(float64(out[i] - want[i])); diff > 1e-5 {
				t.Fatalf("%s: out[%d] = %v, want %v", tc.name, i, out[i], want[i])
			}
		}
	}
}

// TestGenerateWithF16KVCacheMatchesF32 proves end-to-end greedy decode on the
// synthetic model produces identical tokens with the f16 cache: KV rounding
// is far below the argmax decision margin on every step of this model.
func TestGenerateWithF16KVCacheMatchesF32(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 12
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1

	run := func() string {
		res, err := m.Runner().Generate("a b c d", opts)
		if err != nil {
			t.Fatal(err)
		}
		return res.Text
	}
	var f32Text, f16Text string
	withF16KVCache(false, func() { f32Text = run() })
	withF16KVCache(true, func() { f16Text = run() })
	if f32Text != f16Text {
		t.Fatalf("greedy decode diverged: f32=%q f16=%q", f32Text, f16Text)
	}
}
