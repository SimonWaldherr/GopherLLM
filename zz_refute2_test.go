//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// Force the sc==32 (dl==0) case to carry a non-finite d and see whether the
// kernel's own `got != 0` guard would fire on a CORRECT kernel.
func TestZZRefuteSc32NonFiniteD(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	const cols = 256
	for _, tc := range []struct {
		name string
		lo   byte
		hi   byte
	}{
		{"+Inf", 0x00, 0x7c},
		{"-Inf", 0x00, 0xfc},
		{"NaN", 0x01, 0x7c},
		{"finite 1.0", 0x00, 0x3c},
	} {
		rng := rand.New(rand.NewSource(1))
		row := randomQ3KRowForAsm(rng, cols)
		sc := 32
		for i := range 8 {
			row[96+i] = byte(sc&0x0f) | byte(sc&0x0f)<<4
		}
		for i := range 4 {
			hi := byte(sc>>4) & 3
			row[104+i] = hi | hi<<2 | hi<<4 | hi<<6
		}
		row[108], row[109] = tc.lo, tc.hi

		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, 1)
		q8kQuantizePortable(x, q8, xsc, 1)

		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], 1)
		want := dotQ3KQ8KRowRef(row, q8, xsc, 1)
		portable := q3kDotQ8KRowPortable(row, q8, xsc, 1)

		// Would the real test's two assertions fire?
		diff := math.Abs(float64(got - want))
		tolFires := diff > 1e-3*(1+math.Abs(float64(want)))
		zeroFires := got != 0

		t.Logf("d=%-10s got=%-8v want=%-8v portable=%-8v | tolerance-check fires=%v  sc32-check fires=%v",
			tc.name, got, want, portable, tolFires, zeroFires)
		if zeroFires && !tolFires {
			t.Logf("   ^^^ SPURIOUS FAILURE: kernel agrees with BOTH references, yet `sc == 32 && got != 0` trips")
		}
	}
}
