//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// Replays the EXACT input generation of TestQ3KDotQ8KRowAsmScaleBiasAndUnpack.
func TestZZRefuteScaleSweep(t *testing.T) {
	t.Logf("hasAVX2=%v", hasAVX2)
	rng := rand.New(rand.NewSource(90213))
	const cols = 256
	vacuous := []int{}
	for sc := range 64 {
		row := randomQ3KRowForAsm(rng, cols)
		for i := range 8 {
			row[96+i] = byte(sc&0x0f) | byte(sc&0x0f)<<4
		}
		for i := range 4 {
			hi := byte(sc>>4) & 3
			row[104+i] = hi | hi<<2 | hi<<4 | hi<<6
		}
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, 1)
		q8kQuantizePortable(x, q8, xsc, 1)

		d := F16ToF32(binaryLE16(row[108:]))
		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], 1)
		want := dotQ3KQ8KRowRef(row, q8, xsc, 1)

		bad := math.IsNaN(float64(d)) || math.IsInf(float64(d), 0)
		if bad || sc == 32 {
			t.Logf("sc=%2d raw=0x%02x%02x d=%v got=%v want=%v", sc, row[109], row[108], d, got, want)
		}

		// Is the tolerance assertion vacuous? i.e. does a fabricated wrong
		// result pass it?
		fabricated := float32(12345.0)
		diff := math.Abs(float64(fabricated - want))
		if !(diff > 1e-3*(1+math.Abs(float64(want)))) {
			vacuous = append(vacuous, sc)
			t.Logf("  -> sc=%d: fabricated 12345.0 PASSES the diff check (want=%v)", sc, want)
		}
		if sc == 32 {
			t.Logf("  -> sc=32 check: got != 0 is %v (got=%v, d=%v)", got != 0, got, d)
		}
	}
	t.Logf("VACUOUS ITERATIONS: %v (count=%d of 64)", vacuous, len(vacuous))
}

// Check the other three tests for the same vacuity.
func TestZZRefuteOtherTests(t *testing.T) {
	check := func(name string, seed int64, colsList []int) {
		rng := rand.New(rand.NewSource(seed))
		for _, cols := range colsList {
			blocks := cols / 256
			row := randomQ3KRowForAsm(rng, cols)
			x := randomVec(rng, cols)
			q8 := make([]int8, cols)
			xsc := make([]float32, blocks)
			q8kQuantizePortable(x, q8, xsc, blocks)
			want := dotQ3KQ8KRowRef(row, q8, xsc, blocks)
			nonfinite := 0
			for b := range blocks {
				d := F16ToF32(binaryLE16(row[b*110+108:]))
				if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
					nonfinite++
				}
			}
			fab := float32(12345.0)
			diff := math.Abs(float64(fab - want))
			vac := !(diff > 1e-3*(1+math.Abs(float64(want))))
			t.Logf("%s cols=%d: blocks=%d nonfinite_d=%d want=%v vacuous=%v", name, cols, blocks, nonfinite, want, vac)
		}
	}
	check("MatchesReference", 90210, []int{256, 512, 1024, 3072, 4096})
	check("MatchesPortable", 90211, []int{256, 1024, 2048})
}

// Empirically measure the probability that a uniformly random f16 is non-finite.
func TestZZRefuteF16Probability(t *testing.T) {
	nonfinite := 0
	for h := 0; h < 65536; h++ {
		v := F16ToF32(uint16(h))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			nonfinite++
		}
	}
	t.Logf("non-finite f16 patterns: %d/65536 = 1 in %.1f", nonfinite, 65536.0/float64(nonfinite))
}
