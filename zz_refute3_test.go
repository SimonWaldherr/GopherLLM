//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// Proposed fix: identical byte consumption, but pin the f16 d high byte to
// 0x2c exactly as the repo's own randomQ3KRow does.
func randomQ3KRowForAsmFixed(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*110)
	for i := range row {
		row[i] = byte(rng.Intn(256))
	}
	for b := range cols / 256 {
		row[b*110+109] = 0x2c
	}
	return row
}

// Re-run the full scale sweep with the fix, and check whether the previously
// vacuous points sc=2 / sc=5 were hiding a genuine kernel disagreement.
func TestZZRefuteFixedSweep(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(90213))
	const cols = 256
	vacuous, mismatch := 0, 0
	for sc := range 64 {
		row := randomQ3KRowForAsmFixed(rng, cols)
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

		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], 1)
		want := dotQ3KQ8KRowRef(row, q8, xsc, 1)
		port := q3kDotQ8KRowPortable(row, q8, xsc, 1)

		if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
			vacuous++
			t.Errorf("sc=%d still non-finite", sc)
		}
		diff := math.Abs(float64(got - want))
		if diff > 1e-3*(1+math.Abs(float64(want))) {
			mismatch++
			t.Errorf("sc=%d: asm %v != ref %v", sc, got, want)
		}
		if sc == 2 || sc == 5 || sc == 32 {
			t.Logf("sc=%2d (previously vacuous/at-risk): got=%v ref=%v portable=%v OK=%v",
				sc, got, want, port, diff <= 1e-3*(1+math.Abs(float64(want))))
		}
		if sc == 32 && got != 0 {
			t.Errorf("sc=32 must be exactly 0, got %v", got)
		}
	}
	t.Logf("after fix: vacuous=%d mismatches=%d (of 64)", vacuous, mismatch)
}
