//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// Replay TestQ3KDotQ8KRowAsmHmaskExtremes (seed 90212) exactly.
func TestZZRefuteHmaskExtremes(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(90212))
	const cols = 1024
	blocks := cols / 256
	report := func(label string, row []byte) {
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)
		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
		want := dotQ3KQ8KRowRef(row, q8, xsc, blocks)
		fab := float32(12345)
		dead := !(math.Abs(float64(fab-want)) > 1e-3*(1+math.Abs(float64(want))))
		t.Logf("%-16s got=%-10v want=%-10v ASSERTION-DEAD=%v", label, got, want, dead)
	}
	for _, fill := range []byte{0x00, 0xff} {
		row := randomQ3KRowForAsm(rng, cols)
		for b := range blocks {
			for i := range 32 {
				row[b*110+i] = fill
			}
		}
		report("hmask="+string("ab"[fill&1]), row)
	}
	row := randomQ3KRowForAsm(rng, cols)
	for b := range blocks {
		for i := range 32 {
			row[b*110+i] = byte(1 << (i % 8))
		}
	}
	report("striped hmask", row)
}
