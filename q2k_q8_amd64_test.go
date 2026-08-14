//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// dotQ2KQ8KRowRef is an independently-written scalar reference for
// q2kDotQ8KRow, deliberately not reusing q2kDotQ8KRowPortable (the function
// the asm is meant to speed up) so a shared bug in both wouldn't slip past
// this test — mirrors dotQ5KQ8KRowRef's role for q5kDotQ8KRow.
func dotQ2KQ8KRowRef(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*84 : (b+1)*84]
		scales := block[0:16]
		qs := block[16:80]
		d := F16ToF32(binaryLE16(block[80:]))
		dmin := F16ToF32(binaryLE16(block[82:]))
		q8b := q8[b*256 : b*256+256]
		var blockInt int32
		var minTerm float32
		is := 0
		y := 0
		for n := 0; n < 256; n += 128 {
			q := qs[(n/128)*32 : (n/128)*32+32]
			for shift := 0; shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					sc := scales[is]
					var qdot int32
					for l := 0; l < 16; l++ {
						qv := (q[half*16+l] >> shift) & 3
						qdot += int32(qv) * int32(q8b[y+l])
					}
					blockInt += int32(sc&0x0f) * qdot
					minTerm += float32(sc>>4) * xsums[b*16+is]
					is++
					y += 16
				}
			}
		}
		sum += d*xscales[b]*float32(blockInt) - dmin*minTerm
	}
	return sum
}

func TestQ2KDotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(41))
	for _, cols := range []int{256, 1024, 3072, 4096, 9216} {
		row := randomQ2KRow(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ6KXSums16(x, cols, &scratch)
		blocks := cols / 256
		got := q2kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ2KQ8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

// All-max Q2_K block: every scale byte 0xff (sc&0xf=15, sc>>4=15), every
// 2-bit quant 3, activations all +/-1 (q8 = +/-127) — saturation bounds.
func TestAuditQ2KDotExtremes(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 1024
	blocks := cols / 256
	row := make([]byte, blocks*84)
	for b := 0; b < blocks; b++ {
		block := row[b*84 : (b+1)*84]
		for i := 0; i < 80; i++ {
			block[i] = 0xff
		}
		block[80], block[81] = 0x00, 0x3c // d = 1.0
		block[82], block[83] = 0x00, 0x3c // dmin = 1.0
	}
	for _, sign := range []float32{1, -1} {
		x := make([]float32, cols)
		for i := range x {
			x[i] = sign
		}
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ6KXSums16(x, cols, &scratch)
		got := q2kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ2KQ8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("sign=%v: asm %v != ref %v", sign, got, want)
		}
	}
}

// Per-element distinguishing patterns so any lane permutation, shift mixup,
// or scale-slot swap surfaces.
func TestAuditQ2KDotLaneOrder(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 512
	blocks := cols / 256
	row := make([]byte, blocks*84)
	for b := 0; b < blocks; b++ {
		block := row[b*84 : (b+1)*84]
		for i := 0; i < 80; i++ {
			block[i] = byte((i*53 + b*29) & 0xff)
		}
		block[80], block[81] = 0x00, 0x3c
		block[82], block[83] = 0x00, 0x30 // dmin small
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%97)-48) / 48
	}
	q8, xsc := quantizeQ8KRef(x, cols)
	scratch := []float32{}
	xs := fillQ6KXSums16(x, cols, &scratch)
	got := q2kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
	want := dotQ2KQ8KRowRef(row, q8, xsc, xs, blocks)
	if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
		t.Fatalf("asm %v != ref %v", got, want)
	}
}

// TestQ2KQ8MatvecCloseToFloat holds MatvecQ2KInto's int8-activation path to
// the same cosine bound as the other quant types' matvecs.
func TestQ2KQ8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(42))
	const rows, cols = 96, 1024
	rowBytes := (cols / 256) * 84
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomQ2KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)

	fout := []float32{}
	withQ8Activations(false, func() { MatvecQ2KInto(data, x, rows, cols, &fout) })
	qout := []float32{}
	withQ8Activations(true, func() { MatvecQ2KInto(data, x, rows, cols, &qout) })
	requireCosine(t, "q2k", fout, qout)
}
