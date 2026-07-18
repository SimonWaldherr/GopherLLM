//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func randomQ5KRow(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*176)
	for b := 0; b < cols/256; b++ {
		block := row[b*176 : (b+1)*176]
		// Small positive f16 scales (0x2c00 ~ 0.0625, 0x1c00 ~ 0.0039).
		block[0], block[1] = byte(rng.Intn(256)), 0x2c
		block[2], block[3] = byte(rng.Intn(256)), 0x1c
		for i := 4; i < 176; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

// dotQ5KQ8KRowRef is the scalar reference for q5kDotQ8KRow: Q4_K's structure
// with the fifth bit from the qh plane OR'd onto each nibble (sub-block 2s
// takes bit 2s of qh[l], sub-block 2s+1 bit 2s+1, matching DequantRowQ5K).
func dotQ5KQ8KRowRef(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*176 : (b+1)*176]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		qh := block[16:48]
		q := block[48:176]
		var blockInt int32
		for s := range 4 {
			sc1, _ := getScaleMinK4(2*s, scales)
			sc2, _ := getScaleMinK4(2*s+1, scales)
			var lo, hi int32
			for l := range 32 {
				qv := q[s*32+l]
				h1 := int32((qh[l] >> (2 * s)) & 1)
				h2 := int32((qh[l] >> (2*s + 1)) & 1)
				lo += (int32(qv&0x0f) + h1*16) * int32(q8[b*256+s*64+l])
				hi += (int32(qv>>4) + h2*16) * int32(q8[b*256+s*64+32+l])
			}
			blockInt += int32(sc1)*lo + int32(sc2)*hi
		}
		sum += d * xscales[b] * float32(blockInt)
		var minTerm float32
		for j := range 8 {
			_, m := getScaleMinK4(j, scales)
			minTerm += float32(m) * xsums[b*8+j]
		}
		sum -= dmin * minTerm
	}
	return sum
}

func TestQ5KDotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(31))
	for _, cols := range []int{256, 1024, 3072, 4096, 9216} {
		row := randomQ5KRow(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ4KXSums(x, cols, &scratch)
		blocks := cols / 256
		got := q5kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ5KQ8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

// All-max Q5_K block: every nibble 15, every qh bit set (quant 31), every
// scale/min 63, activations all +/-1 (q8 = +/-127) — saturation bounds.
func TestAuditQ5KDotExtremes(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 1024
	blocks := cols / 256
	row := make([]byte, blocks*176)
	for b := 0; b < blocks; b++ {
		block := row[b*176 : (b+1)*176]
		block[0], block[1] = 0x00, 0x3c // d = 1.0
		block[2], block[3] = 0x00, 0x3c // dmin = 1.0
		for i := 4; i < 176; i++ {
			block[i] = 0xff
		}
	}
	for _, sign := range []float32{1, -1} {
		x := make([]float32, cols)
		for i := range x {
			x[i] = sign
		}
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ4KXSums(x, cols, &scratch)
		got := q5kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ5KQ8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("sign=%v: asm %v != ref %v", sign, got, want)
		}
	}
}

// Per-element distinguishing patterns so any lane permutation, qh bit-shift
// mixup, or scale-slot swap surfaces.
func TestAuditQ5KDotLaneOrder(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 512
	blocks := cols / 256
	row := make([]byte, blocks*176)
	for b := 0; b < blocks; b++ {
		block := row[b*176 : (b+1)*176]
		block[0], block[1] = 0x00, 0x3c
		block[2], block[3] = 0x00, 0x30 // dmin small
		for i := 4; i < 176; i++ {
			block[i] = byte((i*53 + b*29) & 0xff)
		}
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%97)-48) / 48
	}
	q8, xsc := quantizeQ8KRef(x, cols)
	scratch := []float32{}
	xs := fillQ4KXSums(x, cols, &scratch)
	got := q5kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
	want := dotQ5KQ8KRowRef(row, q8, xsc, xs, blocks)
	if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
		t.Fatalf("asm %v != ref %v", got, want)
	}
}

// TestQ5KQ8MatvecCloseToFloat holds MatvecQ5KInto's int8-activation path to
// the same cosine bound as the other quant types' matvecs.
func TestQ5KQ8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(32))
	const rows, cols = 96, 1024
	rowBytes := (cols / 256) * 176
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomQ5KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)

	fout := []float32{}
	withQ8Activations(false, func() { MatvecQ5KInto(data, x, rows, cols, &fout) })
	qout := []float32{}
	withQ8Activations(true, func() { MatvecQ5KInto(data, x, rows, cols, &qout) })
	requireCosine(t, "q5k", fout, qout)
}
