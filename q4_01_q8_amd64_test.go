//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func randomQ4_0Row(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/32)*18)
	for b := 0; b < cols/32; b++ {
		block := row[b*18 : (b+1)*18]
		block[0], block[1] = byte(rng.Intn(256)), 0x2c // small positive f16 d
		for i := 2; i < 18; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

func randomQ4_1Row(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/32)*20)
	for b := 0; b < cols/32; b++ {
		block := row[b*20 : (b+1)*20]
		block[0], block[1] = byte(rng.Intn(256)), 0x2c // f16 d
		block[2], block[3] = byte(rng.Intn(256)), 0x1c // f16 m
		for i := 4; i < 20; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

// dotQ4_0Q8KRowRef: per legacy block, d*(xscale*intdot(q, q8) - 8*xsum),
// low nibbles = elements 0..15, high nibbles = 16..31 (DotQ4_0F32's layout).
func dotQ4_0Q8KRowRef(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for g := 0; g < blocks*8; g++ {
		block := row[g*18 : (g+1)*18]
		d := F16ToF32(binaryLE16(block[0:]))
		var intdot int32
		for i := 0; i < 16; i++ {
			p := block[2+i]
			intdot += int32(p&0x0f) * int32(q8[g*32+i])
			intdot += int32(p>>4) * int32(q8[g*32+16+i])
		}
		sum += d * (xscales[g/8]*float32(intdot) - 8*xsums[g])
	}
	return sum
}

// dotQ4_1Q8KRowRef: per legacy block, d*xscale*intdot(q, q8) + m*xsum.
func dotQ4_1Q8KRowRef(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for g := 0; g < blocks*8; g++ {
		block := row[g*20 : (g+1)*20]
		d := F16ToF32(binaryLE16(block[0:]))
		m := F16ToF32(binaryLE16(block[2:]))
		var intdot int32
		for i := 0; i < 16; i++ {
			p := block[4+i]
			intdot += int32(p&0x0f) * int32(q8[g*32+i])
			intdot += int32(p>>4) * int32(q8[g*32+16+i])
		}
		sum += d*xscales[g/8]*float32(intdot) + m*xsums[g]
	}
	return sum
}

func TestQ4_0DotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(41))
	for _, cols := range []int{256, 1024, 3072, 4096} {
		row := randomQ4_0Row(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ4KXSums(x, cols, &scratch)
		blocks := cols / 256
		got := q4_0DotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ4_0Q8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

func TestQ4_1DotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(42))
	for _, cols := range []int{256, 1024, 3072, 4096} {
		row := randomQ4_1Row(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ4KXSums(x, cols, &scratch)
		blocks := cols / 256
		got := q4_1DotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ4_1Q8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

// Saturation bounds: all-0xff nibbles (q=15), max scales, +/-1 activations.
func TestAuditQ4_0Q4_1DotExtremes(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 1024
	blocks := cols / 256
	row0 := make([]byte, (cols/32)*18)
	row1 := make([]byte, (cols/32)*20)
	for g := 0; g < cols/32; g++ {
		b0 := row0[g*18 : (g+1)*18]
		b0[0], b0[1] = 0x00, 0x3c // d = 1.0
		for i := 2; i < 18; i++ {
			b0[i] = 0xff
		}
		b1 := row1[g*20 : (g+1)*20]
		b1[0], b1[1] = 0x00, 0x3c // d = 1.0
		b1[2], b1[3] = 0x00, 0xbc // m = -1.0
		for i := 4; i < 20; i++ {
			b1[i] = 0xff
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
		got0 := q4_0DotQ8KRow(&row0[0], &q8[0], &xsc[0], &xs[0], blocks)
		want0 := dotQ4_0Q8KRowRef(row0, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got0 - want0)); diff > 1e-3*(1+math.Abs(float64(want0))) {
			t.Fatalf("q4_0 sign=%v: asm %v != ref %v", sign, got0, want0)
		}
		got1 := q4_1DotQ8KRow(&row1[0], &q8[0], &xsc[0], &xs[0], blocks)
		want1 := dotQ4_1Q8KRowRef(row1, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got1 - want1)); diff > 1e-3*(1+math.Abs(float64(want1))) {
			t.Fatalf("q4_1 sign=%v: asm %v != ref %v", sign, got1, want1)
		}
	}
}

// Per-element distinguishing patterns so lane permutations and low/high
// nibble mixups surface.
func TestAuditQ4_0Q4_1DotLaneOrder(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 512
	blocks := cols / 256
	row0 := make([]byte, (cols/32)*18)
	row1 := make([]byte, (cols/32)*20)
	for i := range row0 {
		row0[i] = byte((i*53 + 7) & 0xff)
	}
	for i := range row1 {
		row1[i] = byte((i*29 + 3) & 0xff)
	}
	for g := 0; g < cols/32; g++ {
		row0[g*18], row0[g*18+1] = 0x00, 0x3c
		row1[g*20], row1[g*20+1] = 0x00, 0x3c
		row1[g*20+2], row1[g*20+3] = 0x00, 0x30
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%89)-44) / 44
	}
	q8, xsc := quantizeQ8KRef(x, cols)
	scratch := []float32{}
	xs := fillQ4KXSums(x, cols, &scratch)
	got0 := q4_0DotQ8KRow(&row0[0], &q8[0], &xsc[0], &xs[0], blocks)
	want0 := dotQ4_0Q8KRowRef(row0, q8, xsc, xs, blocks)
	if diff := math.Abs(float64(got0 - want0)); diff > 1e-3*(1+math.Abs(float64(want0))) {
		t.Fatalf("q4_0: asm %v != ref %v", got0, want0)
	}
	got1 := q4_1DotQ8KRow(&row1[0], &q8[0], &xsc[0], &xs[0], blocks)
	want1 := dotQ4_1Q8KRowRef(row1, q8, xsc, xs, blocks)
	if diff := math.Abs(float64(got1 - want1)); diff > 1e-3*(1+math.Abs(float64(want1))) {
		t.Fatalf("q4_1: asm %v != ref %v", got1, want1)
	}
}

// Matvec-level cosine bound against the exact float kernels.
func TestQ4_0Q4_1Q8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(43))
	const rows, cols = 96, 1024
	x := randomVec(rng, cols)

	data0 := make([]byte, 0, rows*(cols/32)*18)
	for range rows {
		data0 = append(data0, randomQ4_0Row(rng, cols)...)
	}
	fout := []float32{}
	withQ8Activations(false, func() { MatvecQ4_0Into(data0, x, rows, cols, &fout) })
	qout := []float32{}
	withQ8Activations(true, func() { MatvecQ4_0Into(data0, x, rows, cols, &qout) })
	requireCosine(t, "q4_0", fout, qout)

	data1 := make([]byte, 0, rows*(cols/32)*20)
	for range rows {
		data1 = append(data1, randomQ4_1Row(rng, cols)...)
	}
	fout1 := []float32{}
	withQ8Activations(false, func() { MatvecQ4_1Into(data1, x, rows, cols, &fout1) })
	qout1 := []float32{}
	withQ8Activations(true, func() { MatvecQ4_1Into(data1, x, rows, cols, &qout1) })
	requireCosine(t, "q4_1", fout1, qout1)
}
