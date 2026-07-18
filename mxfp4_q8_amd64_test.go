//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func randomMXFP4Row(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/32)*17)
	for b := 0; b < cols/32; b++ {
		block := row[b*17 : (b+1)*17]
		for i := 0; i < 16; i++ {
			block[i] = byte(rng.Intn(256))
		}
		// E8M0 exponents around 1.0 (127 +/- 8) keep sums well-conditioned.
		block[16] = byte(119 + rng.Intn(17))
	}
	return row
}

// dotMXFP4Q8KRowRef mirrors mxfp4DotQ8KRow exactly: integer dot of doubled
// e2m1 magnitudes (signed) against q8, scaled by 2^(e-127)*0.5*xscale per
// block, with element 2i = low nibble and 2i+1 = high nibble of byte i.
func dotMXFP4Q8KRowRef(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	doubled := [16]int32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
	var sum float32
	for g := 0; g < blocks*8; g++ {
		block := row[g*17 : (g+1)*17]
		e := int(block[16])
		scale := float32(0)
		if e > 0 {
			scale = math.Float32frombits(uint32(e) << 23)
		}
		var intdot int32
		for i := 0; i < 16; i++ {
			v := block[i]
			intdot += doubled[v&0x0f] * int32(q8[g*32+i*2])
			intdot += doubled[v>>4] * int32(q8[g*32+i*2+1])
		}
		sum += scale * 0.5 * xscales[g/8] * float32(intdot)
	}
	return sum
}

func TestMXFP4DotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(51))
	for _, cols := range []int{256, 1024, 3072, 4096} {
		row := randomMXFP4Row(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		blocks := cols / 256
		got := mxfp4DotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
		want := dotMXFP4Q8KRowRef(row, q8, xsc, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

// Every code 0..15 in every position, extreme +/-1 activations: exercises the
// full magnitude/sign LUT including -0 (code 8) and both unpack halves.
func TestAuditMXFP4DotAllCodes(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 512
	blocks := cols / 256
	row := make([]byte, (cols/32)*17)
	for g := 0; g < cols/32; g++ {
		block := row[g*17 : (g+1)*17]
		for i := 0; i < 16; i++ {
			block[i] = byte(((g*16 + i) * 17) & 0xff) // cycles through all code pairs
		}
		block[16] = 127 // scale 1.0
	}
	for _, sign := range []float32{1, -1} {
		x := make([]float32, cols)
		for i := range x {
			x[i] = sign * float32((i%13)+1) / 13
		}
		q8, xsc := quantizeQ8KRef(x, cols)
		got := mxfp4DotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
		want := dotMXFP4Q8KRowRef(row, q8, xsc, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("sign=%v: asm %v != ref %v", sign, got, want)
		}
	}
}

// Matvec-level cosine bound against the exact float kernel (which uses the
// true 0.5-step FP4 values rather than the doubled-integer identity).
func TestMXFP4Q8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(52))
	const rows, cols = 96, 1024
	rowBytes := (cols / 32) * 17
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomMXFP4Row(rng, cols)...)
	}
	x := randomVec(rng, cols)
	fout := []float32{}
	withQ8Activations(false, func() { MatvecMXFP4Into(data, x, rows, cols, &fout) })
	qout := []float32{}
	withQ8Activations(true, func() { MatvecMXFP4Into(data, x, rows, cols, &qout) })
	requireCosine(t, "mxfp4", fout, qout)
}
