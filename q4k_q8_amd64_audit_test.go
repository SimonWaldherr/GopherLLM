//go:build amd64

package gopherllm

import (
	"math"
	"testing"
)

// Adversarial-extreme regression tests for the int8-activation asm kernels
// and siluMulF32AVX2: saturation bounds, lane-order mixups, and (for SiLU)
// the input range where a naive symmetric exp() clamp silently produces
// finite-but-wrong results instead of the correct near-zero output.

// All-max Q4_K block: every nibble 15, every scale/min 63 (0xff scale bytes),
// activations all +1 so q8 = +127 everywhere. Max VPMADDUBSW pair = 3810,
// max VPMADDWD lane = 63*3810*2, 8 adds per block -> checks saturation bounds.
func TestAuditQ4KDotExtremes(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 1024
	blocks := cols / 256
	row := make([]byte, blocks*144)
	for b := 0; b < blocks; b++ {
		block := row[b*144 : (b+1)*144]
		block[0], block[1] = 0x00, 0x3c // d = 1.0
		block[2], block[3] = 0x00, 0x3c // dmin = 1.0
		for i := 4; i < 144; i++ {
			block[i] = 0xff
		}
	}
	for _, sign := range []float32{1, -1} {
		x := make([]float32, cols)
		for i := range x {
			x[i] = sign
		}
		q8, xsc := quantizeQ8KRef(x, cols)
		for _, q := range q8 {
			if q != int8(127*sign) {
				t.Fatalf("expected q8=%v, got %d", 127*sign, q)
			}
		}
		scratch := []float32{}
		xs := fillQ4KXSums(x, cols, &scratch)
		got := q4kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ4KQ8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("sign=%v: asm %v != ref %v", sign, got, want)
		}
	}
}

// All-max Q6_K block with scale extremes -128 and +127.
func TestAuditQ6KDotExtremes(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 1024
	blocks := cols / 256
	for _, scByte := range []byte{0x7f, 0x80} { // +127 and -128
		row := make([]byte, blocks*210)
		for b := 0; b < blocks; b++ {
			block := row[b*210 : (b+1)*210]
			for i := 0; i < 192; i++ {
				block[i] = 0xff // ql/qh all ones -> quant 63
			}
			for i := 192; i < 208; i++ {
				block[i] = scByte
			}
			block[208], block[209] = 0x00, 0x3c // d = 1.0
		}
		for _, sign := range []float32{1, -1} {
			x := make([]float32, cols)
			for i := range x {
				x[i] = sign
			}
			q8, xsc := quantizeQ8KRef(x, cols)
			scratch := []float32{}
			xs := fillQ6KXSums16(x, cols, &scratch)
			ScaleF32(xs, 32)
			got := q6kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
			want := dotQ6KQ8KRowRef(row, q8, xsc, xs, blocks)
			if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
				t.Fatalf("sc=%#x sign=%v: asm %v != ref %v", scByte, sign, got, want)
			}
		}
	}
}

// Per-element distinguishing patterns: element i has a unique activation and
// unique quant so any lane permutation / scale-slot mixup shows up.
func TestAuditQ4KDotLaneOrder(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 512
	blocks := cols / 256
	row := make([]byte, blocks*144)
	for b := 0; b < blocks; b++ {
		block := row[b*144 : (b+1)*144]
		block[0], block[1] = 0x00, 0x3c
		block[2], block[3] = 0x00, 0x30 // dmin small
		for i := 4; i < 16; i++ {
			block[i] = byte((i*37 + b*11) & 0xff)
		}
		for i := 16; i < 144; i++ {
			block[i] = byte((i*89 + b*7) & 0xff)
		}
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%97)-48) / 48
	}
	q8, xsc := quantizeQ8KRef(x, cols)
	scratch := []float32{}
	xs := fillQ4KXSums(x, cols, &scratch)
	got := q4kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
	want := dotQ4KQ8KRowRef(row, q8, xsc, xs, blocks)
	if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
		t.Fatalf("asm %v != ref %v", got, want)
	}
}

func TestAuditQ6KDotLaneOrder(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 512
	blocks := cols / 256
	row := make([]byte, blocks*210)
	for b := 0; b < blocks; b++ {
		block := row[b*210 : (b+1)*210]
		for i := 0; i < 208; i++ {
			block[i] = byte((i*131 + b*17) & 0xff)
		}
		block[208], block[209] = 0x00, 0x3c
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%89)-44) / 44
	}
	q8, xsc := quantizeQ8KRef(x, cols)
	scratch := []float32{}
	xs := fillQ6KXSums16(x, cols, &scratch)
	ScaleF32(xs, 32)
	got := q6kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
	want := dotQ6KQ8KRowRef(row, q8, xsc, xs, blocks)
	if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
		t.Fatalf("asm %v != ref %v", got, want)
	}
}

// Sweep the sigmoid saturation boundary where the 2^n exponent construction
// hits n = -127 (gate ~ +87.7) and the clamp edges, plus denormal inputs.
func TestAuditSiluBoundary(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 required")
	}
	var gate []float32
	for g := -90.0; g <= 90.0; g += 0.003 {
		gate = append(gate, float32(g))
	}
	gate = append(gate, 87.68, -87.68, 88.376, -88.376, 88.3763, -88.3763,
		127.5, -127.5, 1e-40, -1e-40, 0, math.MaxFloat32, -math.MaxFloat32)
	for len(gate)%8 != 0 {
		gate = append(gate, 0)
	}
	n := len(gate)
	up := make([]float32, n)
	for i := range up {
		up[i] = 1
	}
	want := make([]float32, n)
	siluMulF32Scalar(gate, up, want, 0, n)
	got := make([]float32, n)
	siluMulF32AVX2(gate, up, got)
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > 1e-5*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("i=%d gate=%v: asm %v != scalar %v", i, gate[i], got[i], want[i])
		}
	}
}

// q8kQuantize with a zero block sandwiched between non-zero blocks, plus a
// block whose absmax sits at the last element (pointer-advance and
// horizontal-max order checks).
func TestAuditQ8KQuantizeEdges(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 1024
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%53)-26) / 26
	}
	for i := 256; i < 512; i++ {
		x[i] = 0 // zero block 1
	}
	for i := 512; i < 768; i++ {
		x[i] = 0
	}
	x[767] = -3.5 // absmax at final element of block 2
	refQ8, refSc := quantizeQ8KRef(x, cols)
	q8 := make([]int8, cols)
	sc := make([]float32, cols/256)
	// Poison outputs to catch unwritten slots.
	for i := range q8 {
		q8[i] = 99
	}
	for i := range sc {
		sc[i] = 99
	}
	q8kQuantize(&x[0], &q8[0], &sc[0], cols/256)
	for b, wantv := range refSc {
		if sc[b] != wantv {
			t.Fatalf("block %d: scale %v != %v", b, sc[b], wantv)
		}
	}
	for i, wantv := range refQ8 {
		if q8[i] != wantv {
			t.Fatalf("q8[%d] = %d != %d", i, q8[i], wantv)
		}
	}
}
