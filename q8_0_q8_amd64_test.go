//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// randomQ8_0Row builds a random Q8_0 row: per 32-element block, a small
// positive f16 scale followed by 32 signed int8 weights clamped to
// [-127,127] — real Q8_0 quantizers never emit -128, and the asm kernel's
// VPABSB-based abs computation relies on that (matching llama.cpp's own
// AVX2 Q8_0 kernel, which makes the same assumption).
func randomQ8_0Row(rng *rand.Rand, cols int) []byte {
	blocks := cols / 32
	row := make([]byte, blocks*34)
	for b := 0; b < blocks; b++ {
		block := row[b*34 : (b+1)*34]
		block[0], block[1] = byte(rng.Intn(256)), 0x2c // small positive f16 scale
		for i := 2; i < 34; i++ {
			v := int8(rng.Intn(255) - 127) // [-127, 127]
			block[i] = byte(v)
		}
	}
	return row
}

// dotQ8_0Q8KRowRef is the scalar reference for q8_0DotQ8KRow: an exact
// integer dot per 32-element Q8_0 block, scaled by that block's own d and
// the shared per-256-element activation scale.
func dotQ8_0Q8KRowRef(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	var sum float32
	for b := 0; b < blocks; b++ {
		base := b * 272
		var blockSum float32
		for j := 0; j < 8; j++ {
			off := base + j*34
			d := F16ToF32(binaryLE16(row[off:]))
			w := row[off+2 : off+34]
			q8b := q8[b*256+j*32 : b*256+j*32+32]
			var dot int32
			for l := 0; l < 32; l++ {
				dot += int32(int8(w[l])) * int32(q8b[l])
			}
			blockSum += d * float32(dot)
		}
		sum += xscales[b] * blockSum
	}
	return sum
}

func TestQ8_0DotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(21))
	for _, cols := range []int{256, 1024, 3072, 4096, 9216} {
		row := randomQ8_0Row(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		blocks := cols / 256
		got := q8_0DotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
		want := dotQ8_0Q8KRowRef(row, q8, xsc, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

// TestAuditQ8_0DotExtremes: every weight byte 127 (max positive, the largest
// magnitude Q8_0 quantizers actually emit) or -127, activations all +1
// (q8 = +127 everywhere) — exercises VPMADDUBSW/VPMADDWD saturation bounds.
func TestAuditQ8_0DotExtremes(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 1024
	blocks := cols / 256
	negWv := int8(-127)
	for _, wv := range []byte{127, byte(negWv)} {
		row := make([]byte, blocks*272)
		for b := 0; b < blocks; b++ {
			block := row[b*272 : (b+1)*272]
			for j := 0; j < 8; j++ {
				off := j * 34
				block[off], block[off+1] = 0x00, 0x3c // d = 1.0
				for i := 2; i < 34; i++ {
					block[off+i] = wv
				}
			}
		}
		for _, sign := range []float32{1, -1} {
			x := make([]float32, cols)
			for i := range x {
				x[i] = sign
			}
			q8, xsc := quantizeQ8KRef(x, cols)
			got := q8_0DotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
			want := dotQ8_0Q8KRowRef(row, q8, xsc, blocks)
			if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
				t.Fatalf("w=%d sign=%v: asm %v != ref %v", int8(wv), sign, got, want)
			}
		}
	}
}

// TestAuditQ8_0DotLaneOrder: every element has a unique weight and unique
// activation so any lane permutation / block-offset mixup surfaces.
func TestAuditQ8_0DotLaneOrder(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required")
	}
	const cols = 512
	blocks := cols / 256
	rng := rand.New(rand.NewSource(22))
	row := randomQ8_0Row(rng, cols)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%97)-48) / 48
	}
	q8, xsc := quantizeQ8KRef(x, cols)
	got := q8_0DotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
	want := dotQ8_0Q8KRowRef(row, q8, xsc, blocks)
	if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
		t.Fatalf("asm %v != ref %v", got, want)
	}
}

// TestQ8_0Q8MatvecCloseToFloat checks that MatvecQ8_0Into's int8-activation
// fast path (cols%256==0) stays very close to the exact float matvec, the
// same cosine-similarity bound the K-quant matvecs are held to.
func TestQ8_0Q8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(23))
	const rows, cols = 96, 1024
	rowBytes := (cols / 32) * 34
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomQ8_0Row(rng, cols)...)
	}
	x := randomVec(rng, cols)

	fout := []float32{}
	withQ8Activations(false, func() { MatvecQ8_0Into(data, x, rows, cols, &fout) })
	qout := []float32{}
	withQ8Activations(true, func() { MatvecQ8_0Into(data, x, rows, cols, &qout) })
	requireCosine(t, "q8_0", fout, qout)
}
