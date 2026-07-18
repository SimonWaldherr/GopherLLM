//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// quantizeQ8KRef is the scalar reference for q8kQuantize: symmetric absmax
// int8 quantization per 256-element block, round-to-nearest-even (matching
// VCVTPS2DQ under the default MXCSR rounding mode).
func quantizeQ8KRef(x []float32, cols int) ([]int8, []float32) {
	blocks := cols / 256
	q8 := make([]int8, cols)
	scales := make([]float32, blocks)
	for b := range blocks {
		xb := x[b*256 : b*256+256]
		var amax float32
		for _, v := range xb {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		if amax == 0 || amax != amax {
			continue
		}
		scales[b] = amax / 127
		inv := 127 / amax
		for i, v := range xb {
			q8[b*256+i] = int8(math.RoundToEven(float64(v * inv)))
		}
	}
	return q8, scales
}

// dotQ4KQ8KRowRef is the scalar reference for q4kDotQ8KRow: integer
// sub-block dots scaled by the decoded 6-bit scales and the per-block
// activation scale, minus the exact float dmin term over xsums.
func dotQ4KQ8KRowRef(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*144 : (b+1)*144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q := block[16:144]
		var blockInt int32
		for s := range 4 {
			sc1, _ := getScaleMinK4(2*s, scales)
			sc2, _ := getScaleMinK4(2*s+1, scales)
			var lo, hi int32
			for l := range 32 {
				qv := q[s*32+l]
				lo += int32(qv&0x0f) * int32(q8[b*256+s*64+l])
				hi += int32(qv>>4) * int32(q8[b*256+s*64+32+l])
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

// dotQ6KQ8KRowRef is the scalar reference for q6kDotQ8KRow. xsums are the
// per-16-element sums of the original activations pre-scaled by 32.
func dotQ6KQ8KRowRef(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*210 : (b+1)*210]
		ql := block[0:128]
		qh := block[128:192]
		sc := block[192:208]
		d := F16ToF32(binaryLE16(block[208:]))
		q8b := q8[b*256 : b*256+256]
		var blockInt int32
		for half := range 2 {
			qlh := ql[half*64 : half*64+64]
			qhh := qh[half*32 : half*32+32]
			sch := sc[half*8 : half*8+8]
			q8h := q8b[half*128 : half*128+128]
			for l := range 32 {
				is := l / 16
				q1 := int32((qlh[l] & 0x0f) | ((qhh[l] & 0x03) << 4))
				q2 := int32((qlh[l+32] & 0x0f) | (((qhh[l] >> 2) & 0x03) << 4))
				q3 := int32((qlh[l] >> 4) | (((qhh[l] >> 4) & 0x03) << 4))
				q4 := int32((qlh[l+32] >> 4) | (((qhh[l] >> 6) & 0x03) << 4))
				blockInt += int32(int8(sch[is]))*q1*int32(q8h[l]) +
					int32(int8(sch[is+2]))*q2*int32(q8h[32+l]) +
					int32(int8(sch[is+4]))*q3*int32(q8h[64+l]) +
					int32(int8(sch[is+6]))*q4*int32(q8h[96+l])
			}
		}
		sum += d * xscales[b] * float32(blockInt)
		var offTerm float32
		for i := range 16 {
			offTerm += float32(int8(sc[i])) * xsums[b*16+i]
		}
		sum -= d * offTerm
	}
	return sum
}

func TestQ8KQuantizeMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(11))
	for _, cols := range []int{256, 1024, 3072, 9216} {
		x := randomVec(rng, cols)
		// Exercise the zero-block branch too.
		for i := 256; i < min(512, cols); i++ {
			x[i] = 0
		}
		refQ8, refSc := quantizeQ8KRef(x, cols)
		q8 := make([]int8, cols)
		sc := make([]float32, cols/256)
		q8kQuantize(&x[0], &q8[0], &sc[0], cols/256)
		for b, want := range refSc {
			if sc[b] != want {
				t.Fatalf("cols=%d block %d: scale %v != %v", cols, b, sc[b], want)
			}
		}
		for i, want := range refQ8 {
			if q8[i] != want {
				t.Fatalf("cols=%d q8[%d] = %d != %d (x=%v scale=%v)", cols, i, q8[i], want, x[i], sc[i/256])
			}
		}
	}
}

func TestQ4KDotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(12))
	for _, cols := range []int{256, 1024, 3072, 4096, 9216} {
		row := randomQ4KRow(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ4KXSums(x, cols, &scratch)
		blocks := cols / 256
		got := q4kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ4KQ8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

func TestQ6KDotQ8KRowMatchesReference(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(13))
	for _, cols := range []int{256, 1024, 3072, 4096, 9216} {
		row := randomQ6KRow(rng, cols)
		x := randomVec(rng, cols)
		q8, xsc := quantizeQ8KRef(x, cols)
		scratch := []float32{}
		xs := fillQ6KXSums16(x, cols, &scratch)
		ScaleF32(xs, 32)
		blocks := cols / 256
		got := q6kDotQ8KRow(&row[0], &q8[0], &xsc[0], &xs[0], blocks)
		want := dotQ6KQ8KRowRef(row, q8, xsc, xs, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != ref %v (diff %v)", cols, got, want, diff)
		}
	}
}

func TestSiluMulF32MatchesScalar(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 required")
	}
	rng := rand.New(rand.NewSource(21))
	for _, n := range []int{1, 7, 8, 9, 256, 1023} {
		gate := make([]float32, n)
		up := make([]float32, n)
		for i := range gate {
			gate[i] = float32(rng.NormFloat64() * 8) // cover saturated sigmoid tails
			up[i] = float32(rng.NormFloat64())
		}
		gate[0] = -100 // clamp regions
		if n > 1 {
			gate[1] = 100
		}
		want := make([]float32, n)
		siluMulF32Scalar(gate, up, want, 0, n)
		got := make([]float32, n)
		siluMulF32(gate, up, got)
		for i := range want {
			diff := math.Abs(float64(got[i] - want[i]))
			if diff > 1e-5*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("n=%d i=%d gate=%v: %v != %v", n, i, gate[i], got[i], want[i])
			}
		}
	}
}

// withQ8Activations runs fn with the int8-activation path forced on or off.
func withQ8Activations(enabled bool, fn func()) {
	saved := useQ8Activations
	useQ8Activations = enabled && hasAVX2 && hasF16C
	defer func() { useQ8Activations = saved }()
	fn()
}

func requireCosine(t *testing.T, name string, fout, qout []float32) {
	t.Helper()
	cos, err := CosineSimilarity(fout, qout)
	if err != nil {
		t.Fatal(err)
	}
	if cos < 0.999 {
		t.Fatalf("%s: int8 matvec cosine similarity %.5f < 0.999", name, cos)
	}
	var maxAbs float32
	for _, v := range fout {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	for i := range fout {
		if math.Abs(float64(fout[i])) > 0.5*float64(maxAbs) {
			rel := math.Abs(float64(qout[i]-fout[i])) / math.Abs(float64(fout[i]))
			if rel > 0.05 {
				t.Fatalf("%s row %d: int8 %v vs float %v (rel %.4f)", name, i, qout[i], fout[i], rel)
			}
		}
	}
}

// TestQ4KQ8MatvecCloseToFloat checks that the int8-activation matvec stays
// very close to the exact float matvec: the output direction (what the next
// layer sees) must be preserved. int8 introduces a small magnitude error, so
// we bound cosine similarity rather than requiring an exact match.
func TestQ4KQ8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(3))
	const rows, cols = 96, 1024
	rowBytes := (cols / 256) * 144
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)

	fout := []float32{}
	withQ8Activations(false, func() { MatvecQ4KInto(data, x, rows, cols, &fout) })
	qout := []float32{}
	withQ8Activations(true, func() { MatvecQ4KInto(data, x, rows, cols, &qout) })
	requireCosine(t, "q4k", fout, qout)
}

func TestQ6KQ8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(4))
	const rows, cols = 96, 1024
	rowBytes := (cols / 256) * 210
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)

	fout := []float32{}
	withQ8Activations(false, func() { MatvecQ6KInto(data, x, rows, cols, &fout) })
	qout := []float32{}
	withQ8Activations(true, func() { MatvecQ6KInto(data, x, rows, cols, &qout) })
	requireCosine(t, "q6k", fout, qout)
}

// TestMatvecBatchQ8CloseToFloat checks the batched-prefill int8 path against
// the exact float per-token matvec for both K-quant types.
func TestMatvecBatchQ8CloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(6))
	const rows, cols, P = 48, 1024, 5
	xs := make([][]float32, P)
	for p := range xs {
		xs[p] = randomVec(rng, cols)
	}
	for _, tc := range []struct {
		name     string
		typ      GGMLType
		rowBytes int
		gen      func(*rand.Rand, int) []byte
	}{
		{"q4k", GGMLTypeQ4_K, (cols / 256) * 144, randomQ4KRow},
		{"q5k", GGMLTypeQ5_K, (cols / 256) * 176, randomQ5KRow},
		{"q6k", GGMLTypeQ6_K, (cols / 256) * 210, randomQ6KRow},
		{"q8_0", GGMLTypeQ8_0, (cols / 32) * 34, randomQ8_0Row},
		{"q4_0", GGMLTypeQ4_0, (cols / 32) * 18, randomQ4_0Row},
		{"q4_1", GGMLTypeQ4_1, (cols / 32) * 20, randomQ4_1Row},
		{"mxfp4", GGMLTypeMXFP4, (cols / 32) * 17, randomMXFP4Row},
	} {
		data := make([]byte, 0, rows*tc.rowBytes)
		for range rows {
			data = append(data, tc.gen(rng, cols)...)
		}
		w := Weight{Raw: data, Type: tc.typ, Rows: rows, Cols: cols}
		want := make([][]float32, P)
		withQ8Activations(false, func() {
			for p := range want {
				want[p] = w.Matvec(xs[p])
			}
		})
		got := make([][]float32, P)
		for p := range got {
			got[p] = make([]float32, rows)
		}
		withQ8Activations(true, func() {
			if !matvecBatchQ8(w, xs, got) {
				t.Fatalf("%s: matvecBatchQ8 rejected valid shapes", tc.name)
			}
		})
		for p := range P {
			requireCosine(t, tc.name, want[p], got[p])
		}
	}
}

// TestMixedQKVQ8MatvecCloseToFloat covers the fused Q4_K/Q4_K/Q6_K attention
// path (Ministral Q4_K_M's Q/K/V layout) through the int8 branch.
func TestMixedQKVQ8MatvecCloseToFloat(t *testing.T) {
	if !hasAVX2 || !hasF16C {
		t.Skip("AVX2+F16C required for the int8 activation path")
	}
	rng := rand.New(rand.NewSource(5))
	const qRows, kRows, vRows, cols = 64, 16, 16, 1024
	q4RowBytes := (cols / 256) * 144
	q6RowBytes := (cols / 256) * 210
	qData := make([]byte, 0, qRows*q4RowBytes)
	for range qRows {
		qData = append(qData, randomQ4KRow(rng, cols)...)
	}
	kData := make([]byte, 0, kRows*q4RowBytes)
	for range kRows {
		kData = append(kData, randomQ4KRow(rng, cols)...)
	}
	vData := make([]byte, 0, vRows*q6RowBytes)
	for range vRows {
		vData = append(vData, randomQ6KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)

	run := func() (q, k, v []float32) {
		sums := []float32{}
		if !MatvecQ4K2Q6KIntoWithXSums(qData, qRows, cols, kData, kRows, cols, vData, vRows, cols, x, &sums, &q, &k, &v) {
			t.Fatal("MatvecQ4K2Q6KIntoWithXSums rejected valid shapes")
		}
		return q, k, v
	}
	var fq, fk, fv, qq, qk, qv []float32
	withQ8Activations(false, func() { fq, fk, fv = run() })
	withQ8Activations(true, func() { qq, qk, qv = run() })
	requireCosine(t, "mixed-q", fq, qq)
	requireCosine(t, "mixed-k", fk, qk)
	requireCosine(t, "mixed-v", fv, qv)
}
