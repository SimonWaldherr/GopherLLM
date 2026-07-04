package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func randomQ4KRow(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*144)
	for b := 0; b < cols/256; b++ {
		block := row[b*144 : (b+1)*144]
		// Small positive f16 scales (0x2c00 ~ 0.0625, 0x1c00 ~ 0.0039).
		block[0], block[1] = byte(rng.Intn(256)), 0x2c
		block[2], block[3] = byte(rng.Intn(256)), 0x1c
		for i := 4; i < 144; i++ {
			block[i] = byte(rng.Intn(256))
		}
	}
	return row
}

func randomQ6KRow(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*210)
	for b := 0; b < cols/256; b++ {
		block := row[b*210 : (b+1)*210]
		for i := 0; i < 208; i++ {
			block[i] = byte(rng.Intn(256))
		}
		block[208], block[209] = byte(rng.Intn(256)), 0x2c
	}
	return row
}

func randomVec(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}

func TestDotQ4KWithXSumsMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, cols := range []int{256, 512, 1024} {
		row := randomQ4KRow(rng, cols)
		x := randomVec(rng, cols)
		scratch := []float32{}
		xs := fillQ4KXSums(x, cols, &scratch)
		want := DotQ4KF32(row, x, cols)
		got := dotQ4KF32WithXSums(row, x, xs, cols)
		if math.Abs(float64(got-want)) > 1e-2*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("cols=%d: dot = %v, want %v", cols, got, want)
		}
		gotScalar := dotQ4KF32ScalarWithXSums(row, x, xs, cols)
		if math.Abs(float64(gotScalar-want)) > 1e-2*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("cols=%d: scalar dot = %v, want %v", cols, gotScalar, want)
		}
	}
}

func TestDotQ6KWithXSumsMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, cols := range []int{256, 512, 1024} {
		row := randomQ6KRow(rng, cols)
		x := randomVec(rng, cols)
		scratch := []float32{}
		xs := fillQ6KXSums16(x, cols, &scratch)
		want := DotQ6KF32(row, x, cols)
		out := make([]float32, 1)
		dotQ6KRowsWithXSums(row, x, xs, cols, len(row), 0, 1, out)
		if math.Abs(float64(out[0]-want)) > 1e-2*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("cols=%d: dot = %v, want %v", cols, out[0], want)
		}
	}
}

func TestFillQ6KXSums16(t *testing.T) {
	x := make([]float32, 64)
	for i := range x {
		x[i] = float32(i)
	}
	scratch := []float32{}
	xs := fillQ6KXSums16(x, 64, &scratch)
	if len(xs) != 4 {
		t.Fatalf("len = %d, want 4", len(xs))
	}
	for g := range 4 {
		var want float32
		for i := g * 16; i < (g+1)*16; i++ {
			want += x[i]
		}
		if math.Abs(float64(xs[g]-want)) > 1e-4 {
			t.Fatalf("group %d: %v, want %v", g, xs[g], want)
		}
	}
}

func TestMatvecQ4KIntoMatchesPerRowDot(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const rows, cols = 16, 512
	rowBytes := (cols / 256) * 144
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)
	out := []float32{}
	MatvecQ4KInto(data, x, rows, cols, &out)
	for r := range rows {
		want := DotQ4KF32(data[r*rowBytes:(r+1)*rowBytes], x, cols)
		if math.Abs(float64(out[r]-want)) > 1e-2*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("row %d: %v, want %v", r, out[r], want)
		}
	}
}

func TestMatvecQ6KIntoMatchesPerRowDot(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	const rows, cols = 16, 512
	rowBytes := (cols / 256) * 210
	data := make([]byte, 0, rows*rowBytes)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)
	out := []float32{}
	MatvecQ6KInto(data, x, rows, cols, &out)
	for r := range rows {
		want := DotQ6KF32(data[r*rowBytes:(r+1)*rowBytes], x, cols)
		if math.Abs(float64(out[r]-want)) > 1e-2*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("row %d: %v, want %v", r, out[r], want)
		}
	}
}
