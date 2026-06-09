package main

import (
	"math"
	"testing"
)

func TestDotF32MatchesScalar(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 127, 256} {
		a := make([]float32, n)
		b := make([]float32, n+3)
		for i := range a {
			a[i] = float32(i%11) - 5.5
		}
		for i := range b {
			b[i] = 0.25*float32(i%13) - 1.5
		}
		got := DotF32(a, b)
		want := dotF32Scalar(a, b)
		if math.Abs(float64(got-want)) > 1e-4 {
			t.Fatalf("n=%d DotF32 = %v, want %v", n, got, want)
		}
		got = DotF32(b, a)
		want = dotF32Scalar(b, a)
		if math.Abs(float64(got-want)) > 1e-4 {
			t.Fatalf("reversed n=%d DotF32 = %v, want %v", n, got, want)
		}
	}
}

func TestVectorOpsMatchScalar(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 5, 15, 16, 17, 64} {
		base := make([]float32, n)
		x := make([]float32, n+2)
		for i := range base {
			base[i] = float32(i%9) - 4
		}
		for i := range x {
			x[i] = float32(i%7) - 2
		}

		got, want := append([]float32(nil), base...), append([]float32(nil), base...)
		AxpyF32(got, 0.75, x)
		axpyF32Scalar(want, 0.75, x)
		assertFloatSlicesClose(t, "axpy", got, want)

		got, want = append([]float32(nil), base...), append([]float32(nil), base...)
		ScaleF32(got, -1.25)
		scaleF32Scalar(want, -1.25)
		assertFloatSlicesClose(t, "scale", got, want)

		got, want = append([]float32(nil), base...), append([]float32(nil), base...)
		ScaleAddF32(got, 0.5, x)
		scaleAddF32Scalar(want, 0.5, x)
		assertFloatSlicesClose(t, "scaleAdd", got, want)

		if n > 2 {
			shortX := x[:n-2]
			got, want = append([]float32(nil), base...), append([]float32(nil), base...)
			AxpyF32(got, 0.75, shortX)
			axpyF32Scalar(want, 0.75, shortX)
			assertFloatSlicesClose(t, "axpy-short", got, want)

			got, want = append([]float32(nil), base...), append([]float32(nil), base...)
			ScaleAddF32(got, 0.5, shortX)
			scaleAddF32Scalar(want, 0.5, shortX)
			assertFloatSlicesClose(t, "scaleAdd-short", got, want)
		}
	}
}

func assertFloatSlicesClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func TestDequantRowQ4_0UsesHalfBlockLayout(t *testing.T) {
	row := make([]byte, 18)
	row[0], row[1] = 0x00, 0x3c // f16(1.0)
	for i := range 16 {
		row[2+i] = 0xa9 // low=9 -> 1, high=10 -> 2
	}

	got := DequantRowQ4_0(row, 32)
	for i := 0; i < 16; i++ {
		if got[i] != 1 {
			t.Fatalf("got[%d] = %v, want 1", i, got[i])
		}
	}
	for i := 16; i < 32; i++ {
		if got[i] != 2 {
			t.Fatalf("got[%d] = %v, want 2", i, got[i])
		}
	}
}

func TestDotQ4_0MatchesDequantizedDot(t *testing.T) {
	row := make([]byte, 18)
	row[0], row[1] = 0x00, 0x3c
	for i := range 16 {
		row[2+i] = byte(i<<4) | byte(15-i)
	}
	x := make([]float32, 32)
	for i := range x {
		x[i] = float32(i%9) - 4
	}

	want := DotF32(DequantRowQ4_0(row, 32), x)
	got := DotQ4_0F32(row, x, 32)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("dot = %v, want %v", got, want)
	}
}

func TestDequantRowQ4KUsesScalesAndMins(t *testing.T) {
	row := make([]byte, 144)
	row[0], row[1] = 0x00, 0x3c // d = 1
	row[2], row[3] = 0x00, 0x00 // dmin = 0
	copy(row[4:16], []byte{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1})
	for i := range row[16:] {
		row[16+i] = 0x21
	}

	got := DequantRowQ4K(row, 256)
	for i := 0; i < 256; i += 64 {
		for j := range 32 {
			if got[i+j] != 1 {
				t.Fatalf("got[%d] = %v, want 1", i+j, got[i+j])
			}
			if got[i+32+j] != 2 {
				t.Fatalf("got[%d] = %v, want 2", i+32+j, got[i+32+j])
			}
		}
	}
}

func TestDotQ4KMatchesDequantizedDot(t *testing.T) {
	row := make([]byte, 144)
	row[0], row[1] = 0x00, 0x3c
	copy(row[4:16], []byte{1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8})
	for i := range row[16:] {
		row[16+i] = byte(i)
	}
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i%7) - 3
	}

	want := DotF32(DequantRowQ4K(row, 256), x)
	got := DotQ4KF32(row, x, 256)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("dot = %v, want %v", got, want)
	}
}

func TestQ4KMatvec3MatchesSeparateMatvecs(t *testing.T) {
	const rows = 3
	const cols = 256
	qData := make([]byte, rows*144)
	kData := make([]byte, rows*144)
	vData := make([]byte, rows*144)
	fillQ4KRows := func(data []byte, seed byte) {
		for r := range rows {
			row := data[r*144 : (r+1)*144]
			row[0], row[1] = 0x00, 0x3c
			row[2], row[3] = 0x00, 0x00
			copy(row[4:16], []byte{1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8})
			for i := range row[16:] {
				row[16+i] = byte(int(seed) + r + i)
			}
		}
	}
	fillQ4KRows(qData, 1)
	fillQ4KRows(kData, 11)
	fillQ4KRows(vData, 21)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%17) / 9
	}

	wantQ, wantK, wantV := make([]float32, rows), make([]float32, rows), make([]float32, rows)
	MatvecQ4KInto(qData, x, rows, cols, &wantQ)
	MatvecQ4KInto(kData, x, rows, cols, &wantK)
	MatvecQ4KInto(vData, x, rows, cols, &wantV)

	gotQ, gotK, gotV := []float32{}, []float32{}, []float32{}
	ok := Q4KMatvec3Into(
		Q4KMatrix{Data: qData, Rows: rows, Cols: cols},
		Q4KMatrix{Data: kData, Rows: rows, Cols: cols},
		Q4KMatrix{Data: vData, Rows: rows, Cols: cols},
		x,
		&gotQ,
		&gotK,
		&gotV,
	)
	if !ok {
		t.Fatal("Q4KMatvec3Into returned false")
	}
	for i := range rows {
		if math.Abs(float64(gotQ[i]-wantQ[i])) > 1e-3 {
			t.Fatalf("q[%d] = %v, want %v", i, gotQ[i], wantQ[i])
		}
		if math.Abs(float64(gotK[i]-wantK[i])) > 1e-3 {
			t.Fatalf("k[%d] = %v, want %v", i, gotK[i], wantK[i])
		}
		if math.Abs(float64(gotV[i]-wantV[i])) > 1e-3 {
			t.Fatalf("v[%d] = %v, want %v", i, gotV[i], wantV[i])
		}
	}
}

func TestQ4KMatvec2MatchesSeparateMatvecs(t *testing.T) {
	const rowsA = 3
	const rowsB = 4
	const cols = 256
	aData := make([]byte, rowsA*144)
	bData := make([]byte, rowsB*144)
	fillQ4KRows := func(data []byte, rows int, seed byte) {
		for r := range rows {
			row := data[r*144 : (r+1)*144]
			row[0], row[1] = 0x00, 0x3c
			row[2], row[3] = 0x00, 0x00
			copy(row[4:16], []byte{1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8})
			for i := range row[16:] {
				row[16+i] = byte(int(seed) + r + i)
			}
		}
	}
	fillQ4KRows(aData, rowsA, 5)
	fillQ4KRows(bData, rowsB, 25)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%17) / 11
	}

	wantA, wantB := []float32{}, []float32{}
	MatvecQ4KInto(aData, x, rowsA, cols, &wantA)
	MatvecQ4KInto(bData, x, rowsB, cols, &wantB)
	gotA, gotB := []float32{}, []float32{}
	if !MatvecQ4K2Into(aData, rowsA, cols, bData, rowsB, cols, x, &gotA, &gotB) {
		t.Fatal("MatvecQ4K2Into returned false")
	}
	assertFloatSlicesClose(t, "q4k2-a", gotA, wantA)
	assertFloatSlicesClose(t, "q4k2-b", gotB, wantB)
}

func TestDotQ6KMatchesDequantizedDot(t *testing.T) {
	row := make([]byte, 210)
	for i := range 128 {
		row[i] = byte(i)
	}
	for i := range 64 {
		row[128+i] = byte(i * 3)
	}
	for i := range 16 {
		row[192+i] = byte(int8(i - 8))
	}
	row[208], row[209] = 0x00, 0x3c
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i%11) / 5
	}

	want := DotF32(DequantRowQ6K(row, 256), x)
	got := DotQ6KF32(row, x, 256)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("dot = %v, want %v", got, want)
	}
}

func TestQ6KMatvec2MatchesSeparateMatvecs(t *testing.T) {
	const rowsA = 2
	const rowsB = 3
	const cols = 256
	aData := make([]byte, rowsA*210)
	bData := make([]byte, rowsB*210)
	fillQ6KRows := func(data []byte, rows int, seed byte) {
		for r := range rows {
			row := data[r*210 : (r+1)*210]
			for i := range 128 {
				row[i] = byte(int(seed) + r + i)
			}
			for i := range 64 {
				row[128+i] = byte(int(seed) + r*3 + i*5)
			}
			for i := range 16 {
				row[192+i] = byte(int8(i - 8))
			}
			row[208], row[209] = 0x00, 0x3c
		}
	}
	fillQ6KRows(aData, rowsA, 7)
	fillQ6KRows(bData, rowsB, 33)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%13) / 9
	}

	wantA, wantB := []float32{}, []float32{}
	MatvecQ6KInto(aData, x, rowsA, cols, &wantA)
	MatvecQ6KInto(bData, x, rowsB, cols, &wantB)
	gotA, gotB := []float32{}, []float32{}
	if !MatvecQ6K2Into(aData, rowsA, cols, bData, rowsB, cols, x, &gotA, &gotB) {
		t.Fatal("MatvecQ6K2Into returned false")
	}
	assertFloatSlicesClose(t, "q6k2-a", gotA, wantA)
	assertFloatSlicesClose(t, "q6k2-b", gotB, wantB)
}

func TestQ6KMatvec3MatchesSeparateMatvecs(t *testing.T) {
	const rowsA = 2
	const rowsB = 3
	const rowsC = 4
	const cols = 256
	aData := make([]byte, rowsA*210)
	bData := make([]byte, rowsB*210)
	cData := make([]byte, rowsC*210)
	fillQ6KRows := func(data []byte, rows int, seed byte) {
		for r := range rows {
			row := data[r*210 : (r+1)*210]
			for i := range 128 {
				row[i] = byte(int(seed) + r + i)
			}
			for i := range 64 {
				row[128+i] = byte(int(seed) + r*3 + i*5)
			}
			for i := range 16 {
				row[192+i] = byte(int8(i - 8))
			}
			row[208], row[209] = 0x00, 0x3c
		}
	}
	fillQ6KRows(aData, rowsA, 7)
	fillQ6KRows(bData, rowsB, 33)
	fillQ6KRows(cData, rowsC, 61)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%13) / 9
	}

	wantA, wantB, wantC := []float32{}, []float32{}, []float32{}
	MatvecQ6KInto(aData, x, rowsA, cols, &wantA)
	MatvecQ6KInto(bData, x, rowsB, cols, &wantB)
	MatvecQ6KInto(cData, x, rowsC, cols, &wantC)
	gotA, gotB, gotC := []float32{}, []float32{}, []float32{}
	if !MatvecQ6K3Into(aData, rowsA, cols, bData, rowsB, cols, cData, rowsC, cols, x, &gotA, &gotB, &gotC) {
		t.Fatal("MatvecQ6K3Into returned false")
	}
	assertFloatSlicesClose(t, "q6k3-a", gotA, wantA)
	assertFloatSlicesClose(t, "q6k3-b", gotB, wantB)
	assertFloatSlicesClose(t, "q6k3-c", gotC, wantC)
}

func TestDotQ5KMatchesDequantizedDot(t *testing.T) {
	row := make([]byte, 176)
	row[0], row[1] = 0x00, 0x3c
	copy(row[4:16], []byte{1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8})
	for i := range 32 {
		row[16+i] = byte(i * 7)
	}
	for i := range 128 {
		row[48+i] = byte(i * 3)
	}
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i%13) / 7
	}

	want := DotF32(DequantRowQ5K(row, 256), x)
	got := DotQ5KF32(row, x, 256)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("dot = %v, want %v", got, want)
	}
}

func TestDequantRowMXFP4UsesTrailingScaleByte(t *testing.T) {
	row := make([]byte, 17)
	for i := range 16 {
		row[i] = 0x21
	}
	row[16] = 127 // scale = 1

	got := DequantRowMXFP4(row, 32)
	for i := 0; i < 32; i += 2 {
		if got[i] != 0.5 {
			t.Fatalf("got[%d] = %v, want 0.5", i, got[i])
		}
		if got[i+1] != 1 {
			t.Fatalf("got[%d] = %v, want 1", i+1, got[i+1])
		}
	}
}
