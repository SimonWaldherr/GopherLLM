package gopherllm

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDequantRowQ1_0KnownValues(t *testing.T) {
	row := make([]byte, 18)
	binary.LittleEndian.PutUint16(row, F32ToF16(2))
	for i := range row[2:] {
		row[2+i] = 0xaa
	}
	got := DequantRowQ1_0(row, 128)
	for i, value := range got {
		want := float32(-2)
		if i&1 != 0 {
			want = 2
		}
		if value != want {
			t.Fatalf("Q1_0[%d] = %v, want %v", i, value, want)
		}
	}
}

func TestDequantRowQ2_0KnownValues(t *testing.T) {
	row := make([]byte, 18)
	binary.LittleEndian.PutUint16(row, F32ToF16(0.5))
	for i := range row[2:] {
		row[2+i] = 0xe4 // little-endian 2-bit fields: 00, 01, 10, 11
	}
	got := DequantRowQ2_0(row, 64)
	want := [...]float32{-0.5, 0, 0.5, 1}
	for i, value := range got {
		if value != want[i&3] {
			t.Fatalf("Q2_0[%d] = %v, want %v", i, value, want[i&3])
		}
	}
}

func TestDequantRowTQ1_0KnownValues(t *testing.T) {
	row := make([]byte, 54)
	for i := 32; i < 48; i++ {
		row[i] = 255 // five +1 scaled base-3 digits
	}
	for i := 48; i < 52; i++ {
		row[i] = 128 // four zero-valued scaled base-3 digits
	}
	binary.LittleEndian.PutUint16(row[52:], F32ToF16(2))
	got := DequantRowTQ1_0(row, 256)
	for i, value := range got {
		want := float32(-2)
		if i >= 160 && i < 240 {
			want = 2
		} else if i >= 240 {
			want = 0
		}
		if value != want {
			t.Fatalf("TQ1_0[%d] = %v, want %v", i, value, want)
		}
	}
}

func TestDequantRowTQ2_0KnownValues(t *testing.T) {
	row := make([]byte, 66)
	for i := range row[:64] {
		row[i] = 0xe4 // four 2-bit planes: -1, 0, +1, +2
	}
	binary.LittleEndian.PutUint16(row[64:], F32ToF16(0.5))
	got := DequantRowTQ2_0(row, 256)
	want := [...]float32{-0.5, 0, 0.5, 1}
	for i, value := range got {
		if value != want[(i%128)/32] {
			t.Fatalf("TQ2_0[%d] = %v, want %v", i, value, want[(i%128)/32])
		}
	}
}

func TestLowBitDotsMatchDequantizedRows(t *testing.T) {
	q1 := make([]byte, 2*18)
	q2 := make([]byte, 2*18)
	for b := range 2 {
		binary.LittleEndian.PutUint16(q1[b*18:], F32ToF16(0.25+float32(b)*0.25))
		binary.LittleEndian.PutUint16(q2[b*18:], F32ToF16(0.5+float32(b)*0.25))
		for i := 2; i < 18; i++ {
			q1[b*18+i] = byte(i*37 + b*11)
			q2[b*18+i] = byte(i*29 + b*17)
		}
	}
	x1 := make([]float32, 256)
	x2 := make([]float32, 128)
	for i := range x1 {
		x1[i] = float32((i%19)-9) / 7
	}
	for i := range x2 {
		x2[i] = float32((i%13)-6) / 5
	}
	tq1 := make([]byte, 2*54)
	tq2 := make([]byte, 2*66)
	for b := range 2 {
		for i := 0; i < 48; i++ {
			tq1[b*54+i] = byte(i*37 + b*11)
		}
		for i := 48; i < 52; i++ {
			tq1[b*54+i] = byte(i*29 + b*7)
		}
		binary.LittleEndian.PutUint16(tq1[b*54+52:], F32ToF16(0.25+float32(b)*0.25))
		for i := 0; i < 64; i++ {
			tq2[b*66+i] = byte(i*29 + b*17)
		}
		binary.LittleEndian.PutUint16(tq2[b*66+64:], F32ToF16(0.5+float32(b)*0.25))
	}
	xt := make([]float32, 512)
	for i := range xt {
		xt[i] = float32((i%23)-11) / 9
	}
	cases := []struct {
		name string
		row  []byte
		x    []float32
		dot  func([]byte, []float32, int) float32
		deq  func([]byte, int) []float32
	}{
		{"q1_0", q1, x1, DotQ1_0F32, DequantRowQ1_0},
		{"q2_0", q2, x2, DotQ2_0F32, DequantRowQ2_0},
		{"tq1_0", tq1, xt, DotTQ1_0F32, DequantRowTQ1_0},
		{"tq2_0", tq2, xt, DotTQ2_0F32, DequantRowTQ2_0},
	}
	for _, tc := range cases {
		got := tc.dot(tc.row, tc.x, len(tc.x))
		want := DotF32(tc.deq(tc.row, len(tc.x)), tc.x)
		if diff := math.Abs(float64(got - want)); diff > 1e-5*math.Max(1, math.Abs(float64(want))) {
			t.Errorf("%s dot = %v, dequantized dot = %v", tc.name, got, want)
		}
	}
}

func TestLowBitWeightDispatchAndBatch(t *testing.T) {
	const rows, cols, positions = 3, 256, 4
	makeData := func(typ GGMLType) []byte {
		rowBytes, ok := typ.DataSize(cols)
		if !ok {
			t.Fatalf("%s has no row size", typ)
		}
		data := make([]byte, rows*rowBytes)
		for r := range rows {
			switch typ {
			case GGMLTypeTQ1_0:
				base := r * rowBytes
				for i := 0; i < 48; i++ {
					data[base+i] = byte(r*53 + i*17)
				}
				for i := 48; i < 52; i++ {
					data[base+i] = byte(r*31 + i*11)
				}
				binary.LittleEndian.PutUint16(data[base+52:], F32ToF16(0.25+float32(r)/16))
			case GGMLTypeTQ2_0:
				base := r * rowBytes
				for i := 0; i < 64; i++ {
					data[base+i] = byte(r*53 + i*17)
				}
				binary.LittleEndian.PutUint16(data[base+64:], F32ToF16(0.25+float32(r)/16))
			default:
				for b := 0; b < rowBytes/18; b++ {
					base := r*rowBytes + b*18
					binary.LittleEndian.PutUint16(data[base:], F32ToF16(0.25+float32(r+b)/16))
					for i := 2; i < 18; i++ {
						data[base+i] = byte(r*53 + b*31 + i*17)
					}
				}
			}
		}
		return data
	}
	xs := make([][]float32, positions)
	for p := range xs {
		xs[p] = make([]float32, cols)
		for i := range xs[p] {
			xs[p][i] = float32(((p+1)*i)%23-11) / 9
		}
	}
	for _, typ := range []GGMLType{GGMLTypeTQ1_0, GGMLTypeTQ2_0, GGMLTypeQ1_0, GGMLTypeQ2_0} {
		t.Run(typ.String(), func(t *testing.T) {
			w := Weight{Raw: makeData(typ), Type: typ, Rows: rows, Cols: cols}
			row := make([]float32, cols)
			w.RowInto(1, cols, &row)
			rowBytes, _ := typ.DataSize(cols)
			var wantRow []float32
			switch typ {
			case GGMLTypeTQ1_0:
				wantRow = DequantRowTQ1_0(w.Raw[rowBytes:2*rowBytes], cols)
			case GGMLTypeTQ2_0:
				wantRow = DequantRowTQ2_0(w.Raw[rowBytes:2*rowBytes], cols)
			case GGMLTypeQ1_0:
				wantRow = DequantRowQ1_0(w.Raw[rowBytes:2*rowBytes], cols)
			case GGMLTypeQ2_0:
				wantRow = DequantRowQ2_0(w.Raw[rowBytes:2*rowBytes], cols)
			}
			assertFloatSlicesClose(t, "row", row, wantRow)

			want := make([][]float32, positions)
			got := make([][]float32, positions)
			for p := range positions {
				want[p] = w.Matvec(xs[p])
				got[p] = make([]float32, rows)
			}
			matvecBatch(w, xs, got)
			for p := range positions {
				assertFloatSlicesClose(t, "batch", got[p], want[p])
			}
		})
	}
}

func TestLowBitGGUFLoadAndForceF32(t *testing.T) {
	const rows = 2
	for _, tc := range []struct {
		typ     GGMLType
		cols    int
		bytes   int
		scaleAt int
		dequant func([]byte, int) []float32
	}{
		{GGMLTypeTQ1_0, 256, 54, 52, DequantRowTQ1_0},
		{GGMLTypeTQ2_0, 256, 66, 64, DequantRowTQ2_0},
		{GGMLTypeQ1_0, 128, 18, 0, DequantRowQ1_0},
		{GGMLTypeQ2_0, 64, 18, 0, DequantRowQ2_0},
	} {
		t.Run(tc.typ.String(), func(t *testing.T) {
			raw := make([]byte, rows*tc.bytes)
			for r := range rows {
				base := r * tc.bytes
				for i := 0; i < tc.bytes-2; i++ {
					raw[base+i] = byte(r*47 + i*23)
				}
				binary.LittleEndian.PutUint16(raw[base+tc.scaleAt:], F32ToF16(0.25+float32(r)*0.25))
			}
			data := buildGGUF(3, []ggufKV{{"general.architecture", ggufStr, "llama"}},
				[]ggufTensor{{name: "t", dims: []uint64{uint64(tc.cols), rows}, dtype: tc.typ, data: raw}})
			g, err := ParseGGUFQuiet(data)
			if err != nil {
				t.Fatal(err)
			}
			idx, inferred := indexTensors(g), inferTensorSizes(data, g)
			w, err := loadWeight(data, g.DataOffset, "t", idx, inferred, false, false, false, false)
			if err != nil {
				t.Fatal(err)
			}
			if w.Type != tc.typ || w.Rows != rows || w.Cols != tc.cols || len(w.Raw) != len(raw) {
				t.Fatalf("loaded weight = %+v", w)
			}

			want := tc.dequant(raw, rows*tc.cols)
			for r := range rows {
				gotRow := w.Row(r, tc.cols)
				assertFloatSlicesClose(t, "row", gotRow, want[r*tc.cols:(r+1)*tc.cols])
			}
			x := make([]float32, tc.cols)
			for i := range x {
				x[i] = float32(i%17-8) / 9
			}
			gotMatvec := w.Matvec(x)
			for r := range rows {
				wantDot := DotF32(want[r*tc.cols:(r+1)*tc.cols], x)
				if diff := math.Abs(float64(gotMatvec[r] - wantDot)); diff > 1e-5*math.Max(1, math.Abs(float64(wantDot))) {
					t.Fatalf("matvec row %d = %v, want %v", r, gotMatvec[r], wantDot)
				}
			}

			dense, err := loadWeight(data, g.DataOffset, "t", idx, inferred, true, false, false, false)
			if err != nil {
				t.Fatal(err)
			}
			assertFloatSlicesClose(t, "force_f32", dense.F32, want)
		})
	}
}
