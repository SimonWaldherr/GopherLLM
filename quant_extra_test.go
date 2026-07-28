package gopherllm

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// f16One is the little-endian f16 encoding of 1.0 (0x3C00); f16Half of 0.5.
func putF16One(b []byte)  { b[0], b[1] = 0x00, 0x3c }
func putF16Half(b []byte) { b[0], b[1] = 0x00, 0x38 }

func TestDequantRowQ4_1KnownValues(t *testing.T) {
	row := make([]byte, 20)
	putF16One(row[0:])  // d = 1
	putF16Half(row[2:]) // m = 0.5
	for i := 0; i < 16; i++ {
		row[4+i] = 0x31 // low nibble 1, high nibble 3
	}
	got := DequantRowQ4_1(row, 32)
	for i := 0; i < 16; i++ {
		if got[i] != 1.5 { // 1*1 + 0.5
			t.Fatalf("got[%d] = %v, want 1.5", i, got[i])
		}
		if got[16+i] != 3.5 { // 1*3 + 0.5
			t.Fatalf("got[%d] = %v, want 3.5", 16+i, got[16+i])
		}
	}
}

func TestDequantRowIQ4NLKnownValues(t *testing.T) {
	row := make([]byte, 18)
	putF16One(row)
	for i := 0; i < 16; i++ {
		row[2+i] = byte(i) | byte(15-i)<<4
	}
	got := DequantRowIQ4NL(row, 32)
	for i := 0; i < 16; i++ {
		if got[i] != iq4NLValues[i] || got[16+i] != iq4NLValues[15-i] {
			t.Fatalf("IQ4_NL[%d] = %v/%v, want %v/%v", i, got[i], got[16+i], iq4NLValues[i], iq4NLValues[15-i])
		}
	}
}

func TestDequantRowIQ4XSKnownValues(t *testing.T) {
	row := make([]byte, 136)
	putF16One(row)
	// Exercise every signed 6-bit sub-block scale, including the split
	// low-nibble / high-two-bit representation.
	codes := []byte{0, 1, 15, 16, 31, 32, 47, 63}
	var hi uint16
	for ib, code := range codes {
		hi |= uint16(code>>4) << (2 * ib)
		if ib&1 == 0 {
			row[4+ib/2] |= code & 0x0f
		} else {
			row[4+ib/2] |= (code & 0x0f) << 4
		}
		for i := 0; i < 16; i++ {
			row[8+ib*16+i] = 0x0f // low=15, high=0
		}
	}
	binary.LittleEndian.PutUint16(row[2:], hi)
	got := DequantRowIQ4XS(row, 256)
	for ib, code := range codes {
		scale := float32(int(code) - 32)
		for i := 0; i < 16; i++ {
			lo, high := got[ib*32+i], got[ib*32+16+i]
			if lo != scale*iq4NLValues[15] || high != scale*iq4NLValues[0] {
				t.Fatalf("IQ4_XS block %d value %d = %v/%v, want %v/%v", ib, i, lo, high, scale*iq4NLValues[15], scale*iq4NLValues[0])
			}
		}
	}
}

func TestDequantRowQ5_0KnownValues(t *testing.T) {
	row := make([]byte, 22)
	putF16One(row[0:])
	// All 5th bits set: every element is (q | 0x10) - 16 = low nibble value.
	row[2], row[3], row[4], row[5] = 0xff, 0xff, 0xff, 0xff
	for i := 0; i < 16; i++ {
		row[6+i] = 0x52 // low nibble 2, high nibble 5
	}
	got := DequantRowQ5_0(row, 32)
	for i := 0; i < 16; i++ {
		if got[i] != 2 { // (2|16)-16
			t.Fatalf("got[%d] = %v, want 2", i, got[i])
		}
		if got[16+i] != 5 { // (5|16)-16
			t.Fatalf("got[%d] = %v, want 5", 16+i, got[16+i])
		}
	}
	// All 5th bits clear: value is q - 16.
	row[2], row[3], row[4], row[5] = 0, 0, 0, 0
	got = DequantRowQ5_0(row, 32)
	if got[0] != -14 || got[16] != -11 {
		t.Fatalf("clear-high-bit: got[0]=%v want -14, got[16]=%v want -11", got[0], got[16])
	}
}

func TestDequantRowQ5_1KnownValues(t *testing.T) {
	row := make([]byte, 24)
	putF16One(row[0:])  // d = 1
	putF16Half(row[2:]) // m = 0.5
	row[4], row[5], row[6], row[7] = 0xff, 0xff, 0xff, 0xff
	for i := 0; i < 16; i++ {
		row[8+i] = 0x21
	}
	got := DequantRowQ5_1(row, 32)
	for i := 0; i < 16; i++ {
		if got[i] != 17.5 { // (1|16)*1 + 0.5
			t.Fatalf("got[%d] = %v, want 17.5", i, got[i])
		}
		if got[16+i] != 18.5 { // (2|16)*1 + 0.5
			t.Fatalf("got[%d] = %v, want 18.5", 16+i, got[16+i])
		}
	}
}

func TestDequantRowQ8_1KnownValues(t *testing.T) {
	row := make([]byte, 36)
	putF16Half(row[0:]) // d = 0.5
	for i := 0; i < 32; i++ {
		row[4+i] = byte(int8(i - 16))
	}
	got := DequantRowQ8_1(row, 32)
	for i := 0; i < 32; i++ {
		want := 0.5 * float32(i-16)
		if got[i] != want {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestDequantRowQ8KKnownValues(t *testing.T) {
	row := make([]byte, 292)
	binary.LittleEndian.PutUint32(row, math.Float32bits(0.25))
	for i := 0; i < 256; i++ {
		row[4+i] = byte(int8(i - 128))
	}
	got := DequantRowQ8K(row, 256)
	for i := range got {
		want := 0.25 * float32(i-128)
		if got[i] != want {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestDequantRowQ2KUsesScalesAndMins(t *testing.T) {
	row := make([]byte, 84)
	// Every sub-block: scale nibble 2, min nibble 1.
	for i := 0; i < 16; i++ {
		row[i] = 0x12
	}
	// All 2-bit quants = 3 (0b11111111 per byte).
	for i := 16; i < 80; i++ {
		row[i] = 0xff
	}
	putF16One(row[80:])  // d = 1
	putF16Half(row[82:]) // dmin = 0.5
	got := DequantRowQ2K(row, 256)
	for i, v := range got {
		// d*sc*q - dmin*m = 1*2*3 - 0.5*1 = 5.5
		if v != 5.5 {
			t.Fatalf("got[%d] = %v, want 5.5", i, v)
		}
	}
}

func TestDequantRowQ3KHighBitAndScales(t *testing.T) {
	row := make([]byte, 110)
	// hmask all set: no -4 offset anywhere.
	for i := 0; i < 32; i++ {
		row[i] = 0xff
	}
	// All 2-bit lows = 2 (0b10101010).
	for i := 32; i < 96; i++ {
		row[i] = 0xaa
	}
	// Packed scales: choose bytes so every 6-bit scale is 34 → (34-32)=2.
	// Low 4 bits of all 16 scales = 2 (bytes 0..7 = 0x22), high 2 bits = 10
	// binary (byte pattern 0b10101010 = 0xaa for bytes 8..11).
	for i := 96; i < 104; i++ {
		row[i] = 0x22
	}
	for i := 104; i < 108; i++ {
		row[i] = 0xaa
	}
	putF16One(row[108:]) // d = 1
	got := DequantRowQ3K(row, 256)
	for i, v := range got {
		// d*(sc-32)*q = 1*2*2 = 4
		if v != 4 {
			t.Fatalf("got[%d] = %v, want 4", i, v)
		}
	}
	// Clearing the hmask flips every quant to q-4 = -2 → value -4.
	for i := 0; i < 32; i++ {
		row[i] = 0
	}
	got = DequantRowQ3K(row, 256)
	for i, v := range got {
		if v != -4 {
			t.Fatalf("hmask-clear: got[%d] = %v, want -4", i, v)
		}
	}
}

// TestExtraQuantDotMatchesDequantizedDot is the standard differential check
// for every added format: Dot*(row, x) must equal DotF32(DequantRow*(row), x).
func TestExtraQuantDotMatchesDequantizedDot(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	randRow := func(n int) []byte {
		row := make([]byte, n)
		for i := range row {
			row[i] = byte(rng.Intn(256))
		}
		return row
	}
	// Overwrite f16 scale fields with small sane values so random bytes can't
	// produce Inf/NaN scales that would make the comparison meaningless.
	saneF16 := func(row []byte, blockBytes int, scaleOffs ...int) {
		for b := 0; b+blockBytes <= len(row); b += blockBytes {
			for _, off := range scaleOffs {
				putF16Half(row[b+off:])
			}
		}
	}

	cases := []struct {
		name    string
		cols    int
		row     []byte
		fix     func(row []byte)
		dot     func(row []byte, x []float32, cols int) float32
		dequant func(row []byte, cols int) []float32
	}{
		{"IQ4_NL", 64, randRow(2 * 18), func(r []byte) { saneF16(r, 18, 0) }, DotIQ4NLF32, DequantRowIQ4NL},
		{"IQ2_S", 512, randRow(2 * 82), func(r []byte) { saneF16(r, 82, 0) }, DotIQ2SF32, DequantRowIQ2S},
		{"IQ3_S", 512, randRow(2 * 110), func(r []byte) { saneF16(r, 110, 0) }, DotIQ3SF32, DequantRowIQ3S},
		{"IQ4_XS", 512, randRow(2 * 136), func(r []byte) { saneF16(r, 136, 0) }, DotIQ4XSF32, DequantRowIQ4XS},
		{"Q4_1", 64, randRow(2 * 20), func(r []byte) { saneF16(r, 20, 0, 2) }, DotQ4_1F32, DequantRowQ4_1},
		{"Q5_0", 64, randRow(2 * 22), func(r []byte) { saneF16(r, 22, 0) }, DotQ5_0F32, DequantRowQ5_0},
		{"Q5_1", 64, randRow(2 * 24), func(r []byte) { saneF16(r, 24, 0, 2) }, DotQ5_1F32, DequantRowQ5_1},
		{"Q8_1", 64, randRow(2 * 36), func(r []byte) { saneF16(r, 36, 0) }, DotQ8_1F32, DequantRowQ8_1},
		{"Q8_K", 512, randRow(2 * 292), func(r []byte) {
			for b := 0; b < len(r); b += 292 {
				binary.LittleEndian.PutUint32(r[b:], math.Float32bits(0.5))
			}
		}, DotQ8KF32, DequantRowQ8K},
		{"Q2_K", 512, randRow(2 * 84), func(r []byte) { saneF16(r, 84, 80, 82) }, DotQ2KF32, DequantRowQ2K},
		{"Q3_K", 512, randRow(2 * 110), func(r []byte) { saneF16(r, 110, 108) }, DotQ3KF32, DequantRowQ3K},
	}
	for _, c := range cases {
		c.fix(c.row)
		x := randomVec(rng, c.cols)
		want := DotF32(c.dequant(c.row, c.cols), x)
		got := c.dot(c.row, x, c.cols)
		if math.Abs(float64(got-want)) > 1e-2*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("%s: dot = %v, dequantized dot = %v", c.name, got, want)
		}
	}
}

func TestExtraQuantMatvecMatchesPerRowDot(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	const rows = 9
	type mv struct {
		name     string
		cols     int
		rowBytes int
		fixOffs  []int
		matvec   func(data []byte, x []float32, rows, cols int, out *[]float32)
		dot      func(row []byte, x []float32, cols int) float32
	}
	cases := []mv{
		{"IQ4_NL", 64, 2 * 18, []int{0}, MatvecIQ4NLInto, DotIQ4NLF32},
		{"IQ2_S", 512, 2 * 82, []int{0}, MatvecIQ2SInto, DotIQ2SF32},
		{"IQ3_S", 512, 2 * 110, []int{0}, MatvecIQ3SInto, DotIQ3SF32},
		{"IQ4_XS", 256, 136, []int{0}, MatvecIQ4XSInto, DotIQ4XSF32},
		{"Q4_1", 64, 2 * 20, []int{0, 2}, MatvecQ4_1Into, DotQ4_1F32},
		{"Q5_0", 64, 2 * 22, []int{0}, MatvecQ5_0Into, DotQ5_0F32},
		{"Q5_1", 64, 2 * 24, []int{0, 2}, MatvecQ5_1Into, DotQ5_1F32},
		{"Q8_1", 64, 2 * 36, []int{0}, MatvecQ8_1Into, DotQ8_1F32},
		{"Q8_K", 256, 292, nil, MatvecQ8KInto, DotQ8KF32},
		{"Q2_K", 256, 84, []int{80, 82}, MatvecQ2KInto, DotQ2KF32},
		{"Q3_K", 256, 110, []int{108}, MatvecQ3KInto, DotQ3KF32},
	}
	for _, c := range cases {
		blockBytes := c.rowBytes
		if c.cols > 256 {
			blockBytes = c.rowBytes / 2
		}
		data := make([]byte, rows*c.rowBytes)
		for i := range data {
			data[i] = byte(rng.Intn(256))
		}
		for b := 0; b+blockBytes <= len(data); b += blockBytes {
			for _, off := range c.fixOffs {
				putF16Half(data[b+off:])
			}
		}
		if c.name == "Q8_K" {
			for b := 0; b+292 <= len(data); b += 292 {
				binary.LittleEndian.PutUint32(data[b:], math.Float32bits(0.5))
			}
		}
		x := randomVec(rng, c.cols)
		out := []float32{}
		c.matvec(data, x, rows, c.cols, &out)
		for r := 0; r < rows; r++ {
			want := c.dot(data[r*c.rowBytes:(r+1)*c.rowBytes], x, c.cols)
			if math.Abs(float64(out[r]-want)) > 1e-3*math.Max(1, math.Abs(float64(want))) {
				t.Fatalf("%s row %d: matvec = %v, dot = %v", c.name, r, out[r], want)
			}
		}
	}
}

func TestIQSLoadRowMatvecAndForceF32(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	for _, tc := range []struct {
		name     string
		typ      GGMLType
		rowBytes int
		dequant  func([]byte, int) []float32
	}{
		{"IQ2_S", GGMLTypeIQ2_S, 82, DequantRowIQ2S},
		{"IQ3_S", GGMLTypeIQ3_S, 110, DequantRowIQ3S},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, tc.rowBytes)
			for i := range raw {
				raw[i] = byte(rng.Intn(256))
			}
			putF16Half(raw) // use a finite base scale
			data := buildGGUF(3, []ggufKV{{"general.architecture", ggufStr, "llama"}},
				[]ggufTensor{{name: "t", dims: []uint64{256, 1}, dtype: tc.typ, data: raw}})
			g, err := ParseGGUFQuiet(data)
			if err != nil {
				t.Fatal(err)
			}
			idx, inferred := indexTensors(g), inferTensorSizes(data, g)
			w, err := loadWeight(data, g.DataOffset, "t", idx, inferred, false, false, false, false)
			if err != nil {
				t.Fatal(err)
			}
			if w.Type != tc.typ || len(w.Raw) != len(raw) {
				t.Fatalf("loaded weight = %+v", w)
			}
			want := tc.dequant(raw, 256)
			got := w.Row(0, 256)
			assertFloatSlicesClose(t, "row", got, want)
			x := randomVec(rng, 256)
			matvec := w.Matvec(x)
			if wantDot := DotF32(want, x); math.Abs(float64(matvec[0]-wantDot)) > 2e-5*math.Max(1, math.Abs(float64(wantDot))) {
				t.Fatalf("matvec = %v, want %v", matvec[0], wantDot)
			}
			dense, err := loadWeight(data, g.DataOffset, "t", idx, inferred, true, false, false, false)
			if err != nil {
				t.Fatal(err)
			}
			assertFloatSlicesClose(t, "force_f32", dense.F32, want)
		})
	}
}

func TestIQ4XSLoadAndForceF32(t *testing.T) {
	raw := make([]byte, 136)
	putF16One(raw)
	for i := 0; i < 8; i++ {
		raw[4+i/2] |= byte(32+i) << (4 * (i & 1))
		for j := 0; j < 16; j++ {
			raw[8+i*16+j] = byte(j) | byte(15-j)<<4
		}
	}
	data := buildGGUF(3, []ggufKV{{"general.architecture", ggufStr, "llama"}},
		[]ggufTensor{{name: "t", dims: []uint64{256, 1}, dtype: GGMLTypeIQ4_XS, data: raw}})
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	idx, inferred := indexTensors(g), inferTensorSizes(data, g)
	w, err := loadWeight(data, g.DataOffset, "t", idx, inferred, false, false, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if w.Type != GGMLTypeIQ4_XS || len(w.Raw) != len(raw) {
		t.Fatalf("loaded IQ4_XS = %+v", w)
	}
	got := w.Row(0, 256)
	want := DequantRowIQ4XS(raw, 256)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("row[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	dense, err := loadWeight(data, g.DataOffset, "t", idx, inferred, true, false, false, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if dense.F32[i] != want[i] {
			t.Fatalf("force_f32[%d] = %v, want %v", i, dense.F32[i], want[i])
		}
	}
}

func TestF64LoadConversion(t *testing.T) {
	vals := []float64{1, -2.5, 0.25, math.Pi}
	raw := make([]byte, 8*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
	}
	data := buildGGUF(3, []ggufKV{{"general.architecture", ggufStr, "llama"}},
		[]ggufTensor{{name: "t", dims: []uint64{uint64(len(vals))}, dtype: GGMLTypeF64, data: raw}})
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	w, err := loadWeight(data, g.DataOffset, "t", indexTensors(g), inferTensorSizes(data, g), false, false, false, false)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vals {
		if w.F32[i] != float32(v) {
			t.Fatalf("w.F32[%d] = %v, want %v", i, w.F32[i], float32(v))
		}
	}
}

func TestBF16LoadConversion(t *testing.T) {
	// Build a tiny GGUF with one BF16 tensor and confirm exact conversion.
	vals := []float32{1.0, -2.0, 0.5, 3.0, -0.25, 0, 128, -1024}
	raw := make([]byte, 2*len(vals))
	for i, v := range vals {
		bits := math.Float32bits(v) >> 16 // exact for these power-of-two values
		raw[i*2] = byte(bits)
		raw[i*2+1] = byte(bits >> 8)
	}
	data := buildGGUF(3, []ggufKV{{"general.architecture", ggufStr, "llama"}},
		[]ggufTensor{{name: "t", dims: []uint64{uint64(len(vals))}, dtype: GGMLTypeBF16, data: raw}})
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	idx := indexTensors(g)
	w, err := loadWeight(data, g.DataOffset, "t", idx, inferTensorSizes(data, g), false, false, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if w.F32 == nil || len(w.F32) != len(vals) {
		t.Fatalf("weight = %+v", w)
	}
	for i, v := range vals {
		if w.F32[i] != v {
			t.Fatalf("w.F32[%d] = %v, want %v", i, w.F32[i], v)
		}
	}
}
