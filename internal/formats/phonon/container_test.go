package phonon

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"testing"
)

func makeContainer(t *testing.T, idx []entry, data ...[]byte) []byte {
	t.Helper()
	body, err := json.Marshal(header{Format: Format, Index: idx})
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(body)))
	out = append(out, body...)
	for _, b := range data {
		out = append(out, b...)
	}
	return out
}

func halfWords(values ...uint16) []byte {
	var out []byte
	for _, value := range values {
		out = binary.LittleEndian.AppendUint16(out, value)
	}
	return out
}

func TestContainerDecodesOfficialEncodings(t *testing.T) {
	// Little-endian groups of four 6-bit values. The first packed group
	// crosses a row boundary, and the final two codes are padding.
	int6 := append([]byte{0x00, 0xf8, 0x7f, 0x61, 0x04, 0x2f}, halfWords(0x3800, 0x4000)...)
	int8 := append([]byte{0x80, 0x7f, 0xff, 0x01}, halfWords(0x3800, 0x4000)...)
	// Two rows of seven weights. There are five and four nonzero signs
	// respectively, so magnitude bits cross both a row and a byte.
	five := append([]byte{177, 7, 190, 7, 0xba, 0}, halfWords(0x3800, 0x3c00, 0x4000, 0x4200)...)
	allZero := append([]byte{4}, halfWords(0x3c00, 0x4000)...)
	fp16 := halfWords(0x0001, 0xc100)
	counter := halfWords(0x7c00)
	fixtures := []struct {
		entry
		name string
		data []byte
		want []float32
	}{
		{entry{Name: "encoder.a", Kind: "five_value", Shape: []int{2, 7}}, "encoder.a.weight", five,
			[]float32{-.5, 2, 0, -.5, 2, 0, 2, 0, -3, -1, 0, 3, 0, 1}},
		{entry{Name: "encoder.b", Kind: "five_value", Shape: []int{1, 2}}, "encoder.b.weight", allZero, []float32{0, 0}},
		{entry{Name: "a", Kind: "int6", Shape: []int{2, 1, 3}}, "a", int6, []float32{-16, 0, 15.5, -2, 2, -30}},
		{entry{Name: "b", Kind: "int8", Shape: []int{2, 2}}, "b", int8, []float32{-64, 63.5, -2, 2}},
		{entry{Name: "bias", Kind: "fp16", Shape: []int{2}}, "bias", fp16, []float32{0x1p-24, -2.5}},
		{entry{Name: "counter", Kind: "fp16", Shape: []int{}}, "counter", counter, []float32{float32(math.Inf(1))}},
	}
	var index []entry
	var blobs [][]byte
	for _, f := range fixtures {
		e := f.entry
		e.Bytes = len(f.data)
		index = append(index, e)
		blobs = append(blobs, f.data)
	}
	m, err := Parse(makeContainer(t, index, blobs...))
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Tensors) != len(fixtures) {
		t.Fatalf("got %d tensors, want %d", len(m.Tensors), len(fixtures))
	}
	for _, f := range fixtures {
		t.Run(f.name, func(t *testing.T) {
			tensor, ok := m.Tensors[f.name]
			if !ok || !slices.Equal(tensor.Shape, f.Shape) || tensor.NumElements() != len(f.want) {
				t.Fatalf("invalid tensor metadata: %+v", tensor)
			}
			out, err := tensor.Decode()
			if err != nil || !slices.Equal(out, f.want) {
				t.Fatalf("Decode() = %v, %v, want %v", out, err, f.want)
			}
			if err := tensor.DecodeInto(out[:0]); err == nil {
				t.Fatal("short destination accepted")
			}
			if tensor.Kind == "five_value" {
				cols := tensor.Shape[1]
				for row := range tensor.Shape[0] {
					dst := make([]float32, cols)
					tensor.DecodeFiveValueRows(dst, row, row+1)
					for i, v := range dst {
						if math.Float32bits(v) != math.Float32bits(out[row*cols+i]) {
							t.Fatalf("packed row %d column %d = %g, want %g", row, i, v, out[row*cols+i])
						}
					}
				}
				half, err := tensor.DecodeF16()
				if err != nil {
					t.Fatal(err)
				}
				for i, want := range f.want {
					if halfAt(half, i) != want {
						t.Fatalf("FP16 weight %d = %g, want %g", i, halfAt(half, i), want)
					}
				}
			}
		})
	}
}

func TestContainerRejectsCorruption(t *testing.T) {
	fp16 := entry{Name: "a", Kind: "fp16", Shape: []int{1}, Bytes: 2}
	allZero := []byte{1, 0, 0, 0, 0}
	invalid := []struct {
		name string
		data []byte
		want string
	}{
		{"short header", []byte{1}, "truncated"},
		{"header over limit", binary.LittleEndian.AppendUint64(nil, maxHeaderBytes+1), "header length"},
		{"invalid JSON", append(binary.LittleEndian.AppendUint64(nil, 1), '{'), "invalid header"},
		{"empty index", makeContainer(t, nil), "empty"},
		{"trailing bytes", makeContainer(t, []entry{fp16}, []byte{0, 0, 1}), "trailing"},
		{"truncated tensor", makeContainer(t, []entry{fp16}, []byte{1}), "payload length"},
		{"unknown kind", makeContainer(t, []entry{{"a", "int4", []int{1, 1}, 1}}, []byte{0}), "unsupported encoding"},
		{"invalid shape", makeContainer(t, []entry{{"a", "fp16", []int{1, 0}, 2}}, []byte{0, 0}), "shape"},
		{"missing shape", makeContainer(t, []entry{{"a", "fp16", nil, 2}}, []byte{0, 0}), "shape"},
		{"overflow shape", makeContainer(t, []entry{{"a", "fp16", []int{math.MaxInt, 2}, 2}}, []byte{0, 0}), "shape"},
		{"invalid int6 length", makeContainer(t, []entry{{"a", "int6", []int{1, 1}, 4}}, []byte{0, 0, 0, 0}), "integer payload"},
		{"invalid plane", makeContainer(t, []entry{{"a", "five_value", []int{1, 1}, 5}}, []byte{0, 0, 0, 0, 0}), "magnitude"},
		{"invalid ternary digit", makeContainer(t, []entry{{"a", "five_value", []int{1, 1}, 5}}, []byte{243, 0, 0, 0, 0}), "ternary"},
		{"invalid scalar count", makeContainer(t, []entry{{"a", "int8", []int{}, 2}}, []byte{0, 0}), "no rows"},
		{"non-finite scale", makeContainer(t, []entry{{"a", "int8", []int{1, 1}, 3}}, []byte{0, 0, 0x7c}), "non-finite"},
		{"normalized duplicate", makeContainer(t,
			[]entry{{"a", "five_value", []int{1, 1}, 5}, {"a.weight", "fp16", []int{1}, 2}},
			allZero, []byte{0, 0}), "duplicate"},
	}
	for _, tc := range invalid {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Parse(tc.data); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Parse() returned %v, want error containing %q", err, tc.want)
			}
		})
	}
}

func TestCheckHeaderWithoutTensorPayload(t *testing.T) {
	data := makeContainer(t, []entry{{"a", "fp16", []int{1}, 2}})
	if err := CheckHeader(data); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(data); err == nil {
		t.Fatal("Parse accepted missing tensor payload")
	}
	if err := CheckHeader(data[:len(data)-1]); err == nil {
		t.Fatal("CheckHeader accepted truncated JSON")
	}
}
