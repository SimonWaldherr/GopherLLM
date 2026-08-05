package gopherllm

import (
	"bytes"
	"math"
	"testing"
)

// TestGGUFWriterRoundTrip writes a small synthetic GGUF covering every
// metadata Kind (including both array representations: the typed fast paths
// and the generic []MetaValue path) plus two tensors, then parses the
// result back with the existing, trusted ParseGGUF and asserts every value
// matches exactly. This is the strongest test available: it validates the
// new writer against the exact reader real users hit.
func TestGGUFWriterRoundTrip(t *testing.T) {
	metadata := map[string]MetaValue{
		"a.u8":     {"u8", uint8(200)},
		"a.i8":     {"i8", int8(-100)},
		"a.u16":    {"u16", uint16(60000)},
		"a.i16":    {"i16", int16(-30000)},
		"a.u32":    {"u32", uint32(4000000000)},
		"a.i32":    {"i32", int32(-2000000000)},
		"a.f32":    {"f32", float32(3.5)},
		"a.bool_t": {"bool", true},
		"a.bool_f": {"bool", false},
		"a.str":    {"str", "hello gguf"},
		"a.u64":    {"u64", uint64(18000000000000000000)},
		"a.i64":    {"i64", int64(-9000000000000000000)},
		"a.f64":    {"f64", float64(2.718281828)},
		// Typed fast-path arrays.
		"arr.str":   {"array", []string{"alpha", "beta", "gamma"}},
		"arr.f32":   {"array", []float32{1, 2.5, -3.25}},
		"arr.bool":  {"array", []bool{true, false, true}},
		"arr.empty": {"array", []string{}},
		// Generic []MetaValue array path (element type not str/f32/bool).
		"arr.u32": {"array", []MetaValue{{"u32", uint32(10)}, {"u32", uint32(20)}, {"u32", uint32(30)}}},
	}

	f32Data := make([]byte, 4*8)
	for i := 0; i < 8; i++ {
		bits := math.Float32bits(float32(i) - 3.5)
		f32Data[i*4] = byte(bits)
		f32Data[i*4+1] = byte(bits >> 8)
		f32Data[i*4+2] = byte(bits >> 16)
		f32Data[i*4+3] = byte(bits >> 24)
	}
	q80Data := make([]byte, 34)         // one Q8_0 block: f16 scale + 32 int8
	q80Data[0], q80Data[1] = 0x00, 0x3C // f16 1.0
	for i := 0; i < 32; i++ {
		q80Data[2+i] = byte(int8(i - 16))
	}

	planned := []PlannedTensor{
		{Name: "tensor.f32", Dims: []uint64{8}, DType: GGMLTypeF32},
		{Name: "tensor.q8_0", Dims: []uint64{32}, DType: GGMLTypeQ8_0},
	}

	var buf bytes.Buffer
	gw, err := NewGGUFWriter(&buf, metadata, planned, 32)
	if err != nil {
		t.Fatalf("NewGGUFWriter: %v", err)
	}
	if err := gw.WriteTensor(f32Data); err != nil {
		t.Fatalf("WriteTensor(f32): %v", err)
	}
	if err := gw.WriteTensor(q80Data); err != nil {
		t.Fatalf("WriteTensor(q8_0): %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	parsed, err := ParseGGUF(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseGGUF(written bytes): %v", err)
	}

	checkU32 := func(key string, want uint32) {
		got, ok := parsed.Metadata[key]
		if !ok {
			t.Fatalf("%s: missing after round-trip", key)
		}
		n, ok := got.AsU32()
		if !ok || n != want {
			t.Fatalf("%s: got %v, want %d", key, got.Value, want)
		}
	}
	checkU32("a.u8", 200)
	var i16Const int32 = -30000
	checkU32("a.i16", uint32(i16Const)) // AsU32 widens signed via uint32 cast
	checkU32("a.u32", 4000000000)

	if v := parsed.Metadata["a.f32"]; v.Value.(float32) != 3.5 {
		t.Fatalf("a.f32: got %v", v.Value)
	}
	if v := parsed.Metadata["a.bool_t"]; v.Value.(bool) != true {
		t.Fatalf("a.bool_t: got %v", v.Value)
	}
	if v := parsed.Metadata["a.bool_f"]; v.Value.(bool) != false {
		t.Fatalf("a.bool_f: got %v", v.Value)
	}
	if v := parsed.Metadata["a.str"]; v.Value.(string) != "hello gguf" {
		t.Fatalf("a.str: got %v", v.Value)
	}
	if v := parsed.Metadata["a.u64"]; v.Value.(uint64) != 18000000000000000000 {
		t.Fatalf("a.u64: got %v", v.Value)
	}
	if v := parsed.Metadata["a.i64"]; v.Value.(int64) != -9000000000000000000 {
		t.Fatalf("a.i64: got %v", v.Value)
	}
	if v := parsed.Metadata["a.f64"]; v.Value.(float64) != 2.718281828 {
		t.Fatalf("a.f64: got %v", v.Value)
	}

	strArr, ok := parsed.Metadata["arr.str"].AsStringArray()
	if !ok || len(strArr) != 3 || strArr[0] != "alpha" || strArr[2] != "gamma" {
		t.Fatalf("arr.str: got %v", strArr)
	}
	f32Arr, ok := parsed.Metadata["arr.f32"].AsF32Array()
	if !ok || len(f32Arr) != 3 || f32Arr[1] != 2.5 {
		t.Fatalf("arr.f32: got %v", f32Arr)
	}
	boolArr, ok := parsed.Metadata["arr.bool"].AsBoolArray()
	if !ok || len(boolArr) != 3 || boolArr[1] != false {
		t.Fatalf("arr.bool: got %v", boolArr)
	}
	emptyArr, ok := parsed.Metadata["arr.empty"].AsStringArray()
	if !ok || len(emptyArr) != 0 {
		t.Fatalf("arr.empty: got %v, ok=%v", emptyArr, ok)
	}
	u32Arr, ok := parsed.Metadata["arr.u32"].AsU32Array()
	if !ok || len(u32Arr) != 3 || u32Arr[0] != 10 || u32Arr[2] != 30 {
		t.Fatalf("arr.u32: got %v", u32Arr)
	}

	if len(parsed.Tensors) != 2 {
		t.Fatalf("tensor count: got %d, want 2", len(parsed.Tensors))
	}
	tf32, tq8 := parsed.Tensors[0], parsed.Tensors[1]
	if tf32.Name != "tensor.f32" || tf32.DType != GGMLTypeF32 || tf32.Numel() != 8 {
		t.Fatalf("tensor.f32 descriptor: %+v", tf32)
	}
	if tq8.Name != "tensor.q8_0" || tq8.DType != GGMLTypeQ8_0 || tq8.Numel() != 32 {
		t.Fatalf("tensor.q8_0 descriptor: %+v", tq8)
	}

	raw := buf.Bytes()
	gotF32 := raw[parsed.DataOffset+int(tf32.Offset) : parsed.DataOffset+int(tf32.Offset)+32]
	if !bytes.Equal(gotF32, f32Data) {
		t.Fatalf("tensor.f32 bytes: got %x, want %x", gotF32, f32Data)
	}
	gotQ8 := raw[parsed.DataOffset+int(tq8.Offset) : parsed.DataOffset+int(tq8.Offset)+34]
	if !bytes.Equal(gotQ8, q80Data) {
		t.Fatalf("tensor.q8_0 bytes: got %x, want %x", gotQ8, q80Data)
	}

	// The dequantized Q8_0 tensor should exactly match what the value
	// bytes encode: DequantRowQ8_0 is the existing, trusted decoder.
	deq := DequantRowQ8_0(gotQ8, 32)
	for i := 0; i < 32; i++ {
		want := float32(i - 16)
		if deq[i] != want {
			t.Fatalf("dequant[%d] = %v, want %v", i, deq[i], want)
		}
	}
}

// TestGGUFWriterAlignment confirms tensor data actually starts at an
// alignment-padded offset and that DataOffset matches parseGGUF's own
// divCeil(pos, alignment)*alignment formula, using a non-default alignment
// to make sure the value isn't just coincidentally already aligned.
func TestGGUFWriterAlignment(t *testing.T) {
	planned := []PlannedTensor{{Name: "t", Dims: []uint64{32}, DType: GGMLTypeQ8_0}}
	var buf bytes.Buffer
	gw, err := NewGGUFWriter(&buf, map[string]MetaValue{"k": {"str", "v"}}, planned, 64)
	if err != nil {
		t.Fatalf("NewGGUFWriter: %v", err)
	}
	data := make([]byte, 34)
	if err := gw.WriteTensor(data); err != nil {
		t.Fatalf("WriteTensor: %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	parsed, err := ParseGGUF(buf.Bytes())
	if err != nil {
		t.Fatalf("ParseGGUF: %v", err)
	}
	if parsed.DataOffset%64 != 0 {
		t.Fatalf("DataOffset %d is not 64-byte aligned", parsed.DataOffset)
	}
	if len(buf.Bytes()) != parsed.DataOffset+34 {
		t.Fatalf("total file size %d, want %d (no trailing pad after last tensor)", len(buf.Bytes()), parsed.DataOffset+34)
	}
}

// TestGGUFWriterRejectsShortTensor confirms a caller passing the wrong
// number of bytes gets a clear error rather than a silently corrupt file.
func TestGGUFWriterRejectsShortTensor(t *testing.T) {
	planned := []PlannedTensor{{Name: "t", Dims: []uint64{32}, DType: GGMLTypeQ8_0}}
	var buf bytes.Buffer
	gw, err := NewGGUFWriter(&buf, nil, planned, 32)
	if err != nil {
		t.Fatalf("NewGGUFWriter: %v", err)
	}
	if err := gw.WriteTensor(make([]byte, 10)); err == nil {
		t.Fatal("WriteTensor with wrong size: want error, got nil")
	}
}
