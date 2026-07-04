package main

import (
	"math"
	"testing"
)

func TestParseGGUFRoundTrip(t *testing.T) {
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "llama"},
		{"v.u8", ggufU8, uint8(7)},
		{"v.i8", ggufI8, int8(-3)},
		{"v.u16", ggufU16, uint16(300)},
		{"v.i16", ggufI16, int16(-300)},
		{"v.u32", ggufU32, uint32(70000)},
		{"v.i32", ggufI32, int32(-5)},
		{"v.f32", ggufF32, float32(3.5)},
		{"v.bool", ggufBool, true},
		{"v.u64", ggufU64, uint64(1) << 40},
		{"v.i64", ggufI64, int64(-7)},
		{"v.f64", ggufF64, float64(2.5)},
		{"toks", ggufArr, ggufArray{ggufStr, []any{"a", "b", "c"}}},
		{"scores", ggufArr, ggufArray{ggufF32, []any{float32(1), float32(2)}}},
		{"general.alignment", ggufU32, uint32(32)},
	}
	tensors := []ggufTensor{
		{"t0", []uint64{4, 2}, GGMLTypeF32, f32Bytes([]float32{1, 2, 3, 4, 5, 6, 7, 8})},
		{"t1", []uint64{8}, GGMLTypeF32, f32Bytes([]float32{8, 7, 6, 5, 4, 3, 2, 1})},
	}
	data := buildGGUF(3, kvs, tensors)

	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if g.Version != 3 {
		t.Fatalf("version = %d, want 3", g.Version)
	}
	if s, _ := g.GetString("general.architecture"); s != "llama" {
		t.Fatalf("architecture = %q", s)
	}
	// AsU32 must widen every integer kind.
	for _, k := range []string{"v.u8", "v.i8", "v.u16", "v.u32", "v.u64"} {
		if _, ok := g.Metadata[k].AsU32(); !ok {
			t.Fatalf("%s AsU32 not ok", k)
		}
	}
	if g.GetU32("v.u32", 0) != 70000 {
		t.Fatalf("v.u32 = %d", g.GetU32("v.u32", 0))
	}
	if g.GetF32("v.f32", 0) != 3.5 {
		t.Fatalf("v.f32 = %v", g.GetF32("v.f32", 0))
	}
	if f, ok := g.Metadata["v.f64"].AsF32(); !ok || math.Abs(float64(f-2.5)) > 1e-6 {
		t.Fatalf("v.f64 AsF32 = %v ok=%v", f, ok)
	}
	if b, ok := g.Metadata["v.bool"].AsBool(); !ok || !b {
		t.Fatalf("v.bool = %v ok=%v", b, ok)
	}
	if arr, ok := g.Metadata["toks"].AsStringArray(); !ok || len(arr) != 3 || arr[2] != "c" {
		t.Fatalf("toks = %v ok=%v", arr, ok)
	}
	if arr, ok := g.Metadata["scores"].AsF32Array(); !ok || len(arr) != 2 || arr[1] != 2 {
		t.Fatalf("scores = %v ok=%v", arr, ok)
	}
	if len(g.Tensors) != 2 {
		t.Fatalf("tensors = %d, want 2", len(g.Tensors))
	}
	if g.Tensors[0].Numel() != 8 || g.Tensors[0].DType != GGMLTypeF32 {
		t.Fatalf("t0 numel=%d dtype=%v", g.Tensors[0].Numel(), g.Tensors[0].DType)
	}
	if g.DataOffset%32 != 0 {
		t.Fatalf("DataOffset %d not 32-aligned", g.DataOffset)
	}
	// The first tensor's data must be readable at DataOffset.
	got := math.Float32frombits(uint32(data[g.DataOffset]) | uint32(data[g.DataOffset+1])<<8 | uint32(data[g.DataOffset+2])<<16 | uint32(data[g.DataOffset+3])<<24)
	if got != 1 {
		t.Fatalf("first tensor float = %v, want 1", got)
	}
}

func TestParseGGUFRejectsBadInput(t *testing.T) {
	if _, err := ParseGGUFQuiet([]byte{1, 2}); err == nil {
		t.Fatal("tiny buffer should fail")
	}
	if _, err := ParseGGUFQuiet([]byte("XXXXwhatever")); err == nil {
		t.Fatal("bad magic should fail")
	}
	// Header claims one KV entry, but the buffer ends right after the header
	// so reading the key hits EOF.
	valid := buildGGUF(3, []ggufKV{{"k", ggufU32, uint32(1)}}, nil)
	if _, err := ParseGGUFQuiet(valid[:24]); err == nil {
		t.Fatal("truncated metadata should fail")
	}
	// Unknown metadata value type must be rejected.
	bad := buildGGUF(3, []ggufKV{{"k", ggufU32, uint32(1)}}, nil)
	bad[24+8+1] = 99 // overwrite the value-type tag of key "k" with an invalid type
	if _, err := ParseGGUFQuiet(bad); err == nil {
		t.Fatal("unknown value type should fail")
	}
}

func TestGGMLTypeDataSize(t *testing.T) {
	cases := []struct {
		typ  GGMLType
		n    int
		want int
	}{
		{GGMLTypeF32, 10, 40},
		{GGMLTypeF16, 10, 20},
		{GGMLTypeQ8_0, 32, 34},
		{GGMLTypeQ4_K, 256, 144},
		{GGMLTypeQ6_K, 256, 210},
		{GGMLTypeMXFP4, 32, 17},
	}
	for _, c := range cases {
		got, ok := c.typ.DataSize(c.n)
		if !ok || got != c.want {
			t.Fatalf("%s DataSize(%d) = %d,%v want %d", c.typ, c.n, got, ok, c.want)
		}
	}
	if _, ok := GGMLTypeUnknown.DataSize(10); ok {
		t.Fatal("unknown type DataSize should not be ok")
	}
}

func TestGGMLTypeBlockBytes(t *testing.T) {
	tests := []struct {
		typ  GGMLType
		want int
	}{
		{GGMLTypeF32, 4},
		{GGMLTypeF16, 2},
		{GGMLTypeQ4_0, 18},
		{GGMLTypeQ4_1, 20},
		{GGMLTypeQ5_0, 22},
		{GGMLTypeQ5_1, 24},
		{GGMLTypeQ8_0, 34},
		{GGMLTypeQ8_1, 36},
	}
	for _, tt := range tests {
		got, ok := tt.typ.BlockBytes()
		if !ok || got != tt.want {
			t.Fatalf("%s BlockBytes = %d, %v; want %d, true", tt.typ, got, ok, tt.want)
		}
	}
}
