package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

type GGMLType uint32

const (
	GGMLTypeF32     GGMLType = 0
	GGMLTypeF16     GGMLType = 1
	GGMLTypeQ4_0    GGMLType = 2
	GGMLTypeQ4_1    GGMLType = 3
	GGMLTypeQ5_0    GGMLType = 6
	GGMLTypeQ5_1    GGMLType = 7
	GGMLTypeQ8_0    GGMLType = 8
	GGMLTypeQ8_1    GGMLType = 9
	GGMLTypeQ2_K    GGMLType = 10
	GGMLTypeQ3_K    GGMLType = 11
	GGMLTypeQ4_K    GGMLType = 12
	GGMLTypeQ5_K    GGMLType = 13
	GGMLTypeQ6_K    GGMLType = 14
	GGMLTypeQ8_K    GGMLType = 15
	GGMLTypeMXFP4   GGMLType = 39
	GGMLTypeUnknown GGMLType = 255
)

func ggmlTypeFromUint32(v uint32) GGMLType {
	switch GGMLType(v) {
	case GGMLTypeF32, GGMLTypeF16, GGMLTypeQ4_0, GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1,
		GGMLTypeQ8_0, GGMLTypeQ8_1, GGMLTypeQ2_K, GGMLTypeQ3_K, GGMLTypeQ4_K, GGMLTypeQ5_K,
		GGMLTypeQ6_K, GGMLTypeQ8_K, GGMLTypeMXFP4:
		return GGMLType(v)
	default:
		return GGMLTypeUnknown
	}
}

func (t GGMLType) String() string {
	switch t {
	case GGMLTypeF32:
		return "F32"
	case GGMLTypeF16:
		return "F16"
	case GGMLTypeQ4_0:
		return "Q4_0"
	case GGMLTypeQ4_1:
		return "Q4_1"
	case GGMLTypeQ5_0:
		return "Q5_0"
	case GGMLTypeQ5_1:
		return "Q5_1"
	case GGMLTypeQ8_0:
		return "Q8_0"
	case GGMLTypeQ8_1:
		return "Q8_1"
	case GGMLTypeQ4_K:
		return "Q4_K"
	case GGMLTypeQ5_K:
		return "Q5_K"
	case GGMLTypeQ6_K:
		return "Q6_K"
	case GGMLTypeMXFP4:
		return "MXFP4"
	default:
		return "Unknown"
	}
}

func (t GGMLType) BlockSize() int {
	if t == GGMLTypeF32 || t == GGMLTypeF16 {
		return 1
	}
	return 32
}

func (t GGMLType) BlockBytes() (int, bool) {
	switch t {
	case GGMLTypeF32:
		return 4, true
	case GGMLTypeF16:
		return 2, true
	case GGMLTypeQ4_0:
		return 18, true
	case GGMLTypeQ4_1:
		return 20, true
	case GGMLTypeQ5_0:
		return 22, true
	case GGMLTypeQ5_1:
		return 24, true
	case GGMLTypeQ8_0:
		return 34, true
	case GGMLTypeQ8_1:
		return 36, true
	default:
		return 0, false
	}
}

func (t GGMLType) DataSize(n int) (int, bool) {
	switch t {
	case GGMLTypeF32:
		return n * 4, true
	case GGMLTypeF16:
		return n * 2, true
	case GGMLTypeQ4_0, GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1, GGMLTypeQ8_0, GGMLTypeQ8_1:
		b, _ := t.BlockBytes()
		return (n / t.BlockSize()) * b, true
	case GGMLTypeQ4_K:
		return (n / 256) * 144, true
	case GGMLTypeQ5_K:
		return (n / 256) * 176, true
	case GGMLTypeQ6_K:
		return (n / 256) * 210, true
	case GGMLTypeMXFP4:
		return (n / 32) * 17, true
	default:
		return 0, false
	}
}

type MetaValue struct {
	Kind  string
	Value any
}

func (v MetaValue) AsU32() (uint32, bool) {
	switch x := v.Value.(type) {
	case uint8:
		return uint32(x), true
	case int8:
		return uint32(x), true
	case uint16:
		return uint32(x), true
	case int16:
		return uint32(x), true
	case uint32:
		return x, true
	case int32:
		return uint32(x), true
	case uint64:
		return uint32(x), true
	case int64:
		return uint32(x), true
	default:
		return 0, false
	}
}

func (v MetaValue) AsF32() (float32, bool) {
	switch x := v.Value.(type) {
	case float32:
		return x, true
	case float64:
		return float32(x), true
	default:
		return 0, false
	}
}

func (v MetaValue) AsString() (string, bool) {
	x, ok := v.Value.(string)
	return x, ok
}

func (v MetaValue) AsBool() (bool, bool) {
	x, ok := v.Value.(bool)
	return x, ok
}

func (v MetaValue) AsStringArray() ([]string, bool) {
	arr, ok := v.Value.([]MetaValue)
	if !ok {
		return nil, false
	}
	out := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.AsString(); ok {
			out = append(out, s)
		}
	}
	return out, true
}

func (v MetaValue) AsF32Array() ([]float32, bool) {
	arr, ok := v.Value.([]MetaValue)
	if !ok {
		return nil, false
	}
	out := make([]float32, 0, len(arr))
	for _, item := range arr {
		if f, ok := item.AsF32(); ok {
			out = append(out, f)
		}
	}
	return out, true
}

type TensorInfo struct {
	Name   string
	Dims   []uint64
	DType  GGMLType
	Offset uint64
}

func (t TensorInfo) Numel() int {
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	return n
}

type GGUFFile struct {
	Metadata   map[string]MetaValue
	Tensors    []TensorInfo
	DataOffset int
	Version    uint32
}

func ParseGGUF(data []byte) (*GGUFFile, error)      { return parseGGUF(data, true) }
func ParseGGUFQuiet(data []byte) (*GGUFFile, error) { return parseGGUF(data, false) }

func parseGGUF(data []byte, verbose bool) (*GGUFFile, error) {
	c := cursor{data: data}
	if len(data) < 4 {
		return nil, fmt.Errorf("file too small for GGUF header")
	}
	if string(data[:4]) != "GGUF" {
		return nil, fmt.Errorf("invalid GGUF magic: 0x%08X", binary.LittleEndian.Uint32(data[:4]))
	}
	c.pos = 4
	version, err := c.u32()
	if err != nil {
		return nil, err
	}
	nTensors, err := c.u64()
	if err != nil {
		return nil, err
	}
	nKV, err := c.u64()
	if err != nil {
		return nil, err
	}
	if verbose {
		fmt.Fprintf(stderr(), "GGUF v%d - %d tensors, %d metadata entries\n", version, nTensors, nKV)
	}
	metadata := make(map[string]MetaValue, int(nKV))
	for range int(nKV) {
		key, err := c.str()
		if err != nil {
			return nil, err
		}
		typ, err := c.u32()
		if err != nil {
			return nil, err
		}
		val, err := c.value(typ)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", key, err)
		}
		metadata[key] = val
	}
	tensors := make([]TensorInfo, 0, int(nTensors))
	for range int(nTensors) {
		name, err := c.str()
		if err != nil {
			return nil, err
		}
		nDims, err := c.u32()
		if err != nil {
			return nil, err
		}
		dims := make([]uint64, int(nDims))
		for i := range dims {
			dims[i], err = c.u64()
			if err != nil {
				return nil, err
			}
		}
		dt, err := c.u32()
		if err != nil {
			return nil, err
		}
		off, err := c.u64()
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, TensorInfo{Name: name, Dims: dims, DType: ggmlTypeFromUint32(dt), Offset: off})
	}
	alignment := 32
	if v, ok := metadata["general.alignment"]; ok {
		if n, ok := v.AsU32(); ok && n > 0 {
			alignment = int(n)
		}
	}
	return &GGUFFile{Metadata: metadata, Tensors: tensors, DataOffset: divCeil(c.pos, alignment) * alignment, Version: version}, nil
}

func (g *GGUFFile) GetU32(key string, def uint32) uint32 {
	if v, ok := g.Metadata[key]; ok {
		if n, ok := v.AsU32(); ok {
			return n
		}
	}
	return def
}

func (g *GGUFFile) GetF32(key string, def float32) float32 {
	if v, ok := g.Metadata[key]; ok {
		if n, ok := v.AsF32(); ok {
			return n
		}
	}
	return def
}

func (g *GGUFFile) GetString(key string) (string, bool) {
	if v, ok := g.Metadata[key]; ok {
		return v.AsString()
	}
	return "", false
}

type cursor struct {
	data []byte
	pos  int
}

func (c *cursor) need(n int) error {
	if n < 0 || c.pos+n > len(c.data) {
		return fmt.Errorf("unexpected EOF at byte %d", c.pos)
	}
	return nil
}

func (c *cursor) u8() (uint8, error) {
	if err := c.need(1); err != nil {
		return 0, err
	}
	v := c.data[c.pos]
	c.pos++
	return v, nil
}

func (c *cursor) u16() (uint16, error) {
	if err := c.need(2); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint16(c.data[c.pos:])
	c.pos += 2
	return v, nil
}

func (c *cursor) u32() (uint32, error) {
	if err := c.need(4); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint32(c.data[c.pos:])
	c.pos += 4
	return v, nil
}

func (c *cursor) u64() (uint64, error) {
	if err := c.need(8); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(c.data[c.pos:])
	c.pos += 8
	return v, nil
}

func (c *cursor) str() (string, error) {
	n64, err := c.u64()
	if err != nil {
		return "", err
	}
	n := int(n64)
	if err := c.need(n); err != nil {
		return "", err
	}
	s := string(c.data[c.pos : c.pos+n])
	c.pos += n
	return s, nil
}

func (c *cursor) value(t uint32) (MetaValue, error) {
	switch t {
	case 0:
		v, err := c.u8()
		return MetaValue{"u8", v}, err
	case 1:
		v, err := c.u8()
		return MetaValue{"i8", int8(v)}, err
	case 2:
		v, err := c.u16()
		return MetaValue{"u16", v}, err
	case 3:
		v, err := c.u16()
		return MetaValue{"i16", int16(v)}, err
	case 4:
		v, err := c.u32()
		return MetaValue{"u32", v}, err
	case 5:
		v, err := c.u32()
		return MetaValue{"i32", int32(v)}, err
	case 6:
		v, err := c.u32()
		return MetaValue{"f32", math.Float32frombits(v)}, err
	case 7:
		v, err := c.u8()
		return MetaValue{"bool", v != 0}, err
	case 8:
		v, err := c.str()
		return MetaValue{"str", v}, err
	case 9:
		elem, err := c.u32()
		if err != nil {
			return MetaValue{}, err
		}
		count, err := c.u64()
		if err != nil {
			return MetaValue{}, err
		}
		arr := make([]MetaValue, 0, int(count))
		for range int(count) {
			v, err := c.value(elem)
			if err != nil {
				return MetaValue{}, err
			}
			arr = append(arr, v)
		}
		return MetaValue{"array", arr}, nil
	case 10:
		v, err := c.u64()
		return MetaValue{"u64", v}, err
	case 11:
		v, err := c.u64()
		return MetaValue{"i64", int64(v)}, err
	case 12:
		v, err := c.u64()
		return MetaValue{"f64", math.Float64frombits(v)}, err
	default:
		return MetaValue{}, fmt.Errorf("unknown GGUF value type %d", t)
	}
}

func divCeil(a, b int) int {
	if b <= 0 {
		return a
	}
	return (a + b - 1) / b
}
