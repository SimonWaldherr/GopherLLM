package gopherllm

// GGUF container parsing.
//
// A GGUF file is, in order: the 4-byte magic "GGUF", a u32 version, a u64
// tensor count, a u64 metadata count, the metadata key/value entries, the
// tensor descriptors (name, dims, dtype, data offset), and finally — aligned
// to general.alignment (default 32) — the raw tensor data. All integers are
// little-endian. Tensor offsets are relative to the aligned data start
// (GGUFFile.DataOffset), not the file start.
//
// Parsing here reads only the header; tensor data is left in place so callers
// can borrow it directly from a memory-mapped file (see loadWeight's borrow
// mode) instead of copying multi-gigabyte weights.

import (
	"encoding/binary"
	"fmt"
	"math"
)

// GGMLType identifies a tensor's on-disk element encoding, matching the
// ggml type ids used by llama.cpp. F32/F16 are plain element arrays; the
// quantized types pack fixed-size blocks of elements together with their
// scale factors — see BlockBytes/DataSize for the exact per-block sizes the
// matvec kernels in simd.go rely on.
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
	GGMLTypeIQ2_XXS GGMLType = 16
	GGMLTypeIQ2_XS  GGMLType = 17
	GGMLTypeIQ3_XXS GGMLType = 18
	GGMLTypeIQ1_S   GGMLType = 19
	GGMLTypeIQ4_NL  GGMLType = 20
	GGMLTypeIQ3_S   GGMLType = 21
	GGMLTypeIQ2_S   GGMLType = 22
	GGMLTypeIQ4_XS  GGMLType = 23
	GGMLTypeF64     GGMLType = 28
	GGMLTypeIQ1_M   GGMLType = 29
	GGMLTypeBF16    GGMLType = 30
	GGMLTypeTQ1_0   GGMLType = 34
	GGMLTypeTQ2_0   GGMLType = 35
	GGMLTypeMXFP4   GGMLType = 39
	GGMLTypeQ1_0    GGMLType = 41
	GGMLTypeQ2_0    GGMLType = 42
	GGMLTypeUnknown GGMLType = 255
)

// ggmlTypeFromUint32 maps a raw GGUF dtype id to a GGMLType, folding every id
// this runtime cannot execute into GGMLTypeUnknown so the mismatch surfaces as
// a load-time error instead of a bogus kernel dispatch.
func ggmlTypeFromUint32(v uint32) GGMLType {
	switch GGMLType(v) {
	case GGMLTypeF32, GGMLTypeF16, GGMLTypeQ4_0, GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1,
		GGMLTypeQ8_0, GGMLTypeQ8_1, GGMLTypeQ2_K, GGMLTypeQ3_K, GGMLTypeQ4_K, GGMLTypeQ5_K,
		GGMLTypeQ6_K, GGMLTypeQ8_K, GGMLTypeIQ4_NL, GGMLTypeIQ3_S, GGMLTypeIQ2_S, GGMLTypeIQ4_XS,
		GGMLTypeIQ2_XXS, GGMLTypeIQ2_XS, GGMLTypeIQ3_XXS, GGMLTypeIQ1_S, GGMLTypeIQ1_M,
		GGMLTypeF64, GGMLTypeBF16, GGMLTypeTQ1_0, GGMLTypeTQ2_0, GGMLTypeMXFP4, GGMLTypeQ1_0, GGMLTypeQ2_0:
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
	case GGMLTypeQ2_K:
		return "Q2_K"
	case GGMLTypeQ3_K:
		return "Q3_K"
	case GGMLTypeQ4_K:
		return "Q4_K"
	case GGMLTypeQ5_K:
		return "Q5_K"
	case GGMLTypeQ6_K:
		return "Q6_K"
	case GGMLTypeQ8_K:
		return "Q8_K"
	case GGMLTypeIQ4_NL:
		return "IQ4_NL"
	case GGMLTypeIQ3_S:
		return "IQ3_S"
	case GGMLTypeIQ2_S:
		return "IQ2_S"
	case GGMLTypeIQ4_XS:
		return "IQ4_XS"
	case GGMLTypeIQ2_XXS:
		return "IQ2_XXS"
	case GGMLTypeIQ2_XS:
		return "IQ2_XS"
	case GGMLTypeIQ3_XXS:
		return "IQ3_XXS"
	case GGMLTypeIQ1_S:
		return "IQ1_S"
	case GGMLTypeIQ1_M:
		return "IQ1_M"
	case GGMLTypeF64:
		return "F64"
	case GGMLTypeBF16:
		return "BF16"
	case GGMLTypeTQ1_0:
		return "TQ1_0"
	case GGMLTypeTQ2_0:
		return "TQ2_0"
	case GGMLTypeMXFP4:
		return "MXFP4"
	case GGMLTypeQ1_0:
		return "Q1_0"
	case GGMLTypeQ2_0:
		return "Q2_0"
	default:
		return "Unknown"
	}
}

// BlockSize returns how many elements one quantization block encodes (1 for
// the plain float types). Most legacy simple quants are 32-wide; the K-quants
// and IQ4_XS use 256-element superblocks. DataSize remains the authoritative
// byte-size accessor for every format.
func (t GGMLType) BlockSize() int {
	if t == GGMLTypeF32 || t == GGMLTypeF16 || t == GGMLTypeF64 || t == GGMLTypeBF16 {
		return 1
	}
	switch t {
	case GGMLTypeQ1_0:
		return 128
	case GGMLTypeQ2_0:
		return 64
	case GGMLTypeTQ1_0, GGMLTypeTQ2_0,
		GGMLTypeQ2_K, GGMLTypeQ3_K, GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_K,
		GGMLTypeIQ2_S, GGMLTypeIQ3_S, GGMLTypeIQ4_XS,
		GGMLTypeIQ2_XXS, GGMLTypeIQ2_XS, GGMLTypeIQ3_XXS, GGMLTypeIQ1_S, GGMLTypeIQ1_M:
		return 256
	default:
		return 32
	}
}

// BlockBytes returns the byte size of one block for the simple (32-element)
// quant formats, e.g. Q8_0 = 2-byte f16 scale + 32 int8 quants = 34 bytes,
// Q4_0 = 2-byte f16 scale + 16 packed-nibble bytes = 18 bytes. K-quants and
// MXFP4 are not representable here (ok=false); use DataSize instead.
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
	case GGMLTypeIQ4_NL:
		return 18, true
	case GGMLTypeQ8_1:
		return 36, true
	default:
		return 0, false
	}
}

// DataSize returns the exact byte size of n elements of this type. The
// per-block layouts, which the dot kernels in simd.go index by hand:
//
//	Q4_0:  18 B / 32 elems  = f16 scale + 16 nibble-packed bytes
//	Q4_1:  20 B / 32 elems  = f16 scale + f16 min + 16 nibble bytes
//	Q5_0:  22 B / 32 elems  = f16 scale + 4 B 5th-bit plane + 16 nibbles
//	Q5_1:  24 B / 32 elems  = f16 scale + f16 min + 4 B 5th bits + 16 nibbles
//	Q8_0:  34 B / 32 elems  = f16 scale + 32 int8
//	Q8_1:  36 B / 32 elems  = f16 scale + f16 quant-sum + 32 int8
//	Q2_K:  84 B / 256 elems = 16 B packed 4-bit scale/min pairs +
//	       64 B 2-bit quants + f16 d + f16 dmin
//	Q3_K: 110 B / 256 elems = 32 B high-bit plane + 64 B 2-bit lows +
//	       12 B packed 6-bit scales + f16 d
//	Q4_K: 144 B / 256 elems = f16 d + f16 dmin + 12 B packed 6-bit
//	       scales/mins (8 sub-blocks) + 128 nibble-packed bytes
//	Q5_K: 176 B / 256 elems = Q4_K layout + 32 B of 5th-bit planes
//	Q6_K: 210 B / 256 elems = 128 B low nibbles + 64 B 2-bit highs +
//	       16 int8 sub-block scales + f16 d
//	IQ4_XS: 136 B / 256 elems = f16 d + 16-bit high scales + 4-byte low
//	       scales + 128 non-linear nibble-packed values
//	IQ2_S:  82 B / 256 elems = f16 d + 64 byte codebook indices/sign bits +
//	       8 high-index bits + 8 packed scales
//	IQ3_S: 110 B / 256 elems = f16 d + 64 byte codebook indices + 8 high
//	       index bits + 32 sign bits + 4 packed scales
//	IQ2_XXS: 66 B / 256 elems = f16 d + 32 uint16 codebook/sign indices
//	IQ2_XS:  74 B / 256 elems = f16 d + 32 uint16 codebook indices +
//	       8 packed scales (sign bits come from a shared lookup table)
//	IQ3_XXS: 98 B / 256 elems = f16 d + 96 byte codebook indices (the
//	       trailing 4 bytes per 32-elem group double as packed scale+signs)
//	IQ1_S:   50 B / 256 elems = f16 d + 32 byte low grid-index bits +
//	       8 uint16 high-index/scale/sign words
//	IQ1_M:   56 B / 256 elems = 32 byte low grid-index bits + 16 byte
//	       high-index/sign bits + 8 byte packed 3-bit scales (the block
//	       scale is reassembled from spare high bits of the scales word)
//	TQ1_0:  54 B / 256 elems = 48 base-3 bytes + 4 base-3 high bytes + f16 d
//	TQ2_0:  66 B / 256 elems = 64 packed 2-bit ternary values + f16 d
//	MXFP4: 17 B / 32 elems  = 16 nibble-packed FP4 values + 1 shared
//	       power-of-two exponent byte
//	Q1_0:  18 B / 128 elems = f16 scale + 128 packed sign bits
//	Q2_0:  18 B / 64 elems  = f16 scale + 64 packed 2-bit values
//
// ok is false for types this runtime cannot size (callers then fall back to
// offset-difference inference; see inferTensorSizes).
func (t GGMLType) DataSize(n int) (int, bool) {
	switch t {
	case GGMLTypeF32:
		return n * 4, true
	case GGMLTypeF64:
		return n * 8, true
	case GGMLTypeF16, GGMLTypeBF16:
		return n * 2, true
	case GGMLTypeQ4_0, GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1, GGMLTypeQ8_0, GGMLTypeQ8_1, GGMLTypeIQ4_NL:
		b, _ := t.BlockBytes()
		return (n / t.BlockSize()) * b, true
	case GGMLTypeQ2_K:
		return (n / 256) * 84, true
	case GGMLTypeQ3_K:
		return (n / 256) * 110, true
	case GGMLTypeQ4_K:
		return (n / 256) * 144, true
	case GGMLTypeQ5_K:
		return (n / 256) * 176, true
	case GGMLTypeQ6_K:
		return (n / 256) * 210, true
	case GGMLTypeQ8_K:
		// float32 scale + 256 int8 values + 16 int16 block sums.
		// The sums accelerate mixed-quant dot products; dequantization only
		// needs the scale and values.
		return (n / 256) * 292, true
	case GGMLTypeIQ4_XS:
		return (n / 256) * 136, true
	case GGMLTypeIQ2_S:
		return (n / 256) * 82, true
	case GGMLTypeIQ3_S:
		return (n / 256) * 110, true
	case GGMLTypeIQ2_XXS:
		return (n / 256) * 66, true
	case GGMLTypeIQ2_XS:
		return (n / 256) * 74, true
	case GGMLTypeIQ3_XXS:
		return (n / 256) * 98, true
	case GGMLTypeIQ1_S:
		return (n / 256) * 50, true
	case GGMLTypeIQ1_M:
		return (n / 256) * 56, true
	case GGMLTypeTQ1_0:
		return (n / 256) * 54, true
	case GGMLTypeTQ2_0:
		return (n / 256) * 66, true
	case GGMLTypeMXFP4:
		return (n / 32) * 17, true
	case GGMLTypeQ1_0:
		return (n / 128) * 18, true
	case GGMLTypeQ2_0:
		return (n / 64) * 18, true
	default:
		return 0, false
	}
}

// MetaValue is one decoded GGUF metadata value. Kind is the GGUF wire-type
// name ("u32", "str", "array", ...) and Value holds the corresponding Go value
// (arrays are []MetaValue). The As* accessors below do tolerant conversions —
// GGUF producers are inconsistent about integer widths, so e.g. AsU32 accepts
// any integer kind rather than only u32.
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
	if s, ok := v.Value.([]string); ok {
		return s, true
	}
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

// AsBoolArray decodes an array of bools (e.g. Gemma 4's per-layer
// attention.sliding_window_pattern).
func (v MetaValue) AsBoolArray() ([]bool, bool) {
	if b, ok := v.Value.([]bool); ok {
		return b, true
	}
	arr, ok := v.Value.([]MetaValue)
	if !ok {
		return nil, false
	}
	out := make([]bool, 0, len(arr))
	for _, item := range arr {
		b, ok := item.AsBool()
		if !ok {
			return nil, false
		}
		out = append(out, b)
	}
	return out, true
}

func (v MetaValue) AsF32Array() ([]float32, bool) {
	if f, ok := v.Value.([]float32); ok {
		return f, true
	}
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

// AsU32Array decodes an integer metadata array. GGUF writers use different
// integer widths for per-layer architecture fields, so accept every value
// that AsU32 accepts.
func (v MetaValue) AsU32Array() ([]uint32, bool) {
	if a, ok := v.Value.([]uint32); ok {
		return a, true
	}
	arr, ok := v.Value.([]MetaValue)
	if !ok {
		return nil, false
	}
	out := make([]uint32, len(arr))
	for i, item := range arr {
		n, ok := item.AsU32()
		if !ok {
			return nil, false
		}
		out[i] = n
	}
	return out, true
}

// TensorInfo describes one tensor from the GGUF header. Dims follow GGUF's
// convention of fastest-varying dimension first, so for a 2-D weight
// Dims[0] is the input (column) count and Dims[1] the output (row) count.
// Offset is relative to GGUFFile.DataOffset.
type TensorInfo struct {
	Name   string
	Dims   []uint64
	DType  GGMLType
	Offset uint64
}

// Numel returns the total element count (product of Dims).
func (t TensorInfo) Numel() int {
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	return n
}

// GGUFFile is a parsed GGUF header: all metadata, all tensor descriptors, and
// the alignment-adjusted offset where tensor data begins within the original
// byte slice. It does not own or copy any tensor data.
type GGUFFile struct {
	Metadata   map[string]MetaValue
	Tensors    []TensorInfo
	DataOffset int
	Version    uint32
}

// GetU32Array returns an integer metadata array, if present.
func (g *GGUFFile) GetU32Array(key string) ([]uint32, bool) {
	v, ok := g.Metadata[key]
	if !ok {
		return nil, false
	}
	return v.AsU32Array()
}

// GetF32Array reads a float32 array metadata value, e.g. a vision encoder's
// clip.vision.image_mean/image_std.
func (g *GGUFFile) GetF32Array(key string) ([]float32, bool) {
	v, ok := g.Metadata[key]
	if !ok {
		return nil, false
	}
	return v.AsF32Array()
}

// ParseGGUF parses a GGUF header, logging a one-line summary to stderr.
// ParseGGUFQuiet is the same without the log line (used by model discovery,
// which parses every file in a directory).
func ParseGGUF(data []byte) (*GGUFFile, error)      { return parseGGUF(data) }
func ParseGGUFQuiet(data []byte) (*GGUFFile, error) { return parseGGUF(data) }

func parseGGUF(data []byte) (*GGUFFile, error) {
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

// GetU32/GetF32/GetString are convenience metadata lookups; the numeric
// variants return def when the key is absent or has an incompatible kind.
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

// GetBool returns def when the key is absent or has an incompatible kind.
func (g *GGUFFile) GetBool(key string, def bool) bool {
	if v, ok := g.Metadata[key]; ok {
		if b, ok := v.AsBool(); ok {
			return b
		}
	}
	return def
}

// cursor is a bounds-checked little-endian reader over the raw file bytes;
// every read verifies remaining length so truncated files fail with a clear
// "unexpected EOF at byte N" instead of a panic.
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
		// Typed fast paths for the huge tokenizer arrays (100k+ vocab
		// strings, scores, per-layer bool maps): storing a typed slice
		// avoids boxing every element into a MetaValue interface, which
		// dominated model-load time. The As*Array accessors understand
		// both representations.
		switch elem {
		case 8: // str
			out := make([]string, 0, int(count))
			for range int(count) {
				s, err := c.str()
				if err != nil {
					return MetaValue{}, err
				}
				out = append(out, s)
			}
			return MetaValue{"array", out}, nil
		case 6: // f32
			out := make([]float32, 0, int(count))
			for range int(count) {
				v, err := c.u32()
				if err != nil {
					return MetaValue{}, err
				}
				out = append(out, math.Float32frombits(v))
			}
			return MetaValue{"array", out}, nil
		case 7: // bool
			out := make([]bool, 0, int(count))
			for range int(count) {
				v, err := c.u8()
				if err != nil {
					return MetaValue{}, err
				}
				out = append(out, v != 0)
			}
			return MetaValue{"array", out}, nil
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
