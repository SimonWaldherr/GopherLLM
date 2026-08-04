package gopherllm

// GGUF container writing — the inverse of gguf.go's ParseGGUF. A GGUF file's
// tensor offsets are cumulative and computed from every earlier tensor's
// on-disk size (see parseGGUF's DataOffset calc), so the full tensor plan —
// every tensor's name, dims, and OUTPUT type — must be fixed before the
// first byte is written. That splits writing into two phases: NewGGUFWriter
// builds the header, metadata, and tensor descriptor table entirely from a
// plan (no tensor bytes needed yet), then WriteTensor streams each tensor's
// bytes afterward, in the same order, so multi-gigabyte weights are never
// buffered in memory.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// kindToWireType maps a MetaValue.Kind string to the GGUF wire-format value
// type tag cursor.value (gguf.go) reads. Kept as the single source of truth
// for both directions so the tag list can't drift between reader and writer.
var kindToWireType = map[string]uint32{
	"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5,
	"f32": 6, "bool": 7, "str": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12,
}

// countingWriter tracks the absolute byte position written so far, which
// the tensor descriptor table's alignment padding and WriteTensor's
// inter-tensor padding both need — mirroring parseGGUF's own cursor.pos.
type countingWriter struct {
	bw  *bufio.Writer
	pos int64
	err error
}

func newCountingWriter(w io.Writer) *countingWriter {
	return &countingWriter{bw: bufio.NewWriterSize(w, 1<<20)}
}

func (cw *countingWriter) Write(p []byte) (int, error) {
	if cw.err != nil {
		return 0, cw.err
	}
	n, err := cw.bw.Write(p)
	cw.pos += int64(n)
	if err != nil {
		cw.err = err
	}
	return n, err
}

func (cw *countingWriter) u8(v uint8) error {
	_, err := cw.Write([]byte{v})
	return err
}

func (cw *countingWriter) u16(v uint16) error {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], v)
	_, err := cw.Write(b[:])
	return err
}

func (cw *countingWriter) u32(v uint32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	_, err := cw.Write(b[:])
	return err
}

func (cw *countingWriter) u64(v uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	_, err := cw.Write(b[:])
	return err
}

// str writes a GGUF string: a u64 byte length followed by the raw bytes
// (no NUL terminator), matching cursor.str's read side exactly.
func (cw *countingWriter) str(s string) error {
	if err := cw.u64(uint64(len(s))); err != nil {
		return err
	}
	_, err := cw.Write([]byte(s))
	return err
}

func alignUp64(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) / a * a
}

// writeMetaValue writes a metadata value's wire-type tag followed by its
// payload — the reverse of cursor.value(t uint32) (gguf.go:602-698).
func writeMetaValue(cw *countingWriter, v MetaValue) error {
	tag, ok := kindToWireType[v.Kind]
	if !ok {
		return fmt.Errorf("unknown metadata kind %q", v.Kind)
	}
	if err := cw.u32(tag); err != nil {
		return err
	}
	return writeMetaValuePayload(cw, v)
}

// writeMetaValuePayload writes only the value bytes, no type tag — reused
// for array elements, which carry one shared tag for the whole array rather
// than a tag per element.
func writeMetaValuePayload(cw *countingWriter, v MetaValue) error {
	switch v.Kind {
	case "u8":
		x, ok := v.Value.(uint8)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want uint8", v.Kind, v.Value)
		}
		return cw.u8(x)
	case "i8":
		x, ok := v.Value.(int8)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want int8", v.Kind, v.Value)
		}
		return cw.u8(uint8(x))
	case "u16":
		x, ok := v.Value.(uint16)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want uint16", v.Kind, v.Value)
		}
		return cw.u16(x)
	case "i16":
		x, ok := v.Value.(int16)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want int16", v.Kind, v.Value)
		}
		return cw.u16(uint16(x))
	case "u32":
		x, ok := v.Value.(uint32)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want uint32", v.Kind, v.Value)
		}
		return cw.u32(x)
	case "i32":
		x, ok := v.Value.(int32)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want int32", v.Kind, v.Value)
		}
		return cw.u32(uint32(x))
	case "f32":
		x, ok := v.Value.(float32)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want float32", v.Kind, v.Value)
		}
		return cw.u32(math.Float32bits(x))
	case "bool":
		x, ok := v.Value.(bool)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want bool", v.Kind, v.Value)
		}
		b := uint8(0)
		if x {
			b = 1
		}
		return cw.u8(b)
	case "str":
		x, ok := v.Value.(string)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want string", v.Kind, v.Value)
		}
		return cw.str(x)
	case "u64":
		x, ok := v.Value.(uint64)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want uint64", v.Kind, v.Value)
		}
		return cw.u64(x)
	case "i64":
		x, ok := v.Value.(int64)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want int64", v.Kind, v.Value)
		}
		return cw.u64(uint64(x))
	case "f64":
		x, ok := v.Value.(float64)
		if !ok {
			return fmt.Errorf("metadata kind %q has Go type %T, want float64", v.Kind, v.Value)
		}
		return cw.u64(math.Float64bits(x))
	case "array":
		return writeMetaArray(cw, v.Value)
	default:
		return fmt.Errorf("unknown metadata kind %q", v.Kind)
	}
}

// writeMetaArray writes an array's element-type tag, element count, then
// every element's payload — the reverse of cursor.value's case 9 (gguf.go).
// The typed fast paths ([]string/[]float32/[]bool) and the generic
// []MetaValue fallback mirror the exact two representations parsing
// produces (gguf.go:645-685).
func writeMetaArray(cw *countingWriter, value any) error {
	switch arr := value.(type) {
	case []string:
		if err := cw.u32(kindToWireType["str"]); err != nil {
			return err
		}
		if err := cw.u64(uint64(len(arr))); err != nil {
			return err
		}
		for _, s := range arr {
			if err := cw.str(s); err != nil {
				return err
			}
		}
		return nil
	case []float32:
		if err := cw.u32(kindToWireType["f32"]); err != nil {
			return err
		}
		if err := cw.u64(uint64(len(arr))); err != nil {
			return err
		}
		for _, f := range arr {
			if err := cw.u32(math.Float32bits(f)); err != nil {
				return err
			}
		}
		return nil
	case []bool:
		if err := cw.u32(kindToWireType["bool"]); err != nil {
			return err
		}
		if err := cw.u64(uint64(len(arr))); err != nil {
			return err
		}
		for _, b := range arr {
			v := uint8(0)
			if b {
				v = 1
			}
			if err := cw.u8(v); err != nil {
				return err
			}
		}
		return nil
	case []MetaValue:
		if len(arr) == 0 {
			// GGUF still requires a declared element type for a zero-length
			// array; u8 is an arbitrary but harmless choice since no element
			// is ever read back.
			if err := cw.u32(kindToWireType["u8"]); err != nil {
				return err
			}
			return cw.u64(0)
		}
		tag, ok := kindToWireType[arr[0].Kind]
		if !ok {
			return fmt.Errorf("unknown array element kind %q", arr[0].Kind)
		}
		if err := cw.u32(tag); err != nil {
			return err
		}
		if err := cw.u64(uint64(len(arr))); err != nil {
			return err
		}
		for i, item := range arr {
			if item.Kind != arr[0].Kind {
				return fmt.Errorf("array element %d has kind %q, want %q (GGUF arrays are homogeneous)", i, item.Kind, arr[0].Kind)
			}
			if err := writeMetaValuePayload(cw, item); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported metadata array Go type %T", value)
	}
}

// PlannedTensor is one output tensor's identity and size, decided before any
// tensor bytes exist. GGUF's descriptor table needs every tensor's final
// byte offset up front, and offsets are cumulative, so every tensor's
// output type must be fixed at plan time — no "pick the type based on
// measured error" once writing starts.
type PlannedTensor struct {
	Name  string
	Dims  []uint64
	DType GGMLType
}

// GGUFWriter streams a GGUF file: NewGGUFWriter emits the header, metadata,
// and full tensor descriptor table; WriteTensor then emits each tensor's
// bytes in turn, matching the order passed to NewGGUFWriter.
type GGUFWriter struct {
	cw         *countingWriter
	alignment  int
	tensors    []TensorInfo
	dataOffset int64
	next       int
}

// NewGGUFWriter writes a GGUF header, then metadata copied verbatim from
// metadata (callers wanting to change a key do so before calling this),
// then a tensor descriptor table computed from planned via
// GGMLType.DataSize. alignment<=0 defaults to 32, matching parseGGUF's
// default when general.alignment is absent.
func NewGGUFWriter(w io.Writer, metadata map[string]MetaValue, planned []PlannedTensor, alignment int) (*GGUFWriter, error) {
	if alignment <= 0 {
		alignment = 32
	}
	// general.alignment must match the padding actually used below, or a
	// reader computes a different DataOffset than where tensor data really
	// starts (parseGGUF derives its own alignment from this exact key,
	// defaulting to 32 when absent). Copy rather than mutate the caller's
	// map, and always set it explicitly so the file is self-consistent
	// regardless of what the caller passed (or forgot to pass).
	meta := make(map[string]MetaValue, len(metadata)+1)
	for k, v := range metadata {
		meta[k] = v
	}
	meta["general.alignment"] = MetaValue{"u32", uint32(alignment)}
	metadata = meta

	cw := newCountingWriter(w)

	if _, err := cw.Write([]byte("GGUF")); err != nil {
		return nil, err
	}
	if err := cw.u32(3); err != nil {
		return nil, err
	}
	if err := cw.u64(uint64(len(planned))); err != nil {
		return nil, err
	}
	if err := cw.u64(uint64(len(metadata))); err != nil {
		return nil, err
	}
	for key, val := range metadata {
		if err := cw.str(key); err != nil {
			return nil, err
		}
		if err := writeMetaValue(cw, val); err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, err)
		}
	}

	offset := uint64(0)
	tensors := make([]TensorInfo, len(planned))
	for i, p := range planned {
		numel := 1
		for _, d := range p.Dims {
			numel *= int(d)
		}
		size, ok := p.DType.DataSize(numel)
		if !ok {
			return nil, fmt.Errorf("tensor %q: unsupported output type %s", p.Name, p.DType)
		}
		tensors[i] = TensorInfo{Name: p.Name, Dims: p.Dims, DType: p.DType, Offset: offset}
		offset = alignUp64(offset+uint64(size), uint64(alignment))
	}
	for _, t := range tensors {
		if err := cw.str(t.Name); err != nil {
			return nil, err
		}
		if err := cw.u32(uint32(len(t.Dims))); err != nil {
			return nil, err
		}
		for _, d := range t.Dims {
			if err := cw.u64(d); err != nil {
				return nil, err
			}
		}
		if err := cw.u32(uint32(t.DType)); err != nil {
			return nil, err
		}
		if err := cw.u64(t.Offset); err != nil {
			return nil, err
		}
	}

	// Pad to the aligned data start, mirroring parseGGUF's
	// DataOffset = divCeil(pos, alignment) * alignment exactly.
	target := int64(alignUp64(uint64(cw.pos), uint64(alignment)))
	if pad := target - cw.pos; pad > 0 {
		if _, err := cw.Write(make([]byte, pad)); err != nil {
			return nil, err
		}
	}
	if cw.err != nil {
		return nil, cw.err
	}
	return &GGUFWriter{cw: cw, alignment: alignment, tensors: tensors, dataOffset: cw.pos}, nil
}

// WriteTensor writes the next planned tensor's bytes, in order. data must
// be exactly that tensor's DType.DataSize(numel) bytes; this call then pads
// up to the next tensor's aligned offset (or, for the last tensor, does
// nothing further — call Close to flush).
func (gw *GGUFWriter) WriteTensor(data []byte) error {
	if gw.next >= len(gw.tensors) {
		return fmt.Errorf("WriteTensor called more times than planned (%d tensors)", len(gw.tensors))
	}
	t := gw.tensors[gw.next]
	numel := 1
	for _, d := range t.Dims {
		numel *= int(d)
	}
	want, ok := t.DType.DataSize(numel)
	if !ok || len(data) != want {
		return fmt.Errorf("tensor %q: got %d bytes, want %d", t.Name, len(data), want)
	}
	if _, err := gw.cw.Write(data); err != nil {
		return err
	}
	gw.next++
	if gw.next < len(gw.tensors) {
		wantPos := gw.dataOffset + int64(gw.tensors[gw.next].Offset)
		switch pad := wantPos - gw.cw.pos; {
		case pad > 0:
			if _, err := gw.cw.Write(make([]byte, pad)); err != nil {
				return err
			}
		case pad < 0:
			return fmt.Errorf("internal error: tensor %q overran its planned size by %d bytes", t.Name, -pad)
		}
	}
	return gw.cw.err
}

// Close flushes any buffered output. It does not close the underlying
// io.Writer (NewGGUFWriter never took ownership of it).
func (gw *GGUFWriter) Close() error {
	if gw.next != len(gw.tensors) {
		return fmt.Errorf("GGUFWriter closed after %d of %d planned tensors", gw.next, len(gw.tensors))
	}
	if gw.cw.err != nil {
		return gw.cw.err
	}
	return gw.cw.bw.Flush()
}
