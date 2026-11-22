// Package phonon reads Fermion Research's five-value Parakeet weight container.
package phonon

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

// Format is the identifier in the uncompressed model.fermion's JSON header.
const Format = "fermion-five-value-parakeet-v1"

const maxHeaderBytes = 8 << 20

type header struct {
	Format string  `json:"format"`
	Index  []entry `json:"index"`
}

type entry struct {
	Name  string `json:"n"`
	Kind  string `json:"k"`
	Shape []int  `json:"shape"`
	Bytes int    `json:"b"`
}

// Model indexes tensors by their names in the Parakeet state dictionary.
type Model struct {
	Tensors map[string]Tensor
}

// Tensor is a validated weight record. Shape is in row-major order. Data
// borrows the original container. Treat both slices as read-only.
type Tensor struct {
	Shape []int
	Kind  string
	Data  []byte

	count            int
	signBytes        int
	scaleStart       int
	magnitudeOffsets []int
}

// Parse reads the uncompressed Fermion container, borrowing each payload
// from data. The caller keeps data alive and unchanged until decoding ends.
func Parse(data []byte) (*Model, error) {
	h, offset, err := readHeader(data)
	if err != nil {
		return nil, err
	}
	tensors := make(map[string]Tensor, len(h.Index))
	for _, e := range h.Index {
		if e.Name == "" {
			return nil, errors.New("phonon: tensor name is empty")
		}
		if e.Bytes <= 0 || e.Bytes > len(data)-offset {
			return nil, fmt.Errorf("phonon: tensor %q has invalid payload length %d", e.Name, e.Bytes)
		}
		name := e.Name
		if e.Kind == "five_value" {
			name += ".weight"
		}
		if _, ok := tensors[name]; ok {
			return nil, fmt.Errorf("phonon: duplicate tensor %q", name)
		}
		end := offset + e.Bytes
		t, err := validateTensor(e.Kind, e.Shape, data[offset:end:end])
		if err != nil {
			return nil, fmt.Errorf("phonon: tensor %q: %w", name, err)
		}
		tensors[name] = t
		offset = end
	}
	if offset != len(data) {
		return nil, fmt.Errorf("phonon: %d trailing bytes", len(data)-offset)
	}
	return &Model{Tensors: tensors}, nil
}

// CheckHeader checks the format and presence of a tensor index. data must
// include the length and complete JSON header. It may omit the payload.
func CheckHeader(data []byte) error {
	_, _, err := readHeader(data)
	return err
}

func readHeader(data []byte) (header, int, error) {
	if len(data) < 8 {
		return header{}, 0, errors.New("phonon: truncated header length")
	}
	size := binary.LittleEndian.Uint64(data[:8])
	if size == 0 || size > maxHeaderBytes || size > uint64(len(data)-8) {
		return header{}, 0, fmt.Errorf("phonon: invalid header length %d", size)
	}
	offset := 8 + int(size)
	var h header
	if err := json.Unmarshal(data[8:offset], &h); err != nil {
		return header{}, 0, fmt.Errorf("phonon: invalid header: %w", err)
	}
	if h.Format != Format {
		return header{}, 0, fmt.Errorf("phonon: unsupported container format %q", h.Format)
	}
	if len(h.Index) == 0 {
		return header{}, 0, errors.New("phonon: tensor index is empty")
	}
	return h, offset, nil
}

func validateTensor(kind string, shape []int, data []byte) (Tensor, error) {
	if shape == nil {
		return Tensor{}, errors.New("missing shape")
	}
	// Keep both the product and the size of a decoded F32 slice in int range.
	count := 1
	for _, dim := range shape {
		if dim <= 0 || count > (math.MaxInt/4)/dim {
			return Tensor{}, fmt.Errorf("invalid shape %v", shape)
		}
		count *= dim
	}
	t := Tensor{Shape: shape, Kind: kind, Data: data, count: count}
	if kind == "fp16" {
		if len(data) != 2*count {
			return Tensor{}, errors.New("invalid FP16 payload length")
		}
		// The model's unused batch-normalization counters are scalar and
		// some overflowed when exported as FP16.
		if len(shape) == 0 {
			return t, nil
		}
		return t, checkHalfValues(data)
	}
	if len(shape) == 0 {
		return Tensor{}, errors.New("packed tensor has no rows")
	}
	rows := shape[0]
	switch kind {
	case "five_value":
		if len(shape) != 2 {
			return Tensor{}, errors.New("five-value tensor must be a matrix")
		}
		cols := shape[1]
		rowBytes := (cols + 4) / 5
		t.signBytes = rowBytes * rows
		if t.signBytes > len(data) {
			return Tensor{}, errors.New("truncated ternary signs")
		}
		lastDigits := cols - (rowBytes-1)*5
		nonzero := 0
		t.magnitudeOffsets = make([]int, rows)
		for row := range rows {
			t.magnitudeOffsets[row] = nonzero
			codes := data[row*rowBytes : (row+1)*rowBytes]
			for _, code := range codes[:len(codes)-1] {
				if code >= 243 {
					return Tensor{}, fmt.Errorf("invalid ternary byte %d in row %d", code, row)
				}
				nonzero += int(ternary[code].nonzero[4])
			}
			last := codes[len(codes)-1]
			if last >= 243 {
				return Tensor{}, fmt.Errorf("invalid ternary byte %d in row %d", last, row)
			}
			nonzero += int(ternary[last].nonzero[lastDigits-1])
		}
		t.scaleStart = t.signBytes + (nonzero+7)/8
		if t.scaleStart > len(data) || len(data)-t.scaleStart != 4*rows {
			return Tensor{}, errors.New("invalid five-value magnitude or scale size")
		}
	case "int6", "int8":
		t.scaleStart = count
		if kind == "int6" {
			t.scaleStart = ((count + 3) / 4) * 3
		}
		if t.scaleStart > len(data) || len(data)-t.scaleStart != 2*rows {
			return Tensor{}, errors.New("invalid integer payload or scale size")
		}
	default:
		return Tensor{}, fmt.Errorf("unsupported encoding %q", kind)
	}
	return t, checkHalfValues(data[t.scaleStart:])
}

// DecodeFiveValueRows expands rows [start,end) of a validated five-value
// tensor into a slice of exactly (end-start)*cols elements. Independent
// ranges can be decoded concurrently.
func (t Tensor) DecodeFiveValueRows(out []float32, start, end int) {
	rows, cols := t.Shape[0], t.Shape[1]
	rowBytes := t.signBytes / rows
	stream := t.Data[t.signBytes:]
	scales := t.Data[t.scaleStart:]
	for row := start; row < end; row++ {
		lo, hi := halfAt(scales, row), halfAt(scales, rows+row)
		values := [8]float32{0 * lo, lo, hi, -lo, -hi}
		codes := t.Data[row*rowBytes : (row+1)*rowBytes]
		dst := out[(row-start)*cols : (row-start+1)*cols]
		bit := t.magnitudeOffsets[row]
		full := cols / 5
		for i, code := range codes[:full] {
			// The scale bytes follow the magnitude stream. The last
			// two-byte read may include them, but the lookup consumes only
			// the bits for nonzero signs in this group.
			offset := bit / 8
			bucket := (uint16(stream[offset]) | uint16(stream[offset+1])<<8) >> (bit & 7)
			entry := &fiveValueGroups[code]
			indices := entry.values[bucket&31]
			bit += int(entry.nonzero)
			o := 5 * i
			dst[o+4] = values[indices[4]]
			dst[o+3] = values[indices[3]]
			dst[o+2] = values[indices[2]]
			dst[o+1] = values[indices[1]]
			dst[o] = values[indices[0]]
		}
		if tail := dst[full*5:]; len(tail) != 0 {
			offset := bit / 8
			bucket := (uint16(stream[offset]) | uint16(stream[offset+1])<<8) >> (bit & 7)
			indices := fiveValueGroups[codes[full]].values[bucket&31]
			for i := range tail {
				tail[i] = values[indices[i]]
			}
		}
	}
}

type fiveValueGroup struct {
	values  [32][5]uint8
	nonzero uint8
}

var fiveValueGroups = makeFiveValueGroups()

func makeFiveValueGroups() [243]fiveValueGroup {
	var table [243]fiveValueGroup
	for code, digits := range ternary {
		table[code].nonzero = digits.nonzero[4]
		for mask := range 32 {
			bit := 0
			for i, sign := range digits.sign {
				if sign == 0 {
					continue
				}
				index := 1 + uint8(mask>>bit&1)
				if sign < 0 {
					index += 2
				}
				table[code].values[mask][i] = index
				bit++
			}
		}
	}
	return table
}

func checkHalfValues(data []byte) error {
	for i := 0; i < len(data); i += 2 {
		if data[i+1]&0x7c == 0x7c {
			return errors.New("non-finite FP16 value")
		}
	}
	return nil
}

type trits struct {
	sign    [5]int8
	nonzero [5]uint8
}

var ternary = makeTernaryTable()

func makeTernaryTable() [243]trits {
	var table [243]trits
	for code := range table {
		digits := code
		var nonzero uint8
		for i := range 5 {
			sign := int8(digits%3) - 1
			digits /= 3
			if sign != 0 {
				nonzero++
			}
			table[code].sign[i] = sign
			table[code].nonzero[i] = nonzero
		}
	}
	return table
}

// NumElements reports the number of FP32 elements in the decoded tensor.
func (t Tensor) NumElements() int { return t.count }

// Decode allocates a row-major FP32 copy of the tensor.
func (t Tensor) Decode() ([]float32, error) {
	out := make([]float32, t.count)
	if err := t.DecodeInto(out); err != nil {
		return nil, err
	}
	return out, nil
}

// DecodeInto expands the tensor into an existing row-major FP32 buffer.
// The destination length must equal NumElements.
func (t Tensor) DecodeInto(out []float32) error {
	if t.count == 0 || len(out) != t.count {
		return fmt.Errorf("phonon: decode buffer has %d elements for a tensor with %d", len(out), t.count)
	}
	switch t.Kind {
	case "fp16":
		for i := range out {
			out[i] = halfAt(t.Data, i)
		}
	case "five_value":
		t.decodeFiveValue(out, nil)
	case "int6":
		t.decodeInt6(out)
	case "int8":
		t.decodeInt8(out)
	default:
		return fmt.Errorf("phonon: unsupported encoding %q", t.Kind)
	}
	return nil
}

// DecodeF16 expands a five-value matrix to little-endian FP16, preserving
// the exact row-scale bits with the appropriate sign for each weight.
func (t Tensor) DecodeF16() ([]byte, error) {
	if t.Kind != "five_value" || t.count == 0 {
		return nil, errors.New("phonon: DecodeF16 requires a five-value tensor")
	}
	out := make([]byte, t.count*2)
	t.decodeFiveValue(nil, out)
	return out, nil
}

func halfAt(data []byte, i int) float32 {
	return numeric.F16ToF32(binary.LittleEndian.Uint16(data[2*i:]))
}

func (t Tensor) decodeFiveValue(out []float32, raw []byte) {
	rows, cols := t.Shape[0], t.Shape[1]
	rowBytes := t.signBytes / rows
	signs := t.Data[:t.signBytes]
	magnitude := t.Data[t.signBytes:t.scaleStart]
	scales := t.Data[t.scaleStart:]
	// Magnitudes are packed as one continuous LSB-first bit stream across
	// rows. Only nonzero signs consume a bit. The signs themselves are
	// row-padded ternary digits, with the first column in the lowest trit.
	nonzero := 0
	for row := range rows {
		lo := binary.LittleEndian.Uint16(scales[row*2:])
		hi := binary.LittleEndian.Uint16(scales[(rows+row)*2:])
		base := row * cols
		column := 0
		for _, code := range signs[row*rowBytes : (row+1)*rowBytes] {
			digits := ternary[code].sign
			for _, sign := range digits[:min(5, cols-column)] {
				bits := lo & 0x8000
				if sign != 0 {
					bits = lo
					if magnitude[nonzero/8]&(1<<uint(nonzero%8)) != 0 {
						bits = hi
					}
					if sign < 0 {
						bits ^= 0x8000
					}
					nonzero++
				}
				i := base + column
				if raw != nil {
					binary.LittleEndian.PutUint16(raw[i*2:], bits)
				} else {
					out[i] = numeric.F16ToF32(bits)
				}
				column++
			}
		}
	}
}

func (t Tensor) decodeInt6(out []float32) {
	rows := t.Shape[0]
	cols := t.count / rows
	scales := t.Data[t.scaleStart:]
	var packed uint32
	var used, off int
	for row := range rows {
		scale := halfAt(scales, row)
		for col := range cols {
			if used == 0 {
				packed = uint32(t.Data[off]) | uint32(t.Data[off+1])<<8 | uint32(t.Data[off+2])<<16
				off += 3
			}
			out[row*cols+col] = float32(int(packed&63)-32) * scale
			packed >>= 6
			used = (used + 1) & 3
		}
	}
}

func (t Tensor) decodeInt8(out []float32) {
	rows := t.Shape[0]
	cols := t.count / rows
	scales := t.Data[t.scaleStart:]
	for row := range rows {
		scale := halfAt(scales, row)
		for i := row * cols; i < (row+1)*cols; i++ {
			out[i] = float32(int8(t.Data[i])) * scale
		}
	}
}
