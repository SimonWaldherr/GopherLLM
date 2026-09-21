package gopherllm

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// This file parses the safetensors container format: an 8-byte little-endian
// header length, that many bytes of JSON describing every tensor (dtype,
// shape, and a [begin,end) byte range within the data section), then the raw
// tensor data itself starting immediately after the header. Unlike GGUF,
// there is no quantization here -- tensors are dense fp16/bf16/fp32 (or,
// rarely, other IEEE/int types), and shapes are already in the row-major
// [outermost..innermost] order callers expect, needing no dims[0]/dims[1]
// swap the way GGUF's tensor metadata does.
//
// Written for loading official (non-GGUF) Hugging Face / Mistral model
// releases such as mistralai/Voxtral-Mini-4B-Realtime-2602's
// consolidated.safetensors, where GopherLLM's own GGUF-first pipeline has no
// applicable file to load.

type safetensorsDType string

const (
	safetensorsF32  safetensorsDType = "F32"
	safetensorsF16  safetensorsDType = "F16"
	safetensorsBF16 safetensorsDType = "BF16"
)

// SafetensorsTensorInfo is one tensor's entry in the safetensors header.
type SafetensorsTensorInfo struct {
	DType safetensorsDType
	Shape []int
	Begin int // byte offset within the data section (after the header)
	End   int
	Numel int
}

// SafetensorsFile is a parsed safetensors container. Data is the raw file
// bytes (typically mmap-backed); tensor bytes are Data[DataOffset+Begin :
// DataOffset+End].
type SafetensorsFile struct {
	Tensors    map[string]SafetensorsTensorInfo
	DataOffset int
	Data       []byte
}

// ParseSafetensors reads a safetensors container's header and indexes every
// tensor's location, without copying or decoding tensor data.
func ParseSafetensors(data []byte) (*SafetensorsFile, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors: file too short for header length")
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	if headerLen == 0 || headerLen > uint64(len(data))-8 {
		return nil, fmt.Errorf("safetensors: invalid header length %d", headerLen)
	}
	dataOffset := 8 + int(headerLen)
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data[8:dataOffset], &raw); err != nil {
		return nil, fmt.Errorf("safetensors: parsing header: %w", err)
	}
	tensors := make(map[string]SafetensorsTensorInfo, len(raw))
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var entry struct {
			DType       safetensorsDType `json:"dtype"`
			Shape       []int            `json:"shape"`
			DataOffsets [2]int           `json:"data_offsets"`
		}
		if err := json.Unmarshal(msg, &entry); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %s: %w", name, err)
		}
		numel := 1
		for _, d := range entry.Shape {
			numel *= d
		}
		tensors[name] = SafetensorsTensorInfo{
			DType: entry.DType, Shape: entry.Shape,
			Begin: entry.DataOffsets[0], End: entry.DataOffsets[1], Numel: numel,
		}
	}
	if dataOffset > len(data) {
		return nil, fmt.Errorf("safetensors: header claims data section beyond file length")
	}
	return &SafetensorsFile{Tensors: tensors, DataOffset: dataOffset, Data: data}, nil
}

// F32 decodes a tensor to float32, converting from its stored dtype.
// Quantized/integer dtypes are not supported -- official model releases
// ship dense fp16/bf16/fp32 weights, never GGUF-style block quantization.
func (s *SafetensorsFile) F32(name string) ([]float32, error) {
	info, ok := s.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: missing tensor %s", name)
	}
	begin, end := s.DataOffset+info.Begin, s.DataOffset+info.End
	if begin < 0 || end > len(s.Data) || begin > end {
		return nil, fmt.Errorf("safetensors: tensor %s byte range [%d,%d) out of bounds", name, info.Begin, info.End)
	}
	raw := s.Data[begin:end]
	out := make([]float32, info.Numel)
	switch info.DType {
	case safetensorsF32:
		if len(raw) != info.Numel*4 {
			return nil, fmt.Errorf("safetensors: tensor %s has %d bytes, want %d for F32", name, len(raw), info.Numel*4)
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case safetensorsF16:
		if len(raw) != info.Numel*2 {
			return nil, fmt.Errorf("safetensors: tensor %s has %d bytes, want %d for F16", name, len(raw), info.Numel*2)
		}
		for i := range out {
			out[i] = F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case safetensorsBF16:
		if len(raw) != info.Numel*2 {
			return nil, fmt.Errorf("safetensors: tensor %s has %d bytes, want %d for BF16", name, len(raw), info.Numel*2)
		}
		for i := range out {
			// bfloat16 is exactly the top 16 bits of a float32 (same sign/
			// exponent width, a truncated mantissa), so widening is a shift,
			// not a bit-pattern remap the way IEEE float16 needs.
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	default:
		return nil, fmt.Errorf("safetensors: tensor %s has unsupported dtype %s", name, info.DType)
	}
	return out, nil
}
