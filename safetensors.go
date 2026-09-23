package gopherllm

import "github.com/SimonWaldherr/GopherLLM/internal/formats/safetensors"

// SafetensorsDType is the stored numeric representation of a safetensors
// tensor. The root alias keeps the original API available while the parser
// lives in its format package.
type SafetensorsDType = safetensors.DType

// SafetensorsTensorInfo is one tensor's entry in a safetensors header.
type SafetensorsTensorInfo = safetensors.SafetensorsTensorInfo

// SafetensorsFile is a parsed safetensors container. Data is the raw file
// bytes (typically mmap-backed); tensor bytes are indexed by DataOffset,
// Begin, and End.
type SafetensorsFile = safetensors.SafetensorsFile

// Public dtype constants for callers inspecting safetensors metadata.
const (
	SafetensorsF32  = safetensors.F32
	SafetensorsF16  = safetensors.F16
	SafetensorsBF16 = safetensors.BF16
)

// ParseSafetensors reads a safetensors container's header and indexes every
// tensor's location, without copying or decoding tensor data.
func ParseSafetensors(data []byte) (*SafetensorsFile, error) {
	return safetensors.Parse(data)
}
