package gopherllm

import (
	"io"

	ggufformat "github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
)

// GGMLType identifies a tensor's on-disk element encoding.
type GGMLType = ggufformat.GGMLType

const (
	GGMLTypeF32     = ggufformat.GGMLTypeF32
	GGMLTypeF16     = ggufformat.GGMLTypeF16
	GGMLTypeQ4_0    = ggufformat.GGMLTypeQ4_0
	GGMLTypeQ4_1    = ggufformat.GGMLTypeQ4_1
	GGMLTypeQ5_0    = ggufformat.GGMLTypeQ5_0
	GGMLTypeQ5_1    = ggufformat.GGMLTypeQ5_1
	GGMLTypeQ8_0    = ggufformat.GGMLTypeQ8_0
	GGMLTypeQ8_1    = ggufformat.GGMLTypeQ8_1
	GGMLTypeQ2_K    = ggufformat.GGMLTypeQ2_K
	GGMLTypeQ3_K    = ggufformat.GGMLTypeQ3_K
	GGMLTypeQ4_K    = ggufformat.GGMLTypeQ4_K
	GGMLTypeQ5_K    = ggufformat.GGMLTypeQ5_K
	GGMLTypeQ6_K    = ggufformat.GGMLTypeQ6_K
	GGMLTypeQ8_K    = ggufformat.GGMLTypeQ8_K
	GGMLTypeIQ2_XXS = ggufformat.GGMLTypeIQ2_XXS
	GGMLTypeIQ2_XS  = ggufformat.GGMLTypeIQ2_XS
	GGMLTypeIQ3_XXS = ggufformat.GGMLTypeIQ3_XXS
	GGMLTypeIQ1_S   = ggufformat.GGMLTypeIQ1_S
	GGMLTypeIQ4_NL  = ggufformat.GGMLTypeIQ4_NL
	GGMLTypeIQ3_S   = ggufformat.GGMLTypeIQ3_S
	GGMLTypeIQ2_S   = ggufformat.GGMLTypeIQ2_S
	GGMLTypeIQ4_XS  = ggufformat.GGMLTypeIQ4_XS
	GGMLTypeF64     = ggufformat.GGMLTypeF64
	GGMLTypeIQ1_M   = ggufformat.GGMLTypeIQ1_M
	GGMLTypeBF16    = ggufformat.GGMLTypeBF16
	GGMLTypeTQ1_0   = ggufformat.GGMLTypeTQ1_0
	GGMLTypeTQ2_0   = ggufformat.GGMLTypeTQ2_0
	GGMLTypeMXFP4   = ggufformat.GGMLTypeMXFP4
	GGMLTypeQ1_0    = ggufformat.GGMLTypeQ1_0
	GGMLTypeQ2_0    = ggufformat.GGMLTypeQ2_0
	GGMLTypeUnknown = ggufformat.GGMLTypeUnknown
)

type MetaValue = ggufformat.MetaValue
type TensorInfo = ggufformat.TensorInfo
type GGUFFile = ggufformat.GGUFFile
type PlannedTensor = ggufformat.PlannedTensor
type GGUFWriter = ggufformat.GGUFWriter

func ParseGGUF(data []byte) (*GGUFFile, error)      { return ggufformat.Parse(data) }
func ParseGGUFQuiet(data []byte) (*GGUFFile, error) { return ggufformat.ParseQuiet(data) }

func NewGGUFWriter(w io.Writer, metadata map[string]MetaValue, planned []PlannedTensor, alignment int) (*GGUFWriter, error) {
	return ggufformat.NewGGUFWriter(w, metadata, planned, alignment)
}

// Keep the package-local helper available to existing root-package callers.
func ggmlTypeFromUint32(v uint32) GGMLType { return ggufformat.TypeFromID(v) }
