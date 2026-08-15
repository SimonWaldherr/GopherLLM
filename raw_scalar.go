package gopherllm

import (
	"encoding/binary"
	"math"
	"unsafe"
)

// rawScalarWeight reports whether w is an mmap-backed, non-quantized matrix.
// Out-of-core loading keeps these bytes in their GGUF representation instead
// of expanding F16/BF16 (or copying F32) into a process-owned []float32.
func rawScalarWeight(w Weight) bool {
	if w.F32 != nil || len(w.Raw) == 0 {
		return false
	}
	switch w.Type {
	case GGMLTypeF32, GGMLTypeF16, GGMLTypeBF16, GGMLTypeF64:
		return true
	default:
		return false
	}
}

func scalarBytesPerElement(typ GGMLType) int {
	switch typ {
	case GGMLTypeF32:
		return 4
	case GGMLTypeF16, GGMLTypeBF16:
		return 2
	case GGMLTypeF64:
		return 8
	default:
		return 0
	}
}

// f16RowView reinterprets n little-endian f16 elements at raw[offset:] as the
// []uint16 the SIMD f16 kernels take, without copying.
//
// The alias is safe for exactly the inputs this package produces: offset is
// always row*cols*2, so it is even, and raw is either an mmap (page-aligned)
// or a Go allocation (word-aligned), which makes the result 2-byte aligned.
// GGUF is little-endian and every supported GOARCH is too, so the in-memory
// uint16 and the on-disk f16 bit patterns coincide.
func f16RowView(raw []byte, offset, n int) []uint16 {
	return unsafe.Slice((*uint16)(unsafe.Pointer(&raw[offset])), n)
}

func rawScalarDot(raw []byte, typ GGMLType, offset int, x []float32, cols int) float32 {
	width := scalarBytesPerElement(typ)
	if width == 0 || offset < 0 || offset > len(raw) || cols <= 0 {
		return 0
	}
	n := min(cols, len(x), (len(raw)-offset)/width)
	// F16 has a real vector kernel on every target that has one at all —
	// VCVTPH2PS under AVX2 on amd64, FCVTL on arm64 — and it was reachable
	// only from the KV cache, never from weights. Raw f16 weights are not a
	// corner case: they are what out-of-core loading keeps on purpose (see
	// rawScalarWeight) and what a vision tower's mmproj is stored as, so this
	// scalar loop with a software F16ToF32 per element was the whole matvec
	// for those models.
	if typ == GGMLTypeF16 && n > 0 {
		return dotF32F16(x[:n], f16RowView(raw, offset, n))
	}
	var s0, s1, s2, s3 float32
	i := 0
	switch typ {
	case GGMLTypeF32:
		for ; i+4 <= n; i += 4 {
			s0 += math.Float32frombits(binary.LittleEndian.Uint32(raw[offset+i*4:])) * x[i]
			s1 += math.Float32frombits(binary.LittleEndian.Uint32(raw[offset+(i+1)*4:])) * x[i+1]
			s2 += math.Float32frombits(binary.LittleEndian.Uint32(raw[offset+(i+2)*4:])) * x[i+2]
			s3 += math.Float32frombits(binary.LittleEndian.Uint32(raw[offset+(i+3)*4:])) * x[i+3]
		}
		for ; i < n; i++ {
			s0 += math.Float32frombits(binary.LittleEndian.Uint32(raw[offset+i*4:])) * x[i]
		}
	case GGMLTypeF16:
		for ; i+4 <= n; i += 4 {
			s0 += F16ToF32(binary.LittleEndian.Uint16(raw[offset+i*2:])) * x[i]
			s1 += F16ToF32(binary.LittleEndian.Uint16(raw[offset+(i+1)*2:])) * x[i+1]
			s2 += F16ToF32(binary.LittleEndian.Uint16(raw[offset+(i+2)*2:])) * x[i+2]
			s3 += F16ToF32(binary.LittleEndian.Uint16(raw[offset+(i+3)*2:])) * x[i+3]
		}
		for ; i < n; i++ {
			s0 += F16ToF32(binary.LittleEndian.Uint16(raw[offset+i*2:])) * x[i]
		}
	case GGMLTypeBF16:
		for ; i+4 <= n; i += 4 {
			s0 += math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[offset+i*2:]))<<16) * x[i]
			s1 += math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[offset+(i+1)*2:]))<<16) * x[i+1]
			s2 += math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[offset+(i+2)*2:]))<<16) * x[i+2]
			s3 += math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[offset+(i+3)*2:]))<<16) * x[i+3]
		}
		for ; i < n; i++ {
			s0 += math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[offset+i*2:]))<<16) * x[i]
		}
	case GGMLTypeF64:
		for ; i+4 <= n; i += 4 {
			s0 += float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[offset+i*8:]))) * x[i]
			s1 += float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[offset+(i+1)*8:]))) * x[i+1]
			s2 += float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[offset+(i+2)*8:]))) * x[i+2]
			s3 += float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[offset+(i+3)*8:]))) * x[i+3]
		}
		for ; i < n; i++ {
			s0 += float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[offset+i*8:]))) * x[i]
		}
	}
	return s0 + s1 + s2 + s3
}

func (w Weight) rawScalarMatvecInto(x []float32, out *[]float32) bool {
	if !rawScalarWeight(w) || w.Rows < 0 || w.Cols <= 0 {
		return false
	}
	width := scalarBytesPerElement(w.Type)
	if width == 0 || len(w.Raw) < w.Rows*w.Cols*width {
		return false
	}
	ensureLenNoClear(out, w.Rows)
	rowBytes := w.Cols * width
	parallelRows(w.Rows, func(start, end int) {
		for row := start; row < end; row++ {
			(*out)[row] = rawScalarDot(w.Raw, w.Type, row*rowBytes, x, w.Cols)
		}
	})
	return true
}

func (w Weight) rawScalarRowInto(row, cols int, out *[]float32) bool {
	if !rawScalarWeight(w) || row < 0 || row >= w.Rows || cols < 0 || cols > w.Cols {
		return false
	}
	width := scalarBytesPerElement(w.Type)
	start := row * w.Cols * width
	if width == 0 || start < 0 || start+cols*width > len(w.Raw) {
		return false
	}
	ensureLenNoClear(out, cols)
	switch w.Type {
	case GGMLTypeF32:
		for i := range cols {
			(*out)[i] = math.Float32frombits(binary.LittleEndian.Uint32(w.Raw[start+i*4:]))
		}
	case GGMLTypeF16:
		for i := range cols {
			(*out)[i] = F16ToF32(binary.LittleEndian.Uint16(w.Raw[start+i*2:]))
		}
	case GGMLTypeBF16:
		for i := range cols {
			(*out)[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(w.Raw[start+i*2:])) << 16)
		}
	case GGMLTypeF64:
		for i := range cols {
			(*out)[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(w.Raw[start+i*8:])))
		}
	default:
		return false
	}
	return true
}

func (w Weight) rawScalarArgmaxMatvec(x []float32) (uint32, bool) {
	if !rawScalarWeight(w) || w.Rows <= 0 || w.Cols != len(x) {
		return 0, false
	}
	width := scalarBytesPerElement(w.Type)
	if width == 0 || len(w.Raw) < w.Rows*w.Cols*width {
		return 0, false
	}
	rowBytes := w.Cols * width
	return argmaxMatvecRows(w.Rows, func(row int) float32 {
		return rawScalarDot(w.Raw, w.Type, row*rowBytes, x, w.Cols)
	}), true
}
