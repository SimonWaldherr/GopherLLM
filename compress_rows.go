package gopherllm

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Float and output slabs share this budget. A single wider row is the minimum
// unit, and the existing row encoders additionally allocate one encoded row per
// active worker. No allocation scales with the number of experts in a tensor.
const compressScratchBytes = 8 << 20

func compressChunkRows(rows, cols, rowBytes int) int {
	return min(rows, max(1, compressScratchBytes/(cols*4+rowBytes)))
}

// writeCompressedTensor converts one tensor in bounded, row-aligned chunks.
// GGUF's leading dimension is contiguous; all remaining dimensions, including
// MoE experts, form one sequence of independent rows.
func writeCompressedTensor(ctx context.Context, dst io.Writer, t TensorInfo, src []byte, outType GGMLType) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(t.Dims) < 2 {
		return fmt.Errorf("tensor %q: expected a matrix or expert bank", t.Name)
	}
	maxInt := int(^uint(0) >> 1)
	numel := 1
	for _, d := range t.Dims {
		if d == 0 || d > uint64(maxInt/numel) {
			return fmt.Errorf("tensor %q: invalid or overflowing dimensions", t.Name)
		}
		numel *= int(d)
	}
	cols := int(t.Dims[0])
	rows := numel / cols
	if cols%t.DType.BlockSize() != 0 || cols%outType.BlockSize() != 0 {
		return fmt.Errorf("tensor %q: row length %d is incompatible with %s -> %s", t.Name, cols, t.DType, outType)
	}
	srcRowBytes, ok := t.DType.DataSize(cols)
	if !ok || srcRowBytes <= 0 || rows > maxInt/srcRowBytes || len(src) != rows*srcRowBytes {
		return fmt.Errorf("tensor %q: invalid source type or byte length", t.Name)
	}
	rowBytes, ok := outType.DataSize(cols)
	encode := quantizeRowFor(outType)
	decode := compressRowDecoder(t.DType)
	if !ok || rowBytes <= 0 || encode == nil || decode == nil {
		return fmt.Errorf("tensor %q: unsupported conversion %s -> %s", t.Name, t.DType, outType)
	}
	if cols > (maxInt-rowBytes)/4 {
		return fmt.Errorf("tensor %q: row scratch size overflows", t.Name)
	}
	chunkRows := compressChunkRows(rows, cols, rowBytes)
	floats := make([]float32, chunkRows*cols)
	output := make([]byte, chunkRows*rowBytes)
	for first := 0; first < rows; first += chunkRows {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := min(chunkRows, rows-first)
		parallelRows(n, func(start, end int) {
			for row := start; row < end; row++ {
				in := src[(first+row)*srcRowBytes : (first+row+1)*srcRowBytes]
				decode(in, floats[row*cols:(row+1)*cols])
			}
		})
		// Reject non-finite input before the integer encoders see it. F64
		// conversion can overflow F32, so validation follows decoding.
		for i, v := range floats[:n*cols] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("tensor %q: non-finite value in row %d", t.Name, first+i/cols)
			}
		}
		parallelRows(n, func(start, end int) {
			for row := start; row < end; row++ {
				encoded := encode(floats[row*cols:(row+1)*cols], cols)
				copy(output[row*rowBytes:(row+1)*rowBytes], encoded)
			}
		})
		if err := ctx.Err(); err != nil {
			return err
		}
		p := output[:n*rowBytes]
		written, err := dst.Write(p)
		if err != nil {
			return err
		}
		if written != len(p) {
			return io.ErrShortWrite
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

// compressRowDecoder reuses the caller's float slab for both scalar source
// formats and quantized GGUF rows. It matches tensorToF32's supported types.
func compressRowDecoder(dtype GGMLType) func([]byte, []float32) {
	switch dtype {
	case GGMLTypeF32:
		return func(raw []byte, out []float32) {
			for i := range out {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			}
		}
	case GGMLTypeF16:
		return func(raw []byte, out []float32) {
			for i := range out {
				out[i] = F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
			}
		}
	case GGMLTypeBF16:
		return func(raw []byte, out []float32) {
			for i := range out {
				out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
			}
		}
	case GGMLTypeF64:
		return func(raw []byte, out []float32) {
			for i := range out {
				out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
			}
		}
	}
	var into func([]byte, int, []float32)
	switch dtype {
	case GGMLTypeQ8_0:
		into = DequantRowQ8_0Into
	case GGMLTypeQ4_0:
		into = DequantRowQ4_0Into
	case GGMLTypeQ4_1:
		into = DequantRowQ4_1Into
	case GGMLTypeQ5_0:
		into = DequantRowQ5_0Into
	case GGMLTypeQ5_1:
		into = DequantRowQ5_1Into
	case GGMLTypeQ8_1:
		into = DequantRowQ8_1Into
	case GGMLTypeQ8_K:
		into = DequantRowQ8KInto
	case GGMLTypeQ2_K:
		into = DequantRowQ2KInto
	case GGMLTypeQ3_K:
		into = DequantRowQ3KInto
	case GGMLTypeQ4_K:
		into = DequantRowQ4KInto
	case GGMLTypeQ5_K:
		into = DequantRowQ5KInto
	case GGMLTypeQ6_K:
		into = DequantRowQ6KInto
	case GGMLTypeMXFP4:
		into = DequantRowMXFP4Into
	case GGMLTypeTQ1_0:
		into = DequantRowTQ1_0Into
	case GGMLTypeTQ2_0:
		into = DequantRowTQ2_0Into
	case GGMLTypeQ1_0:
		into = DequantRowQ1_0Into
	case GGMLTypeQ2_0:
		into = DequantRowQ2_0Into
	default:
		return nil
	}
	return func(raw []byte, out []float32) { into(raw, len(out), out) }
}
