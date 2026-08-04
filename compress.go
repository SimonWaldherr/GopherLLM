package gopherllm

// Post-training model compression (Phase A: plain round-to-nearest
// requantization — the calibration-driven GPTQ/AWQ/SmoothQuant/SparseGPT
// methods land in later files/phases). CompressModel reads a source GGUF,
// requantizes eligible weight tensors to a target format, and writes the
// result to a new GGUF file, following the same mmap+ParseGGUF pattern
// analyze.go/inspectGGUF already use for header-only work — this just also
// touches tensor bytes.

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// tensorToF32 fully decodes raw as numel float32 values, regardless of
// whether dtype is a plain float format or a quantized one. dequantTensor
// (model.go) only covers the quantized formats — F32/F16/F64/BF16 aren't
// "quantized" in that sense, so requantizing a typical unquantized source
// GGUF (this feature's main use case) needs this small extra step first.
func tensorToF32(dtype GGMLType, raw []byte, numel int) ([]float32, bool) {
	switch dtype {
	case GGMLTypeF32:
		f := make([]float32, numel)
		for i := range numel {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return f, true
	case GGMLTypeF16:
		f := make([]float32, numel)
		for i := range numel {
			f[i] = F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return f, true
	case GGMLTypeF64:
		f := make([]float32, numel)
		for i := range numel {
			f[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return f, true
	case GGMLTypeBF16:
		f := make([]float32, numel)
		for i := range numel {
			f[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return f, true
	default:
		return dequantTensor(dtype, raw, numel)
	}
}

// ParseCompressFormat maps a CLI-facing format name to the GGMLType
// CompressModel accepts, restricted to the formats quantize_rtn.go has
// encoders for (the full GGMLType enum covers many more read-only formats
// that aren't valid compression targets here).
func ParseCompressFormat(name string) (GGMLType, bool) {
	switch name {
	case "Q8_0", "q8_0":
		return GGMLTypeQ8_0, true
	case "Q4_0", "q4_0":
		return GGMLTypeQ4_0, true
	case "Q4_K", "q4_k", "Q4_K_M", "q4_k_m":
		return GGMLTypeQ4_K, true
	case "Q6_K", "q6_k":
		return GGMLTypeQ6_K, true
	default:
		return 0, false
	}
}

// CompressOptions configures CompressModel.
type CompressOptions struct {
	// TargetFormat is the GGML type every eligible weight tensor is
	// requantized to. Only Q8_0, Q4_0, Q4_K, Q6_K are supported (the
	// formats quantize_rtn.go has encoders for).
	TargetFormat GGMLType
	// LogWriter receives one line per tensor decision (requantized,
	// passed through verbatim, and why). Defaults to io.Discard.
	LogWriter io.Writer
}

// tensorScale reports numel/BlockSize for the target format, and whether
// numel divides evenly — a tensor that doesn't can't be requantized into
// this format's fixed block layout and is passed through unchanged.
func blockDivides(dtype GGMLType, numel int) bool {
	bs := dtype.BlockSize()
	return bs > 0 && numel%bs == 0
}

// quantizeRowFor returns the per-row RTN encoder for dtype, or nil if dtype
// isn't one of the formats quantize_rtn.go supports.
func quantizeRowFor(dtype GGMLType) func(row []float32, cols int) []byte {
	switch dtype {
	case GGMLTypeQ8_0:
		return QuantizeRowQ8_0
	case GGMLTypeQ4_0:
		return QuantizeRowQ4_0
	case GGMLTypeQ4_K:
		return QuantizeRowQ4K
	case GGMLTypeQ6_K:
		return QuantizeRowQ6K
	default:
		return nil
	}
}

// isCompressible reports whether t is the kind of tensor Phase A touches at
// all: a 2-D-or-higher weight matrix, as opposed to a 1-D norm/bias vector
// (always left as F32 — the savings are negligible and the quality risk
// isn't, matching llama.cpp's own quantize tool convention).
func isCompressibleTensor(t TensorInfo) bool {
	return len(t.Dims) >= 2
}

// planTensor decides one tensor's OUTPUT type: the target format if it's a
// compressible weight matrix whose row length divides the format's block
// size, otherwise the tensor's own current type (verbatim passthrough).
// Every tensor gets a decision up front, before any bytes are written,
// because GGUF's descriptor table needs every tensor's final offset before
// the header is serialized (see gguf_write.go).
func planTensor(t TensorInfo, target GGMLType) (outType GGMLType, requantize bool) {
	if !isCompressibleTensor(t) {
		return t.DType, false
	}
	rowLen := int(t.Dims[0]) // GGUF's fastest-varying dim is the row/input length
	if !blockDivides(target, rowLen) {
		return t.DType, false
	}
	if t.DType == target {
		return t.DType, false // already the target format, nothing to do
	}
	return target, true
}

// rowsOf returns a tensor's (rows, cols) as used by the per-row encoders:
// Dims[0] is the fastest-varying (column/input) dimension, Dims[1..] the
// row/output dimensions collapsed together (matching Weight.Rows/Cols,
// model.go).
func rowsOf(t TensorInfo) (rows, cols int) {
	cols = int(t.Dims[0])
	rows = 1
	for _, d := range t.Dims[1:] {
		rows *= int(d)
	}
	return rows, cols
}

// CompressModel reads sourcePath's GGUF and writes a requantized copy to
// outPath. Weight tensors already at a higher-than-target precision (or
// even a different quantized format — dequantized via the existing,
// trusted dequantTensor first) are requantized row by row; everything else
// (norms, biases, tensors whose row length doesn't fit the target format's
// block size, tensors already at the target format) is copied through
// unchanged.
func CompressModel(sourcePath, outPath string, opts CompressOptions) error {
	quantizeRow := quantizeRowFor(opts.TargetFormat)
	if quantizeRow == nil {
		return fmt.Errorf("compress: unsupported target format %s (supported: Q8_0, Q4_0, Q4_K, Q6_K)", opts.TargetFormat)
	}
	logw := opts.LogWriter
	if logw == nil {
		logw = io.Discard
	}

	mmap, err := OpenMmap(sourcePath)
	if err != nil {
		return fmt.Errorf("compress: open %s: %w", sourcePath, err)
	}
	defer mmap.Close()
	src, err := ParseGGUF(mmap.Bytes())
	if err != nil {
		return fmt.Errorf("compress: parse %s: %w", sourcePath, err)
	}
	raw := mmap.Bytes()

	// Phase 1: decide every tensor's output type up front (required by
	// GGUFWriter's two-phase contract) and log the decision.
	planned := make([]PlannedTensor, len(src.Tensors))
	requantize := make([]bool, len(src.Tensors))
	var beforeBytes, afterBytes int64
	for i, t := range src.Tensors {
		outType, doRequant := planTensor(t, opts.TargetFormat)
		planned[i] = PlannedTensor{Name: t.Name, Dims: t.Dims, DType: outType}
		requantize[i] = doRequant
		before, _ := t.DType.DataSize(t.Numel())
		after, _ := outType.DataSize(t.Numel())
		beforeBytes += int64(before)
		afterBytes += int64(after)
		if doRequant {
			fmt.Fprintf(logw, "requantize %-40s %s -> %s (%d bytes -> %d bytes)\n", t.Name, t.DType, outType, before, after)
		} else {
			fmt.Fprintf(logw, "passthrough %-40s %s\n", t.Name, t.DType)
		}
	}
	fmt.Fprintf(logw, "total: %d bytes -> %d bytes (%.1f%%)\n", beforeBytes, afterBytes, 100*float64(afterBytes)/float64(beforeBytes))

	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("compress: create %s: %w", outPath, err)
	}
	defer out.Close()

	alignment := int(src.GetU32("general.alignment", 32))
	gw, err := NewGGUFWriter(out, src.Metadata, planned, alignment)
	if err != nil {
		return fmt.Errorf("compress: write header: %w", err)
	}

	// Phase 2: stream each tensor's bytes, in the same order as planned.
	for i, t := range src.Tensors {
		size, ok := t.DType.DataSize(t.Numel())
		if !ok {
			return fmt.Errorf("compress: tensor %q: unsupported source type %s", t.Name, t.DType)
		}
		start := src.DataOffset + int(t.Offset)
		if start < 0 || start+size > len(raw) {
			return fmt.Errorf("compress: tensor %q: out of bounds (offset %d, size %d, file %d bytes)", t.Name, start, size, len(raw))
		}
		srcBytes := raw[start : start+size]

		var outBytes []byte
		if !requantize[i] {
			outBytes = srcBytes
		} else {
			rows, cols := rowsOf(t)
			f32, ok := tensorToF32(t.DType, srcBytes, t.Numel())
			if !ok {
				return fmt.Errorf("compress: tensor %q: cannot dequantize source type %s for requantization", t.Name, t.DType)
			}
			outSize, _ := opts.TargetFormat.DataSize(t.Numel())
			outBytes = make([]byte, 0, outSize)
			for r := 0; r < rows; r++ {
				rowBytes := quantizeRow(f32[r*cols:(r+1)*cols], cols)
				outBytes = append(outBytes, rowBytes...)
			}
		}
		if err := gw.WriteTensor(outBytes); err != nil {
			return fmt.Errorf("compress: tensor %q: %w", t.Name, err)
		}
	}
	return gw.Close()
}
