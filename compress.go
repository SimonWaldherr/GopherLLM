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
	"time"
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
	case "Q5_K", "q5_k", "Q5_K_M", "q5_k_m":
		return GGMLTypeQ5_K, true
	case "Q6_K", "q6_k":
		return GGMLTypeQ6_K, true
	default:
		return 0, false
	}
}

// compressFormatRank orders the compress-target formats by approximate
// quality/size, ascending. Used only to decide whether the output/embedding
// quality floor (see CompressOptions.Uniform) needs to raise a tensor's
// format above the user's chosen target — never to pick the target itself.
func compressFormatRank(t GGMLType) int {
	switch t {
	case GGMLTypeQ4_0:
		return 0
	case GGMLTypeQ4_K:
		return 1
	case GGMLTypeQ5_K:
		return 2
	case GGMLTypeQ6_K:
		return 3
	case GGMLTypeQ8_0:
		return 4
	default:
		return -1
	}
}

// outputFloorFormat is llama.cpp's own convention for its quantize tool:
// the token embedding and output/LM-head tensors are disproportionately
// sensitive to quantization error, so even aggressive presets keep them at
// Q6_K rather than the base target. namedOutputTensors are the two
// well-known GGUF tensor names this applies to across every architecture
// this codebase loads (model.go, gemma4.go, qwen35.go, deepseek2.go all
// agree on these names).
const outputFloorFormat = GGMLTypeQ6_K

var namedOutputTensors = map[string]bool{
	"token_embd.weight": true,
	"output.weight":     true,
}

// CompressOptions configures CompressModel.
type CompressOptions struct {
	// TargetFormat is the GGML type every eligible weight tensor is
	// requantized to. Only Q8_0, Q4_0, Q4_K, Q5_K, Q6_K are supported (the
	// formats quantize_rtn.go has encoders for).
	TargetFormat GGMLType
	// Uniform disables the output/embedding quality floor (outputFloorFormat)
	// and applies TargetFormat everywhere without exception. Default false
	// matches llama.cpp's own quantize tool convention; set true only for
	// research/testing that specifically wants strictly uniform quantization.
	Uniform bool
	// LogWriter receives one line per tensor decision (requantized, passed
	// through verbatim, and why) plus periodic progress during large
	// compressions. Defaults to io.Discard.
	LogWriter io.Writer
}

// blockDivides reports whether numel divides evenly into dtype's block
// size — a tensor that doesn't can't be requantized into this format's
// fixed block layout and is passed through unchanged.
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
	case GGMLTypeQ5_K:
		return QuantizeRowQ5K
	case GGMLTypeQ6_K:
		return QuantizeRowQ6K
	default:
		return nil
	}
}

// isCompressibleTensor reports whether t is the kind of tensor Phase A
// touches at all: a 2-D-or-higher weight matrix, as opposed to a 1-D
// norm/bias vector (always left as F32 — the savings are negligible and the
// quality risk isn't, matching llama.cpp's own quantize tool convention).
func isCompressibleTensor(t TensorInfo) bool {
	return len(t.Dims) >= 2
}

// planTensor decides one tensor's OUTPUT type: the target format (or,
// unless opts.Uniform, outputFloorFormat for the two named output/embedding
// tensors when that floor is higher quality than target) if it's a
// compressible weight matrix whose row length divides the chosen format's
// block size, otherwise the tensor's own current type (verbatim
// passthrough). Every tensor gets a decision up front, before any bytes are
// written, because GGUF's descriptor table needs every tensor's final
// offset before the header is serialized (see gguf_write.go).
func planTensor(t TensorInfo, opts CompressOptions) (outType GGMLType, requantize bool) {
	if !isCompressibleTensor(t) {
		return t.DType, false
	}
	target := opts.TargetFormat
	if !opts.Uniform && namedOutputTensors[t.Name] && compressFormatRank(outputFloorFormat) > compressFormatRank(target) {
		target = outputFloorFormat
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

// sameFile reports whether a and b name the same file on disk (by identity,
// not just string equality — catches relative-vs-absolute paths, symlinks,
// and hardlinks). A path that doesn't exist yet (the common case for a
// fresh --compress-out) is never the same file as an existing source.
func sameFile(a, b string) bool {
	ai, err := os.Stat(a)
	if err != nil {
		return false
	}
	bi, err := os.Stat(b)
	if err != nil {
		return false
	}
	return os.SameFile(ai, bi)
}

// CompressModel reads sourcePath's GGUF and writes a requantized copy to
// outPath. Weight tensors already at a higher-than-target precision (or
// even a different quantized format — dequantized via the existing,
// trusted dequantTensor first) are requantized row by row, in parallel
// across rows; everything else (norms, biases, tensors whose row length
// doesn't fit the target format's block size, tensors already at the target
// format) is copied through unchanged.
//
// outPath must not be the same file as sourcePath: CompressModel keeps
// sourcePath memory-mapped and readable throughout, and truncating that
// same file via outPath's os.Create would corrupt the mapping mid-read
// (SIGBUS on Unix; an outright open failure on Windows, where the mapping
// keeps the file locked). Compress to a temporary path and rename it into
// place afterward if an in-place update is what's wanted.
//
// A source that is one shard of a split/sharded GGUF (split.count > 1 in
// its metadata) is rejected outright rather than silently compressing only
// that shard's tensor subset — merging shards first is a real, separate
// feature (see gguf_split.go's loadSplitRunner), not something Phase A
// attempts.
func CompressModel(sourcePath, outPath string, opts CompressOptions) error {
	quantizeRow := quantizeRowFor(opts.TargetFormat)
	if quantizeRow == nil {
		return fmt.Errorf("compress: unsupported target format %s (supported: Q8_0, Q4_0, Q4_K, Q5_K, Q6_K)", opts.TargetFormat)
	}
	if sameFile(sourcePath, outPath) {
		return fmt.Errorf("compress: --compress-out must not be the same file as the source model (%s); compress to a different path and rename it afterward if you want to replace the original", sourcePath)
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
	if shardNo, shardCount, ok := splitInfo(src); ok {
		return fmt.Errorf("compress: %s is shard %d/%d of a split GGUF; --compress needs a single merged file (merging split shards is not yet supported by this command)", sourcePath, shardNo+1, shardCount)
	}
	raw := mmap.Bytes()

	// Phase 1: decide every tensor's output type up front (required by
	// GGUFWriter's two-phase contract) and log the decision.
	planned := make([]PlannedTensor, len(src.Tensors))
	requantize := make([]bool, len(src.Tensors))
	var beforeBytes, afterBytes int64
	for i, t := range src.Tensors {
		outType, doRequant := planTensor(t, opts)
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

	// split.no/split.count/split.tensors.count are rejected above for the
	// common case (a raw shard), but strip them defensively too: a
	// hand-edited or already-merged-then-relabeled source could still carry
	// them, and they would otherwise describe a shard layout that no longer
	// exists in the (single-file) output.
	metadata := src.Metadata
	for _, key := range splitKeys {
		if _, ok := metadata[key]; ok {
			if metadata == nil || len(metadata) == len(src.Metadata) {
				cp := make(map[string]MetaValue, len(src.Metadata))
				for k, v := range src.Metadata {
					cp[k] = v
				}
				metadata = cp
			}
			delete(metadata, key)
		}
	}

	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("compress: create %s: %w", outPath, err)
	}
	defer out.Close()

	alignment := int(src.GetU32("general.alignment", 32))
	gw, err := NewGGUFWriter(out, metadata, planned, alignment)
	if err != nil {
		return fmt.Errorf("compress: write header: %w", err)
	}

	// Phase 2: encode each tensor (rows in parallel across numThreads()
	// cores — RTN is pure per-row scalar arithmetic with no cross-row
	// dependency, so this is embarrassingly parallel), then stream bytes in
	// the same order as planned. Progress is logged periodically, not per
	// tensor, so a multi-hundred-tensor model doesn't spam the log while a
	// multi-minute run still gives visible signs of life.
	lastProgress := time.Now()
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
			outType := planned[i].DType
			rowSize, ok := outType.DataSize(cols)
			if !ok {
				return fmt.Errorf("compress: tensor %q: cannot size target type %s", t.Name, outType)
			}
			outBytes = make([]byte, rows*rowSize)
			parallelRows(rows, func(rstart, rend int) {
				for r := rstart; r < rend; r++ {
					rowBytes := quantizeRow(f32[r*cols:(r+1)*cols], cols)
					copy(outBytes[r*rowSize:(r+1)*rowSize], rowBytes)
				}
			})
		}
		if err := gw.WriteTensor(outBytes); err != nil {
			return fmt.Errorf("compress: tensor %q: %w", t.Name, err)
		}
		if now := time.Now(); now.Sub(lastProgress) > 3*time.Second {
			lastProgress = now
			fmt.Fprintf(logw, "... %d/%d tensors (%s)\n", i+1, len(src.Tensors), t.Name)
		}
	}
	if err := gw.Close(); err != nil {
		return err
	}

	return verifyCompressedOutput(outPath, planned)
}

// verifyCompressedOutput re-opens a just-written GGUF and checks its
// descriptor table against what was planned: tensor count, name, dtype, and
// dims must match exactly. This is a header/descriptor-table check, not a
// full re-decode, so it stays cheap even on large models — its purpose is
// catching a GGUFWriter logic bug or an interrupted write immediately,
// instead of leaving the user to discover a broken file the next time they
// try to load it.
func verifyCompressedOutput(outPath string, planned []PlannedTensor) error {
	mmap, err := OpenMmap(outPath)
	if err != nil {
		return fmt.Errorf("compress: wrote %s but could not reopen it to verify: %w", outPath, err)
	}
	defer mmap.Close()
	got, err := ParseGGUF(mmap.Bytes())
	if err != nil {
		return fmt.Errorf("compress: wrote %s but it does not parse back as GGUF: %w", outPath, err)
	}
	if len(got.Tensors) != len(planned) {
		return fmt.Errorf("compress: wrote %s but it has %d tensors, planned %d", outPath, len(got.Tensors), len(planned))
	}
	for i, want := range planned {
		gotT := got.Tensors[i]
		if gotT.Name != want.Name || gotT.DType != want.DType {
			return fmt.Errorf("compress: wrote %s but tensor %d is %q/%s, planned %q/%s", outPath, i, gotT.Name, gotT.DType, want.Name, want.DType)
		}
		if gotT.Numel() != (TensorInfo{Dims: want.Dims}).Numel() {
			return fmt.Errorf("compress: wrote %s but tensor %q has %d elements, planned %d", outPath, want.Name, gotT.Numel(), (TensorInfo{Dims: want.Dims}).Numel())
		}
		size, ok := gotT.DType.DataSize(gotT.Numel())
		if !ok {
			return fmt.Errorf("compress: wrote %s but tensor %q has an unsizable type %s", outPath, want.Name, gotT.DType)
		}
		if got.DataOffset+int(gotT.Offset)+size > len(mmap.Bytes()) {
			return fmt.Errorf("compress: wrote %s but tensor %q's byte range runs past the end of the file", outPath, want.Name)
		}
	}
	return nil
}
