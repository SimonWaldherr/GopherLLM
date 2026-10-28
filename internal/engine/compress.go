package engine

// GGUF post-training compression with round-to-nearest row encoders and
// optional mixed-precision tensor presets. Source weights remain mapped while
// bounded row chunks are decoded, quantized, and streamed to a new container.

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"
)

// tensorToF32 fully decodes raw as numel float32 values, regardless of
// whether dtype is a plain float format or a quantized one. dequantTensor
// (model_load.go) only covers the quantized formats — F32/F16/F64/BF16 aren't
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
// that aren't valid compression targets here). This legacy parser returns
// only a base type; use ParseCompressOptions to retain mixed preset policies.
// Q3_K and Q2_K are opt-in: they reduce the memory bandwidth consumed per generated token, at a
// progressively larger model-quality cost.
func ParseCompressFormat(name string) (GGMLType, bool) {
	switch name {
	case "Q8_0", "q8_0":
		return GGMLTypeQ8_0, true
	case "Q4_0", "q4_0":
		return GGMLTypeQ4_0, true
	case "Q2_K", "q2_k":
		return GGMLTypeQ2_K, true
	case "Q3_K", "q3_k":
		return GGMLTypeQ3_K, true
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
	case GGMLTypeQ2_K:
		return 0
	case GGMLTypeQ3_K:
		return 1
	case GGMLTypeQ4_0:
		return 2
	case GGMLTypeQ4_K:
		return 3
	case GGMLTypeQ5_K:
		return 4
	case GGMLTypeQ6_K:
		return 5
	case GGMLTypeQ8_0:
		return 6
	default:
		return -1
	}
}

// outputFloorFormat is the legacy single-target policy's quality floor:
// token embeddings and output heads keep Q6_K when the base target is lower.
// Mixed presets use their own policy, including separate input embeddings.
const outputFloorFormat = GGMLTypeQ6_K

var namedOutputTensors = map[string]bool{
	"token_embd.weight": true,
	"output.weight":     true,
}

// CompressOptions configures CompressModel.
type CompressOptions struct {
	// TargetFormat is the GGML type every eligible weight tensor is
	// requantized to. Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K are
	// supported (the formats quantize_rtn.go has encoders for). Q2_K/Q3_K
	// trade output quality for less memory traffic during decode.
	TargetFormat GGMLType
	// Preset selects a mixed tensor policy, such as Q4_K_M. TargetFormat may
	// be omitted or set to the preset's base type. Empty preserves the legacy
	// single-target policy.
	Preset string
	// Uniform disables the output/embedding quality floor (outputFloorFormat)
	// for eligible matrices. Default false
	// preserves the legacy quality floor. Mixed presets reject Uniform;
	// use their base TargetFormat for strictly uniform quantization.
	Uniform bool
	// Context cancels conversion between bounded chunks. Nil means Background.
	Context context.Context
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
// has no encoder. This includes shape-fallback formats that are not public
// compression targets.
func quantizeRowFor(dtype GGMLType) func(row []float32, cols int) []byte {
	switch dtype {
	case GGMLTypeF16:
		return quantizeRowF16
	case GGMLTypeQ8_0:
		return QuantizeRowQ8_0
	case GGMLTypeQ4_0:
		return QuantizeRowQ4_0
	case GGMLTypeQ5_0:
		return QuantizeRowQ5_0
	case GGMLTypeQ5_1:
		return QuantizeRowQ5_1
	case GGMLTypeQ2_K:
		return QuantizeRowQ2K
	case GGMLTypeQ3_K:
		return QuantizeRowQ3K
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

// CompressModel writes a requantized, single-file GGUF. Rows are converted in
// bounded chunks, including 3-D expert banks. Preset selects a mixed tensor
// policy; without it the existing TargetFormat/output-floor policy applies.
//
// The source remains mapped and must differ from outPath, including through
// symlinks or hardlinks. The destination is replaced only after a sibling
// temporary file has been flushed, closed, and verified. Cancellation or a
// conversion error preserves any existing destination.
func CompressModel(sourcePath, outPath string, opts CompressOptions) error {
	opts, err := resolveCompressOptions(opts)
	if err != nil {
		return err
	}
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if sameFile(sourcePath, outPath) {
		return fmt.Errorf("compress: output must not be the same file as the source model (%s)", sourcePath)
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
	raw := mmap.Bytes()
	src, err := ParseGGUF(raw)
	if err != nil {
		return fmt.Errorf("compress: parse %s: %w", sourcePath, err)
	}
	if shardNo, shardCount, ok := splitInfo(src); ok {
		return fmt.Errorf("compress: %s is shard %d/%d of a split GGUF; merge the shards before quantizing", sourcePath, shardNo+1, shardCount)
	}
	planner := newCompressionPlanner(src, opts)
	planned := make([]PlannedTensor, len(src.Tensors))
	requantize := make([]bool, len(src.Tensors))
	sourceSizes := make([]int, len(src.Tensors))
	var beforeBytes, afterBytes int64
	for i, t := range src.Tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		before, err := compressionTensorSize(t)
		if err != nil {
			return err
		}
		if !compressionRangeValid(raw, src.DataOffset, t.Offset, before) {
			return fmt.Errorf("compress: tensor %q: byte range runs past the end of the source file", t.Name)
		}
		outType, doRequant := planner.plan(t)
		after, err := compressionTensorSize(TensorInfo{Name: t.Name, Dims: t.Dims, DType: outType})
		if err != nil {
			return err
		}
		planned[i] = PlannedTensor{Name: t.Name, Dims: t.Dims, DType: outType}
		requantize[i], sourceSizes[i] = doRequant, before
		beforeBytes += int64(before)
		afterBytes += int64(after)
		if doRequant {
			fmt.Fprintf(logw, "requantize %-40s %s -> %s (%d bytes -> %d bytes)\n", t.Name, t.DType, outType, before, after)
		} else {
			fmt.Fprintf(logw, "passthrough %-40s %s\n", t.Name, t.DType)
		}
	}
	if beforeBytes > 0 {
		fmt.Fprintf(logw, "total: %d bytes -> %d bytes (%.1f%%)\n", beforeBytes, afterBytes, 100*float64(afterBytes)/float64(beforeBytes))
	}

	metadata := compressionMetadata(src.Metadata, opts)
	out, err := os.CreateTemp(filepath.Dir(outPath), "."+filepath.Base(outPath)+".*.tmp")
	if err != nil {
		return fmt.Errorf("compress: create temporary output for %s: %w", outPath, err)
	}
	tempPath := out.Name()
	defer func() {
		out.Close()
		os.Remove(tempPath)
	}()
	alignment := int(src.GetU32("general.alignment", 32))
	gw, err := NewGGUFWriter(out, metadata, planned, alignment)
	if err != nil {
		return fmt.Errorf("compress: write header: %w", err)
	}
	lastProgress := time.Now()
	for i, t := range src.Tensors {
		if err := ctx.Err(); err != nil {
			return err
		}
		start := src.DataOffset + int(t.Offset)
		data := raw[start : start+sourceSizes[i]]
		err := gw.WriteTensorStream(func(dst io.Writer) error {
			if requantize[i] {
				return writeCompressedTensor(ctx, dst, t, data, planned[i].DType)
			}
			for len(data) > 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
				n := min(len(data), 1<<20)
				written, err := dst.Write(data[:n])
				if err != nil {
					return err
				}
				if written != n {
					return io.ErrShortWrite
				}
				data = data[n:]
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("compress: tensor %q: %w", t.Name, err)
		}
		if now := time.Now(); now.Sub(lastProgress) > 3*time.Second {
			lastProgress = now
			fmt.Fprintf(logw, "... %d/%d tensors (%s)\n", i+1, len(src.Tensors), t.Name)
		}
	}
	if err := gw.Close(); err != nil {
		return fmt.Errorf("compress: flush output: %w", err)
	}
	if err := out.Sync(); err != nil {
		return fmt.Errorf("compress: sync output: %w", err)
	}
	if err := out.Close(); err != nil {
		return fmt.Errorf("compress: close output: %w", err)
	}
	if err := verifyCompressedOutput(tempPath, planned); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := os.Rename(tempPath, outPath); err != nil {
		return fmt.Errorf("compress: install %s: %w", outPath, err)
	}
	return nil
}

// compressionTensorSize checks dimensions before using unchecked Numel/DataSize
// arithmetic. Sizing one block first avoids overflow in DataSize's products.
func compressionTensorSize(t TensorInfo) (int, error) {
	maxInt := int(^uint(0) >> 1)
	numel := 1
	if len(t.Dims) == 0 {
		return 0, fmt.Errorf("compress: tensor %q: missing dimensions", t.Name)
	}
	for _, d := range t.Dims {
		if d == 0 || d > uint64(maxInt/numel) {
			return 0, fmt.Errorf("compress: tensor %q: invalid or overflowing dimensions", t.Name)
		}
		numel *= int(d)
	}
	block := t.DType.BlockSize()
	bytes, ok := t.DType.DataSize(block)
	if !ok || bytes <= 0 {
		return 0, fmt.Errorf("compress: tensor %q: unsupported type %s", t.Name, t.DType)
	}
	if t.Dims[0]%uint64(block) != 0 || numel/block > maxInt/bytes {
		return 0, fmt.Errorf("compress: tensor %q: invalid row length or overflowing byte size for %s", t.Name, t.DType)
	}
	return numel / block * bytes, nil
}

func compressionRangeValid(raw []byte, dataOffset int, offset uint64, size int) bool {
	if dataOffset < 0 || dataOffset > len(raw) || size < 0 {
		return false
	}
	remaining := len(raw) - dataOffset
	return offset <= uint64(remaining) && size <= remaining-int(offset)
}

func compressionMetadata(source map[string]MetaValue, opts CompressOptions) map[string]MetaValue {
	metadata := make(map[string]MetaValue, len(source)+3)
	for k, v := range source {
		metadata[k] = v
	}
	for _, key := range splitKeys {
		delete(metadata, key)
	}
	// These are llama_ftype values, not GGML tensor type IDs.
	fileTypes := map[GGMLType]uint32{
		GGMLTypeQ4_0: 2, GGMLTypeQ8_0: 7, GGMLTypeQ2_K: 10,
		GGMLTypeQ3_K: 11, GGMLTypeQ4_K: 14, GGMLTypeQ5_K: 16, GGMLTypeQ6_K: 18,
	}
	fileType := fileTypes[opts.TargetFormat]
	if opts.Preset == "Q4_K_M" {
		fileType = 15
	} else if opts.Preset == "Q5_K_M" {
		fileType = 17
	}
	metadata["general.file_type"] = MetaValue{Kind: "u32", Value: fileType}
	metadata["general.quantization_version"] = MetaValue{Kind: "u32", Value: uint32(2)}
	metadata["gopherllm.quantization.method"] = MetaValue{Kind: "str", Value: "rtn"}
	return metadata
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
		if !slices.Equal(gotT.Dims, want.Dims) {
			return fmt.Errorf("compress: wrote %s but tensor %q has dimensions %v, planned %v", outPath, want.Name, gotT.Dims, want.Dims)
		}
		size, err := compressionTensorSize(gotT)
		if err != nil {
			return err
		}
		if !compressionRangeValid(mmap.Bytes(), got.DataOffset, gotT.Offset, size) {
			return fmt.Errorf("compress: wrote %s but tensor %q's byte range runs past the end of the file", outPath, want.Name)
		}
	}
	return nil
}
