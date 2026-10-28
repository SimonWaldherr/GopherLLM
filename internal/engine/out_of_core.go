package engine

import (
	"fmt"
	"strings"
)

// MmapPrefaultMode controls how much of a mapped model is eagerly touched at
// load time. It affects physical page residency, not the immutable mmap view
// itself; the operating system remains free to evict pages later.
type MmapPrefaultMode uint8

const (
	// MmapPrefaultAll retains the established behavior: warm every page before
	// returning from load, making first-token latency predictable.
	MmapPrefaultAll MmapPrefaultMode = iota
	// MmapPrefaultCore warms dense/router/backbone tensors but skips known
	// sparse-MoE expert banks, Qwen's optional NextN draft layer, and untied
	// quantized input embeddings and Gemma 4's per-layer lookup tables.
	// These tensors remain demand-paged until inference needs their rows.
	MmapPrefaultCore
	// MmapPrefaultNone leaves every tensor demand-paged.
	MmapPrefaultNone
)

func (m MmapPrefaultMode) String() string {
	switch m {
	case MmapPrefaultAll:
		return "all"
	case MmapPrefaultCore:
		return "core"
	case MmapPrefaultNone:
		return "none"
	default:
		return "invalid"
	}
}

func (m MmapPrefaultMode) valid() bool {
	return m == MmapPrefaultAll || m == MmapPrefaultCore || m == MmapPrefaultNone
}

func validateLoadOptions(options LoadOptions) error {
	if !options.Prefault.valid() {
		return fmt.Errorf("invalid mmap prefault mode: %d", options.Prefault)
	}
	if options.OutOfCore && options.UseMetal {
		return fmt.Errorf("out-of-core loading is CPU-only; disable Metal")
	}
	if options.OutOfCore && options.PrepareQuantized {
		return fmt.Errorf("out-of-core loading cannot prepare quantized weights; disable prepared quantization")
	}
	return nil
}

func effectivePrefaultMode(options LoadOptions) MmapPrefaultMode {
	if options.OutOfCore && options.Prefault == MmapPrefaultAll {
		// The zero-value remains backwards compatible for ordinary loads. In
		// OOC mode it becomes core-only, avoiding an intentional read of every
		// sparse expert before the first request.
		return MmapPrefaultCore
	}
	return options.Prefault
}

func isSparseExpertTensor(name string, info TensorInfo) bool {
	if len(info.Dims) != 3 {
		return false
	}
	return strings.HasSuffix(name, ".ffn_gate_exps.weight") ||
		strings.HasSuffix(name, ".ffn_up_exps.weight") ||
		strings.HasSuffix(name, ".ffn_gate_up_exps.weight") ||
		strings.HasSuffix(name, ".ffn_down_exps.weight")
}

// qwen35NextNPrefaultPrefix recognizes the single trailing NextN layer that
// LoadQwen35Model excludes from the target decoder. Do not infer optional
// layers from unrelated architectures or an incomplete metadata declaration.
func qwen35NextNPrefaultPrefix(gguf *GGUFFile) string {
	arch, _ := gguf.GetString("general.architecture")
	if arch != "qwen35" && arch != "qwen35moe" {
		return ""
	}
	layers := gguf.GetU32(arch+".block_count", 0)
	nextN := gguf.GetU32(arch+".nextn_predict_layers", 0)
	if layers <= 1 || nextN > 1 {
		return ""
	}
	prefix := fmt.Sprintf("blk.%d.", layers-1)
	marker, previousMarker := false, false
	previous := fmt.Sprintf("blk.%d.nextn.eh_proj.weight", layers-2)
	for _, tensor := range gguf.Tensors {
		marker = marker || tensor.Name == prefix+"nextn.eh_proj.weight"
		previousMarker = previousMarker || tensor.Name == previous
	}
	if !marker || (nextN == 0 && previousMarker) {
		return ""
	}
	return prefix
}

// An untied packed input embedding is only read one token row at a time.
// Warming its entire vocabulary bank can evict decoder weights that every
// generated token needs. Scalar embeddings are expanded by the normal loader,
// and tied embeddings double as the output projection, so keep those warm.
func hasRowOnlyPackedEmbedding(gguf *GGUFFile) bool {
	separateOutput, packedEmbedding := false, false
	for _, tensor := range gguf.Tensors {
		if tensor.Name == "output.weight" {
			separateOutput = true
		}
		if tensor.Name == "token_embd.weight" && len(tensor.Dims) == 2 && tensor.Dims[0] > 0 && tensor.Dims[1] > 0 {
			block := tensor.DType.BlockSize()
			bytes, known := tensor.DType.DataSize(tensor.Numel())
			packedEmbedding = known && bytes > 0 && block > 1 && tensor.Dims[0]%uint64(block) == 0
		}
	}
	return separateOutput && packedEmbedding
}

// corePrefaultRanges returns all data ranges except sparse expert banks and
// Qwen's optional NextN layer, untied packed inputs and Gemma's PLE tables. These weights
// need not displace target-decoder pages during ordinary startup. The
// byte-size fallback deliberately mirrors loadWeight: unusual GGUF producers
// with ambiguous packing still get the offset-derived range inferred by the
// regular loader.
func corePrefaultRanges(data []byte, gguf *GGUFFile) (ranges []mmapByteRange, skippedExperts int) {
	if gguf == nil {
		return nil, 0
	}
	inferred := inferTensorSizes(data, gguf)
	nextNPrefix := qwen35NextNPrefaultPrefix(gguf)
	rowEmbedding := hasRowOnlyPackedEmbedding(gguf)
	arch, _ := gguf.GetString("general.architecture")
	gemmaLookups := arch == "gemma4"
	for _, info := range gguf.Tensors {
		if isSparseExpertTensor(info.Name, info) {
			skippedExperts++
			continue
		}
		if nextNPrefix != "" && strings.HasPrefix(info.Name, nextNPrefix) {
			continue
		}
		if rowEmbedding && info.Name == "token_embd.weight" {
			continue
		}
		if gemmaLookups && info.Name == "per_layer_token_embd.weight" {
			continue
		}
		bytes, ok := info.DType.DataSize(info.Numel())
		if !ok || bytes <= 0 {
			bytes = inferred[info.Name]
		}
		if bytes <= 0 || info.Offset > uint64(len(data)) {
			continue
		}
		start64 := uint64(gguf.DataOffset) + info.Offset
		if start64 > uint64(len(data)) {
			continue
		}
		if inferredBytes := inferred[info.Name]; inferredBytes > 0 && (bytes == 0 || start64+uint64(bytes) > uint64(len(data))) {
			bytes = inferredBytes
		}
		end64 := start64 + uint64(bytes)
		if end64 > uint64(len(data)) {
			end64 = uint64(len(data))
		}
		if end64 > start64 {
			ranges = append(ranges, mmapByteRange{start: int(start64), end: int(end64)})
		}
	}
	return normalizeMmapRanges(len(data), ranges), skippedExperts
}

func prefaultMappedModel(data []byte, gguf *GGUFFile, options LoadOptions) {
	switch effectivePrefaultMode(options) {
	case MmapPrefaultAll:
		prefaultPages(data)
	case MmapPrefaultCore:
		ranges, skippedExperts := corePrefaultRanges(data, gguf)
		// A dense model has no cold expert bank. In out-of-core mode, touching
		// every dense range would defeat its memory goal, so leave it lazy.
		if options.OutOfCore && skippedExperts == 0 {
			return
		}
		prefaultRanges(data, ranges)
	case MmapPrefaultNone:
		return
	}
}
