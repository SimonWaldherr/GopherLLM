package gopherllm

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
	// sparse-MoE expert banks. It is useful when the experts dominate model
	// size and normal routing selects only a small top-k subset.
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

// corePrefaultRanges returns all data ranges except sparse expert banks. The
// byte-size fallback deliberately mirrors loadWeight: unusual GGUF producers
// with ambiguous packing still get the offset-derived range inferred by the
// regular loader.
func corePrefaultRanges(data []byte, gguf *GGUFFile) (ranges []mmapByteRange, skippedExperts int) {
	if gguf == nil {
		return nil, 0
	}
	inferred := inferTensorSizes(data, gguf)
	for _, info := range gguf.Tensors {
		if isSparseExpertTensor(info.Name, info) {
			skippedExperts++
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
