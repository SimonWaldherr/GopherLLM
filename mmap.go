package gopherllm

import "github.com/SimonWaldherr/GopherLLM/internal/mmapfile"

// MmapFile exposes a model file as one immutable byte slice, memory-mapped
// where the platform allows so multi-gigabyte weights are paged in on demand
// rather than copied. It must stay open for the Runner's lifetime.
type MmapFile = mmapfile.File

// OpenMmap opens path through the platform's read-only file-mapping backend.
// If mapping is unavailable for a specific file, it falls back to a read copy.
func OpenMmap(path string) (*MmapFile, error) {
	return mmapfile.Open(path)
}

type mmapByteRange struct {
	start int
	end   int
}

func prefaultPages(data []byte) {
	mmapfile.PrefaultPages(data, numThreads())
}

func prefaultRanges(data []byte, ranges []mmapByteRange) {
	mmapfile.PrefaultRanges(data, toInternalMmapRanges(ranges), numThreads())
}

func normalizeMmapRanges(dataLen int, ranges []mmapByteRange) []mmapByteRange {
	normalized := mmapfile.NormalizeRanges(dataLen, toInternalMmapRanges(ranges))
	out := make([]mmapByteRange, len(normalized))
	for i, r := range normalized {
		out[i] = mmapByteRange{start: r.Start, end: r.End}
	}
	return out
}

func toInternalMmapRanges(ranges []mmapByteRange) []mmapfile.ByteRange {
	out := make([]mmapfile.ByteRange, len(ranges))
	for i, r := range ranges {
		out[i] = mmapfile.ByteRange{Start: r.start, End: r.end}
	}
	return out
}
