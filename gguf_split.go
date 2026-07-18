package gopherllm

// Split (sharded) GGUF support: llama.cpp's gguf-split tool publishes large
// models (commonly 70B+ downloads) as multiple files named
// "<prefix>-00001-of-00005.gguf", "<prefix>-00002-of-00005.gguf", etc. (1-based
// index in the filename). Every shard carries "split.no" (its own 0-based
// index) and "split.count" (total shard count) metadata; by convention the
// shard with split.no == 0 also carries the model's full metadata (arch,
// tokenizer, rope, ...), while the others carry only their own tensor data
// plus the split keys.
//
// GopherLLM's tensor-loading path (model.go's loadWeight/inferTensorSizes)
// assumes one contiguous byte slice with every TensorInfo.Offset relative to
// one GGUFFile.DataOffset. Rather than thread a multi-file tensor-source
// abstraction through that hot path, loadSplitRunner reconstructs an
// equivalent single-buffer view: it concatenates each shard's tensor-data
// region (from that shard's own DataOffset to its file end) back to back,
// rebasing every tensor's Offset by how many bytes were copied before it.
// This costs one full copy of the model's weights at load time (true
// zero-copy mmap borrowing is only available for single-file GGUFs) but
// requires no changes to weight-loading logic — the merged buffer is
// byte-for-byte what a single-file GGUF with these tensors would look like.

import (
	"fmt"
	"regexp"
)

// splitFilePattern matches "<prefix>-NNNNN-of-MMMMM.gguf" (5-digit,
// zero-padded, as produced by llama.cpp's gguf-split and llama_split_path).
var splitFilePattern = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// splitShardPath builds the filename for shard number (1-based) of count,
// given the shared prefix extracted by splitFilePattern.
func splitShardPath(prefix string, index, count int) string {
	return fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, index, count)
}

// splitInfo reports a parsed GGUF's split.no/split.count metadata. ok is
// false when the file is not part of a split (split.count absent or <= 1).
func splitInfo(gguf *GGUFFile) (no, count int, ok bool) {
	c := gguf.GetU32("split.count", 1)
	if c <= 1 {
		return 0, 0, false
	}
	n := gguf.GetU32("split.no", 0)
	return int(n), int(c), true
}

// splitKeys are stripped from the merged metadata: they describe the shard
// layout, which no longer exists once the shards are merged into one buffer.
var splitKeys = []string{"split.no", "split.count", "split.tensors.count"}

// loadSplitRunner merges the shard set that path belongs to (detected via
// firstGGUF's split.no/split.count metadata) into one synthetic in-memory
// GGUF and loads it exactly like a single-file model. firstMmap is the
// caller's already-open mapping of path; it is reused for whichever shard
// path turns out to be (avoiding opening it twice) and closed, along with
// every other shard's mapping, once their tensor bytes are copied out.
func loadSplitRunner(path string, firstGGUF *GGUFFile, firstMmap *MmapFile, options LoadOptions) (*Runner, int64, error) {
	// This function owns firstMmap and every shard mapping it opens: all of
	// them are closed on every path (success copies the tensor bytes into
	// the merged buffer first). Leaving one mapped would keep the file
	// locked on Windows.
	type shard struct {
		gguf *GGUFFile
		mmap *MmapFile
		no   int
	}
	var shards []shard
	defer func() {
		for _, s := range shards {
			if s.mmap != nil && s.mmap != firstMmap {
				_ = s.mmap.Close()
			}
		}
		_ = firstMmap.Close()
	}()

	firstNo, count, _ := splitInfo(firstGGUF)

	m := splitFilePattern.FindStringSubmatch(path)
	if m == nil {
		return nil, 0, fmt.Errorf("model declares split.count=%d but its filename %q does not match the "+
			"<prefix>-NNNNN-of-MMMMM.gguf split convention; rename it to match its sibling shards", count, path)
	}
	prefix := m[1]

	shards = make([]shard, count)
	for i := 1; i <= count; i++ {
		shardPath := splitShardPath(prefix, i, count)
		var mm *MmapFile
		if i-1 == firstNo && shardPath == path {
			mm = firstMmap
		} else {
			var err error
			mm, err = OpenMmap(shardPath)
			if err != nil {
				return nil, 0, fmt.Errorf("split model: failed to open shard %d/%d (%s): %w", i, count, shardPath, err)
			}
		}
		shards[i-1].mmap = mm
		g, err := ParseGGUFQuiet(mm.Bytes())
		if err != nil {
			return nil, 0, fmt.Errorf("split model: failed to parse shard %d/%d (%s): %w", i, count, shardPath, err)
		}
		no, shardCount, ok := splitInfo(g)
		if !ok || shardCount != count {
			no = i - 1 // tolerate a shard missing/mismatching its own split metadata; filename order still wins
		}
		shards[i-1] = shard{gguf: g, mmap: mm, no: no}
	}

	// Order shards by their declared split.no (0-based) rather than trusting
	// filename order alone, in case a producer numbered them unconventionally.
	byNo := make([]shard, count)
	for _, s := range shards {
		if s.no < 0 || s.no >= count {
			return nil, 0, fmt.Errorf("split model: shard %s declares split.no=%d out of range [0,%d)", path, s.no, count)
		}
		byNo[s.no] = s
	}

	totalBytes := 0
	for _, s := range byNo {
		totalBytes += len(s.mmap.Bytes()) - s.gguf.DataOffset
	}
	merged := make([]byte, 0, totalBytes)
	mergedTensors := make([]TensorInfo, 0)
	var metadata map[string]MetaValue
	for _, s := range byNo {
		base := uint64(len(merged))
		tensorRegion := s.gguf.DataOffset
		merged = append(merged, s.mmap.Bytes()[tensorRegion:]...)
		for _, t := range s.gguf.Tensors {
			t.Offset += base
			mergedTensors = append(mergedTensors, t)
		}
		if s.no == 0 {
			metadata = make(map[string]MetaValue, len(s.gguf.Metadata))
			for k, v := range s.gguf.Metadata {
				metadata[k] = v
			}
		}
	}
	if metadata == nil {
		return nil, 0, fmt.Errorf("split model %s: no shard declares split.no=0 (no shard carries full model metadata)", path)
	}
	for _, k := range splitKeys {
		delete(metadata, k)
	}

	combined := &GGUFFile{Metadata: metadata, Tensors: mergedTensors, DataOffset: 0, Version: firstGGUF.Version}
	// The merged buffer is a fresh Go allocation the returned Runner's
	// weights keep alive by reference, so borrowing from it is safe — except
	// under Metal, which must never retain a C pointer into Go heap memory
	// (only real OS mappings qualify for bytesNoCopy).
	r, err := runnerFromParsedGGUF(merged, combined, !options.UseMetal, options)
	if err != nil {
		return nil, 0, err
	}
	return r, int64(len(merged)), nil
}
