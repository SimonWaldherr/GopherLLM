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
// There are two loading strategies, and which one runs is decided by
// LoadOptions.OutOfCore.
//
// The default MERGES. GopherLLM's tensor-loading path assumes one contiguous
// byte slice with every TensorInfo.Offset relative to one GGUFFile.DataOffset,
// so loadSplitRunner reconstructs an equivalent single-buffer view: it
// concatenates each shard's tensor-data region (from that shard's own
// DataOffset to its file end) back to back, rebasing every tensor's Offset by
// how many bytes were copied before it. The merged buffer is byte-for-byte
// what a single-file GGUF with these tensors would look like, at the cost of
// one full copy of the model's weights into anonymous memory.
//
// OutOfCore does NOT merge. That copy is exactly what makes the merging path
// unusable for the models people actually shard — a 400 GB checkpoint would
// need 400 GB of RAM before the first token. Instead each shard keeps its own
// mmap and every TensorInfo carries a view of the shard it lives in
// (TensorInfo.Shard), which model.go's tensorContainer resolves against. A
// tensor never spans shards, so no tensor-level stitching is needed, and the
// Runner takes ownership of all the mappings (Runner.extraMappedFiles).
// TestOutOfCoreSplitMatchesMergedSplit pins the two strategies to identical
// output.

import (
	"fmt"
)

// splitPathPrefix is the inverse of splitShardPath: it recognises
// "<prefix>-NNNNN-of-MMMMM.gguf" (5-digit, zero-padded, as produced by
// llama.cpp's gguf-split and llama_split_path) and returns the shared prefix
// that names the whole shard set. ok is false for anything else.
//
// Only the prefix is consumed by callers — the authoritative shard count comes
// from each shard's split.count metadata, never from the filename — but the two
// numeric fields are still validated, because that validation is the whole
// point of the call: it establishes that the file really was named by
// gguf-split, and therefore that its siblings can be found by feeding the
// prefix back through splitShardPath.
//
// The suffix is anchored to the end of the string and has a fixed length, so
// there is exactly one offset it can occupy and the prefix is unambiguously
// everything before it. (This used to be a regexp with a greedy (.*) prefix
// group, which for the same reason could only ever match at that one offset;
// hand-parsing it keeps regexp — ~458 KB of binary — out of the library's
// dependency closure for this, its single use.)
//
// One deliberate behaviour change from that regexp: '.' does not match '\n'
// without (?s), so a path containing a newline anywhere before the suffix used
// to be rejected. Newlines are legal in POSIX filenames and nothing downstream
// cares (the prefix is only ever concatenated back into sibling paths), so such
// a path is now accepted — the old rejection was an artefact of the regexp
// engine's defaults rather than a rule about how shards may be named.
func splitPathPrefix(path string) (prefix string, ok bool) {
	const suffixLen = len("-NNNNN-of-MMMMM.gguf")
	if len(path) < suffixLen {
		return "", false
	}
	s := path[len(path)-suffixLen:]
	// The extension test is case-sensitive, as the regexp was: gguf-split emits
	// lowercase, and a ".GGUF" sibling would not be found under the name we
	// reconstruct for it anyway.
	if s[0] != '-' || s[6:10] != "-of-" || s[15:] != ".gguf" {
		return "", false
	}
	if !splitFieldIsDigits(s[1:6]) || !splitFieldIsDigits(s[10:15]) {
		return "", false
	}
	return path[:len(path)-suffixLen], true
}

// splitFieldIsDigits stands in for the regexp's \d, which is ASCII-only, and is
// called with the two fixed-width numeric fields of a shard filename. It checks
// the bytes itself rather than round-tripping through strconv.Atoi, which
// accepts a leading '+' or '-' and would let "-0001" pass as a shard number.
func splitFieldIsDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// splitShardPath builds the filename for shard number (1-based) of count,
// given the shared prefix extracted by splitPathPrefix.
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
	// This function owns firstMmap and every shard mapping it opens. The
	// merging path closes all of them (it has copied the tensor bytes out
	// first); leaving one mapped would keep the file locked on Windows.
	//
	// The out-of-core path is the exception: its whole point is that the
	// weights stay as views of the live mappings, so on success it transfers
	// ownership to the Runner and sets keepMappings, and only the error paths
	// still unmap.
	type shard struct {
		gguf *GGUFFile
		mmap *MmapFile
		no   int
	}
	var shards []shard
	keepMappings := false
	defer func() {
		if keepMappings {
			return
		}
		for _, s := range shards {
			if s.mmap != nil && s.mmap != firstMmap {
				_ = s.mmap.Close()
			}
		}
		_ = firstMmap.Close()
	}()

	firstNo, count, _ := splitInfo(firstGGUF)

	prefix, isSplitFilename := splitPathPrefix(path)
	if !isSplitFilename {
		return nil, 0, fmt.Errorf("model declares split.count=%d but its filename %q does not match the "+
			"<prefix>-NNNNN-of-MMMMM.gguf split convention; rename it to match its sibling shards", count, path)
	}

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

	// Out-of-core: no concatenation at all. Each tensor keeps a view of its
	// own shard's mapping, so a 400 GB sharded model costs address space and
	// demand-paged residency rather than 400 GB of anonymous memory. This is
	// the only way out-of-core is useful in practice — every model large
	// enough to want it ships sharded.
	if options.OutOfCore {
		for _, s := range byNo {
			if !s.mmap.IsMapped() {
				return nil, 0, fmt.Errorf("out-of-core loading requires an OS memory map, but shard %d of %s fell back to an in-memory read", s.no+1, path)
			}
		}
		var shardedTensors []TensorInfo
		var metadata map[string]MetaValue
		var mappedBytes int64
		for _, s := range byNo {
			region := s.mmap.Bytes()[s.gguf.DataOffset:]
			mappedBytes += int64(len(region))
			for _, t := range s.gguf.Tensors {
				// Offset stays relative to this shard's tensor region; the
				// Shard view is what tensorContainer resolves against.
				t.Shard = region
				shardedTensors = append(shardedTensors, t)
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
		combined := &GGUFFile{Metadata: metadata, Tensors: shardedTensors, DataOffset: 0, Version: firstGGUF.Version}
		// borrowQuantized is safe and is the entire point: every borrowed
		// slice points into a live mapping the Runner now owns. Metal is
		// already rejected for out-of-core by validateLoadOptions.
		r, err := runnerFromParsedGGUF(nil, combined, true, options)
		if err != nil {
			return nil, 0, err
		}
		for _, s := range byNo {
			r.extraMappedFiles = append(r.extraMappedFiles, s.mmap)
		}
		keepMappings = true
		return r, mappedBytes, nil
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
