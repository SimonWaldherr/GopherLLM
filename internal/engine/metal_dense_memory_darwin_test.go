//go:build darwin && cgo && metal

package engine

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func metalDenseMemoryMatrices(l *LayerWeights) []*Weight {
	return []*Weight{&l.WQ, &l.WK, &l.WV, &l.WO, &l.W1, &l.W3, &l.W2}
}

func metalDenseMemoryMappedRunner(t *testing.T, c Config, w ModelWeights) *Runner {
	t.Helper()
	w.Layers = append([]LayerWeights(nil), w.Layers...)
	type matrixRange struct{ start, end int }
	var ranges []matrixRange
	var data []byte
	page := os.Getpagesize()
	for i := range w.Layers {
		for _, matrix := range metalDenseMemoryMatrices(&w.Layers[i]) {
			// Real GGUF tensor starts need not be page aligned. Include a small
			// offset so the backend's rounded mapping view is exercised too.
			start := (len(data)+page-1)/page*page + 32
			data = append(data, make([]byte, start-len(data))...)
			data = append(data, matrix.Raw...)
			ranges = append(ranges, matrixRange{start, len(data)})
		}
	}
	data = append(data, make([]byte, (page-len(data)%page)%page)...)
	path := filepath.Join(t.TempDir(), "dense-weights.bin")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	mapping, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mapping.Close() })
	if !mapping.IsMapped() {
		t.Skip("requires an OS-backed mapping")
	}
	n := 0
	for i := range w.Layers {
		for _, matrix := range metalDenseMemoryMatrices(&w.Layers[i]) {
			span := ranges[n]
			matrix.Raw = mapping.Bytes()[span.start:span.end:span.end]
			n++
		}
	}
	r := &Runner{kind: loadedStandard, arch: c.Arch, config: c, standard: w, mappedFile: mapping, metalMapping: mapping.Bytes()}
	t.Cleanup(func() { _ = r.Close() })
	return r
}

func metalDenseMemoryExact(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s lengths: %d vs %d", label, len(got), len(want))
	}
	for i := range got {
		if !finite32(got[i]) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: %g vs %g", label, i, got[i], want[i])
		}
	}
}

func TestMetalDenseMappedWeightsAvoidCopies(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	old := metalDenseDecodeEnabled
	metalDenseDecodeEnabled = true
	t.Cleanup(func() { metalDenseDecodeEnabled = old })
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	t.Setenv("GOPHERLLM_METAL_DENSE_PREFILL", "1")
	for _, shared := range []string{"0", "1"} {
		for _, source := range []string{"mapped", "heap-copy", "heap-padded"} {
			t.Run(shared+"/"+source, func(t *testing.T) {
				t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
				c, _, gpu := metalPrefillFixture(t, "ministral")
				_, _, owned := metalPrefillFixture(t, "ministral")
				r := metalDenseMemoryMappedRunner(t, c, gpu)
				ref := &Runner{kind: loadedStandard, arch: c.Arch, config: c, standard: owned}
				t.Cleanup(func() { _ = ref.Close() })
				var expectedBorrowed, expectedCopied int64
				for i := range r.standard.Layers {
					for _, matrix := range metalDenseMemoryMatrices(&r.standard.Layers[i]) {
						if matrix.Metal == nil {
							expectedBorrowed += int64(len(matrix.Raw))
						}
					}
				}
				if source != "mapped" {
					matrix := &r.standard.Layers[0].WK
					expectedCopied = int64(len(matrix.Raw))
					expectedBorrowed -= expectedCopied
					matrix.Raw = append([]byte(nil), matrix.Raw...)
					if source == "heap-padded" {
						matrix.Raw = append(matrix.Raw, make([]byte, 64)...)
					}
					if weightBytesInMapping(matrix.Raw, r.metalMapping) {
						t.Fatal("Go-owned matrix was mistaken for a mapped view")
					}
				}
				cache, buf := r.generationWorkspace(40)
				refCache, refBuf := ref.generationWorkspace(40)
				if !weightBytesInMapping(buf.metalMapping, r.mappedFile.Bytes()) || refBuf.metalMapping != nil {
					t.Fatal("workspace lost the mapping ownership distinction")
				}
				var got, want []float32
				tokens := metalPrefillTokens(17)
				ForwardBatchInto(c, r.standard, cache, buf, tokens, 0, true, &got)
				ForwardBatchInto(c, ref.standard, refCache, refBuf, tokens, 0, true, &want)
				state, refState := buf.metalDense, refBuf.metalDense
				if state == nil || refState == nil || state.decoder.PrefillScratchBytes() == 0 {
					t.Fatal("whole Metal prefill did not initialize the dense decoder")
				}
				if expectedBorrowed == 0 || state.borrowedWeightBytes != expectedBorrowed || state.copiedWeightBytes != expectedCopied {
					t.Fatalf("weight bytes: borrowed=%d copied=%d; want %d/%d", state.borrowedWeightBytes, state.copiedWeightBytes, expectedBorrowed, expectedCopied)
				}
				if refState.borrowedWeightBytes != 0 || refState.copiedWeightBytes != expectedBorrowed+expectedCopied {
					t.Fatalf("owned reference weight bytes: borrowed=%d copied=%d", refState.borrowedWeightBytes, refState.copiedWeightBytes)
				}
				check := func(end int) {
					metalDenseMemoryExact(t, "logits", got, want)
					metalDenseMemoryExact(t, "hidden", buf.XN, refBuf.XN)
					for i := range cache.K {
						metalDenseMemoryExact(t, "K", cache.K[i][:end*cache.PerPosKDim], refCache.K[i][:end*refCache.PerPosKDim])
						metalDenseMemoryExact(t, "V", cache.V[i][:end*cache.PerPosVDim], refCache.V[i][:end*refCache.PerPosVDim])
					}
				}
				check(len(tokens))
				for pos := len(tokens); pos < len(tokens)+3; pos++ {
					token := uint32(pos % c.VocabSize)
					if !tryMetalDenseDecodeOutput(c, r.standard, cache, buf, token, pos, &got) ||
						!tryMetalDenseDecodeOutput(c, ref.standard, refCache, refBuf, token, pos, &want) {
						t.Fatal("Metal decode failed:", MetalError())
					}
					check(pos + 1)
				}
				mapping := r.mappedFile
				if err := r.Close(); err != nil {
					t.Fatal(err)
				}
				if state.decoder != nil || state.owned != nil || state.refs != nil || mapping.Len() != 0 || r.metalMapping != nil || r.workspaceBuf != nil {
					t.Fatal("Close retained a decoder, GPU weight, or file mapping")
				}
			})
		}
	}
}

func TestMetalWarmupMappingProvenance(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	t.Setenv("GOPHERLLM_NO_METAL_WARMUP", "0")
	data := buildTinyQuantizedMistralGGUF()
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultNone})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	if !r.IsMapped() {
		t.Skip("requires an OS-backed mapping")
	}
	// warmup runs inside construction, before RunnerFromPath assigns
	// mappedFile. Its workspace must already know the actual mapping range.
	if r.workspaceBuf == nil || !weightBytesInMapping(r.metalMapping, r.mappedFile.Bytes()) ||
		!weightBytesInMapping(r.workspaceBuf.metalMapping, r.mappedFile.Bytes()) {
		t.Fatal("load-time warmup did not receive the mapped weight range")
	}
	byteRunner, err := RunnerFromGGUFBytesWithOptions(data, LoadOptions{UseMetal: true, BorrowQuantized: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = byteRunner.Close() })
	if byteRunner.metalMapping != nil || byteRunner.workspaceBuf == nil || byteRunner.workspaceBuf.metalMapping != nil {
		t.Fatal("byte-backed Metal runner exposed Go memory as an OS mapping")
	}
}
