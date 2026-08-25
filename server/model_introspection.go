package server

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// analyzeModelFile parses a GGUF file's header only (mmap'd, no weight
// bytes touched — the same cheap path gopherllm.DiscoverModels uses) into an gopherllm.Analysis,
// for building Ollama-shaped model metadata without loading a full gopherllm.Runner.
func analyzeModelFile(path string) (*gopherllm.Analysis, error) {
	mmap, err := gopherllm.OpenMmap(path)
	if err != nil {
		return nil, err
	}
	defer mmap.Close()
	gguf, err := gopherllm.ParseGGUFQuiet(mmap.Bytes())
	if err != nil {
		return nil, err
	}
	return gopherllm.AnalyzeGGUF(gguf, nil), nil
}

// resolveModelAnalysis answers /api/show's "which model": empty/matching
// name means the currently loaded gopherllm.Runner (full gopherllm.Analysis, tokenizer
// included); any other name is looked up in ModelDir (if configured) and
// header-analyzed on demand.
func resolveModelAnalysis(state *runnerState, modelDir, requested string) (*gopherllm.Analysis, bool) {
	r := state.get()
	if r != nil && (requested == "" || requested == modelID(r)) {
		return gopherllm.AnalyzeGGUF(r.GGUF(), r.Tokenizer()), true
	}
	if modelDir == "" {
		return nil, false
	}
	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
	if err != nil {
		return nil, false
	}
	entry, err := gopherllm.SelectModel(entries, requested)
	if err != nil {
		return nil, false
	}
	a, err := analyzeModelFile(entry.Path)
	if err != nil {
		return nil, false
	}
	return a, true
}

// archGraphForFile parses a GGUF file's header only (mmap'd, no weight bytes
// touched) into a gopherllm.ArchGraph, mirroring analyzeModelFile but for the
// architecture viewer.
func archGraphForFile(path string) (*gopherllm.ArchGraph, error) {
	mmap, err := gopherllm.OpenMmap(path)
	if err != nil {
		return nil, err
	}
	defer mmap.Close()
	gguf, err := gopherllm.ParseGGUFQuiet(mmap.Bytes())
	if err != nil {
		return nil, err
	}
	tok, _ := gopherllm.TokenizerFromMetadata(gguf.Metadata)
	cfg := gopherllm.ConfigFromGGUF(gguf)
	a := gopherllm.AnalyzeGGUF(gguf, tok)
	return gopherllm.BuildArchGraph(gguf, cfg, a, nil), nil
}

// resolveArchGraph answers the architecture viewer's "which model": empty or
// matching name means the currently loaded gopherllm.Runner (vision encoder
// included, if paired); any other name is looked up in ModelDir (if
// configured) and header-analyzed on demand, same resolution rule as
// resolveModelAnalysis.
func resolveArchGraph(state *runnerState, modelDir, requested string) (*gopherllm.ArchGraph, bool) {
	r := state.get()
	if r != nil && (requested == "" || requested == modelID(r)) {
		return gopherllm.RunnerArchGraph(r), true
	}
	if modelDir == "" {
		return nil, false
	}
	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
	if err != nil {
		return nil, false
	}
	entry, err := gopherllm.SelectModel(entries, requested)
	if err != nil {
		return nil, false
	}
	g, err := archGraphForFile(entry.Path)
	if err != nil {
		return nil, false
	}
	return g, true
}

// ollamaTagEntries builds /api/tags' model list: every entry under ModelDir
// (header-analyzed, cheap) if configured, else just the currently loaded
// model.
func ollamaTagEntries(state *runnerState, modelDir string) []map[string]any {
	if modelDir == "" {
		r := state.get()
		if r == nil {
			return []map[string]any{}
		}
		name := modelID(r)
		a := gopherllm.AnalyzeGGUF(r.GGUF(), r.Tokenizer())
		return []map[string]any{ollamaTagEntry(name, state.getPath(), a.FileBytes, a)}
	}
	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
	if err != nil {
		return []map[string]any{}
	}
	tags := make([]map[string]any, 0, len(entries))
	for _, e := range entries {
		if e.IsProjector || !e.IsSupported {
			continue
		}
		name := e.ModelName
		if name == "" {
			name = e.FileName
		}
		a, err := analyzeModelFile(e.Path)
		if err != nil {
			continue
		}
		tags = append(tags, ollamaTagEntry(name, e.Path, e.SizeBytes, a))
	}
	return tags
}

func ollamaTagEntry(name, path string, sizeBytes int64, a *gopherllm.Analysis) map[string]any {
	return map[string]any{
		"name":        name,
		"model":       name,
		"modified_at": time.Now().Format(time.RFC3339Nano),
		"size":        sizeBytes,
		"digest":      modelDigest(path),
		"details":     ollamaModelDetails(a),
	}
}

// ollamaModelDetails builds Ollama's "details" object from a header gopherllm.Analysis.
func ollamaModelDetails(a *gopherllm.Analysis) map[string]any {
	quant := "unknown"
	if len(a.DTypes) > 0 {
		quant = a.DTypes[0].Type.String()
	}
	family := a.Architecture
	if family == "" {
		family = "unknown"
	}
	return map[string]any{
		"parent_model":       "",
		"format":             "gguf",
		"family":             family,
		"families":           []string{family},
		"parameter_size":     humanParamSize(a.Params),
		"quantization_level": quant,
	}
}

// humanParamSize formats a parameter count the way Ollama's "parameter_size"
// field does (e.g. "3.3B", "125M").
func humanParamSize(params int64) string {
	switch {
	case params >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(params)/1_000_000_000)
	case params >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(params)/1_000_000)
	default:
		return fmt.Sprintf("%d", params)
	}
}

// modelDigest returns a stable, cheap-to-compute "sha256:"-prefixed
// identifier for a model file: real Ollama content-addresses the whole blob,
// but sha256-ing a multi-gigabyte GGUF on every /api/tags or /api/show
// request would be far too slow. This hashes the path, size, and first 1MiB
// only — good enough as an opaque, stable client-facing id, not a real
// content hash.
func modelDigest(path string) string {
	h := sha256.New()
	io.WriteString(h, path)
	if st, err := os.Stat(path); err == nil {
		fmt.Fprintf(h, "%d", st.Size())
		if f, err := os.Open(path); err == nil {
			defer f.Close()
			_, _ = io.CopyN(h, f, 1<<20)
		}
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
