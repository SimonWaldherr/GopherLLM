package huggingface

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func containsHFTag(tags []string, tag string) bool {
	for _, value := range tags {
		if value == tag {
			return true
		}
	}
	return false
}

// File describes a repository artifact, independently of its inference runtime.
// Role is a filename-based hint; it does not certify model compatibility.
type File struct {
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes"`
	Format    string `json:"format"`
	Role      string `json:"role"`
}

// Manifest is the inventory of a revision. Commit is the Hub commit when
// supplied by the endpoint; Revision retains the requested branch or tag.
type Manifest struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Commit     string `json:"commit,omitempty"`
	Files      []File `json:"files"`
}

func describeHFFile(entry hfTreeEntry) File {
	name := strings.ToLower(filepath.Base(entry.Path))
	format := strings.TrimPrefix(strings.ToLower(filepath.Ext(name)), ".")
	role := "asset"
	switch {
	case strings.Contains(name, "mmproj") || strings.Contains(name, "projector"):
		role = "projector"
	case strings.Contains(name, "tokenizer") || strings.Contains(name, "vocab") || name == "merges.txt":
		role = "tokenizer"
	case strings.Contains(name, "processor") || strings.Contains(name, "preprocessor"):
		role = "processor"
	case strings.HasSuffix(name, ".index.json"):
		role = "weight-index"
	case format == "gguf" || format == "safetensors" || format == "onnx" || format == "bin" || format == "pt" || format == "pth":
		role = "weights"
	case strings.Contains(name, "config") || name == "params.json":
		role = "config"
	case format == "py":
		role = "code"
	}
	return File{Path: entry.Path, SizeBytes: entry.Size, Format: format, Role: role}
}

func artifactReference(ref string) (hfReference, error) {
	r, err := ParseHuggingFaceReference(strings.TrimSpace(ref))
	if err == nil && r.Quant != "" {
		err = errors.New("artifact reference must not contain a GGUF selector; pass exact file paths separately")
	}
	return r, err
}

// Inspect lists all file types without downloading weights or executing code.
// Offline inventories contain only locally available files, not a remote listing.
func Inspect(ctx context.Context, ref string, opts Options) (Manifest, error) {
	r, err := artifactReference(ref)
	if err != nil {
		return Manifest{}, err
	}
	if err := ctx.Err(); err != nil {
		return Manifest{}, err
	}
	var entries []hfTreeEntry
	var commit string
	if opts.Offline || hfBoolEnv("HF_HUB_OFFLINE") {
		snapshot := newHFCache(r).snapshotForRef(r.Revision)
		if snapshot == "" {
			return Manifest{}, errors.New("Hugging Face revision is not cached")
		}
		commit = filepath.Base(snapshot)
		err = filepath.WalkDir(snapshot, func(path string, d os.DirEntry, walkErr error) error {
			if err := ctx.Err(); err != nil {
				return err
			}
			if walkErr != nil {
				return walkErr
			}
			if d.IsDir() {
				return nil
			}
			st, e := os.Stat(path)
			if e != nil || !st.Mode().IsRegular() {
				return nil
			}
			rel, e := filepath.Rel(snapshot, path)
			if e != nil {
				return e
			}
			entries = append(entries, hfTreeEntry{Path: filepath.ToSlash(rel), Type: "file", Size: st.Size()})
			return nil
		})
	} else {
		entries, commit, err = hfListFiles(ctx, r)
	}
	if err != nil {
		return Manifest{}, err
	}
	m := Manifest{Repository: r.Repository, Revision: r.Revision, Commit: commit, Files: []File{}}
	for _, entry := range entries {
		if entry.Type == "file" && safeHFFilePath(entry.Path) {
			m.Files = append(m.Files, describeHFFile(entry))
		}
	}
	sort.Slice(m.Files, func(i, j int) bool { return m.Files[i].Path < m.Files[j].Path })
	return m, nil
}

// DownloadFiles fetches an explicit set of artifacts, retaining subdirectories.
// It supports all file formats, but never loads a model or executes repository
// code. Choose companion files from Inspect; no unrelated weights are guessed.
// All transfers use one commit when the Hub supplies it. refs are published only
// after the entire requested set succeeds. Offline mode requires every file.
func DownloadFiles(ctx context.Context, ref string, names []string, logw io.Writer, opts Options) ([]string, error) {
	if len(names) == 0 {
		return nil, errors.New("select at least one repository file")
	}
	seen := map[string]bool{}
	for _, name := range names {
		if !safeHFFilePath(name) {
			return nil, fmt.Errorf("invalid repository file %q", name)
		}
		if seen[name] {
			return nil, fmt.Errorf("duplicate repository file %q", name)
		}
		seen[name] = true
	}
	m, err := Inspect(ctx, ref, opts)
	if err != nil {
		return nil, err
	}
	inventory := map[string]File{}
	for _, file := range m.Files {
		inventory[file.Path] = file
	}
	for _, name := range names {
		if _, ok := inventory[name]; !ok {
			return nil, fmt.Errorf("repository file %q is unavailable at %s", name, m.Revision)
		}
	}
	r := hfReference{Repository: m.Repository, Revision: m.Revision}
	cache := newHFCache(r)
	commit := m.Commit
	if commit == "" {
		commit = safeHFPathPart(r.Revision)
	}
	snapshot := cache.snapshot(commit)
	resolved := make([]string, len(names))
	tasks := []hfDownloadTask{}
	for i, name := range names {
		local := filepath.Join(snapshot, filepath.FromSlash(name))
		resolved[i] = local
		size := inventory[name].SizeBytes
		if st, e := os.Stat(local); e == nil && st.Mode().IsRegular() && st.Size() == size {
			continue
		}
		if opts.Offline || hfBoolEnv("HF_HUB_OFFLINE") {
			return nil, fmt.Errorf("file %q is not completely cached", name)
		}
		tasks = append(tasks, hfDownloadTask{name: name, path: local, size: size})
	}
	if logw == nil {
		logw = io.Discard
	} else {
		logw = &synchronizedWriter{w: logw}
	}
	if m.Commit != "" {
		r.Revision = m.Commit
	}
	if err := downloadHFFiles(ctx, r, cache, tasks, logw, opts.OnProgress); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := cache.writeRef(m.Revision, commit); err != nil {
		return nil, err
	}
	return resolved, nil
}
