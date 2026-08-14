package gopherllm

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Ollama keeps pulled models in a content-addressed store rather than as
// browsable .gguf files, which is why the ordinary DiscoverModels walk finds
// nothing useful there: every blob is named "sha256-<hex>" with no extension.
// The mapping from a human name like "llama3.2:3b" to its weights lives in an
// OCI image manifest.
//
//	<root>/manifests/<registry>/<namespace>/<model>/<tag>   JSON manifest
//	<root>/blobs/sha256-<hex>                               content-addressed layers
//
// GopherLLM already speaks Ollama's HTTP API and can proxy to an Ollama
// server; this is the missing third side of that integration — reusing the
// models a user has already pulled, with no copy, conversion, or re-download.
const (
	ollamaManifestsDir = "manifests"
	ollamaBlobsDir     = "blobs"

	// ollamaModelMediaType marks the layer holding the GGUF weights.
	ollamaModelMediaType = "application/vnd.ollama.image.model"
	// ollamaProjectorMediaType marks a multimodal projector layer. Ollama
	// states the pairing explicitly, so vision models resolve their projector
	// from the manifest instead of the same-directory heuristic
	// DiscoverModels has to use — every blob shares one directory here, so
	// that heuristic would pair unrelated models.
	ollamaProjectorMediaType = "application/vnd.ollama.image.projector"
)

type ollamaManifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

// OllamaModelsRoot reports the Ollama model store to scan, honouring the
// OLLAMA_MODELS override Ollama itself uses. It returns "" when no store is
// present, so callers can treat Ollama as simply absent rather than an error.
func OllamaModelsRoot() string {
	if dir := strings.TrimSpace(os.Getenv("OLLAMA_MODELS")); dir != "" {
		if isDir(dir) {
			return dir
		}
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	// The per-user default. A Linux system service instead uses
	// /usr/share/ollama/.ollama/models, which OLLAMA_MODELS can point at.
	root := filepath.Join(home, ".ollama", "models")
	if isDir(root) {
		return root
	}
	return ""
}

func isDir(path string) bool {
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}

// DiscoverOllamaModels lists the GGUF models in an Ollama store at root,
// identified by the same "name:tag" a user would pass to `ollama run`. Entries
// are otherwise identical to DiscoverModels' — the same header parse fills in
// architecture, context length and support status — so callers can merge the
// two lists freely.
//
// A manifest that is unreadable, unparseable, or whose model blob is missing
// is skipped with a note rather than failing the scan: a partially pulled or
// half-deleted model should not hide the rest of the store.
func DiscoverOllamaModels(root string, logw io.Writer) ([]ModelEntry, error) {
	if logw == nil {
		logw = io.Discard
	}
	manifestRoot := filepath.Join(root, ollamaManifestsDir)
	if !isDir(manifestRoot) {
		return nil, fmt.Errorf("not an Ollama model store (no %s directory): %s", ollamaManifestsDir, root)
	}

	var manifests []string
	err := filepath.WalkDir(manifestRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			// A single unreadable subtree must not abort the walk.
			return nil
		}
		if !d.IsDir() {
			manifests = append(manifests, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to scan %s: %w", manifestRoot, err)
	}
	sort.Strings(manifests)

	entries := make([]ModelEntry, 0, len(manifests))
	for _, manifestPath := range manifests {
		entry, err := inspectOllamaManifest(root, manifestRoot, manifestPath)
		if err != nil {
			fmt.Fprintf(logw, "Skipping Ollama manifest %s: %v\n", manifestPath, err)
			continue
		}
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool { return modelSortKey(entries[i]) < modelSortKey(entries[j]) })
	return entries, nil
}

// DiscoverOllamaModelsDefault scans the default store, returning no entries
// and no error when Ollama is not installed.
func DiscoverOllamaModelsDefault(logw io.Writer) ([]ModelEntry, error) {
	root := OllamaModelsRoot()
	if root == "" {
		return nil, nil
	}
	return DiscoverOllamaModels(root, logw)
}

func inspectOllamaManifest(root, manifestRoot, manifestPath string) (ModelEntry, error) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return ModelEntry{}, err
	}
	var manifest ollamaManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return ModelEntry{}, fmt.Errorf("unparseable manifest: %w", err)
	}
	var modelDigest, projectorDigest string
	for _, l := range manifest.Layers {
		switch l.MediaType {
		case ollamaModelMediaType:
			modelDigest = l.Digest
		case ollamaProjectorMediaType:
			projectorDigest = l.Digest
		}
	}
	if modelDigest == "" {
		return ModelEntry{}, fmt.Errorf("no %s layer", ollamaModelMediaType)
	}
	blobPath, err := ollamaBlobPath(root, modelDigest)
	if err != nil {
		return ModelEntry{}, err
	}

	// Reuse the ordinary header inspection so support status, architecture,
	// embedding detection and context length cannot drift between the two
	// discovery paths. root is passed as the blobs directory only so the
	// derived ID is the bare digest; it is replaced below.
	entry, err := inspectModel(filepath.Dir(blobPath), blobPath)
	if err != nil {
		return ModelEntry{}, err
	}

	name := ollamaModelName(manifestRoot, manifestPath)
	entry.ID = name
	entry.ModelName = firstNonEmpty(entry.ModelName, name)
	entry.FileName = name
	entry.Repository = "ollama"

	// Ollama names the projector explicitly, so this pairing is exact rather
	// than heuristic. It is still filtered by the same compatibility rules:
	// only a projector this engine can actually run gets advertised.
	entry.ProjectorPath = ""
	if projectorDigest != "" && entry.visionTemplateCapable && entry.visionDimension > 0 {
		if projPath, err := ollamaBlobPath(root, projectorDigest); err == nil {
			if proj, err := inspectModel(filepath.Dir(projPath), projPath); err == nil {
				if proj.IsProjector && proj.visionPixtralProjector && proj.visionDimension == entry.visionDimension {
					entry.ProjectorPath = projPath
				}
			}
		}
	}
	return entry, nil
}

// ollamaBlobPath maps an OCI digest ("sha256:<hex>") to its on-disk blob
// ("sha256-<hex>"). The digest is validated rather than pasted into a path:
// it comes from a file on disk, and a manifest containing "../" must not be
// able to escape the blobs directory.
func ollamaBlobPath(root, digest string) (string, error) {
	alg, hex, ok := strings.Cut(digest, ":")
	if !ok || alg == "" || hex == "" {
		return "", fmt.Errorf("malformed digest %q", digest)
	}
	if !isHexString(hex) || !isPlainToken(alg) {
		return "", fmt.Errorf("malformed digest %q", digest)
	}
	path := filepath.Join(root, ollamaBlobsDir, alg+"-"+hex)
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("missing blob %s: %w", digest, err)
	}
	return path, nil
}

func isHexString(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
			return false
		}
	}
	return true
}

func isPlainToken(s string) bool {
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9') {
			return false
		}
	}
	return s != ""
}

// ollamaModelName turns a manifest path back into the reference a user typed:
// <registry>/<namespace>/<model>/<tag> becomes "model:tag" for the default
// library namespace and "namespace/model:tag" otherwise, matching how Ollama
// itself displays them.
func ollamaModelName(manifestRoot, manifestPath string) string {
	rel, err := filepath.Rel(manifestRoot, manifestPath)
	if err != nil {
		return filepath.Base(manifestPath)
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) < 2 {
		return strings.Join(parts, "/")
	}
	tag := parts[len(parts)-1]
	model := parts[len(parts)-2]
	namespace := ""
	if len(parts) >= 3 {
		namespace = parts[len(parts)-3]
	}
	if namespace != "" && namespace != "library" {
		return namespace + "/" + model + ":" + tag
	}
	return model + ":" + tag
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
