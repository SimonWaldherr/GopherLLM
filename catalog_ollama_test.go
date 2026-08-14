package gopherllm

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeOllamaBlob stores content the way Ollama does — content-addressed,
// with the digest's colon replaced by a dash in the filename — and returns
// the "sha256:<hex>" digest a manifest would reference.
func writeOllamaBlob(t *testing.T, root string, content []byte) string {
	t.Helper()
	sum := sha256.Sum256(content)
	digest := hex.EncodeToString(sum[:])
	blobs := filepath.Join(root, "blobs")
	if err := os.MkdirAll(blobs, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blobs, "sha256-"+digest), content, 0o600); err != nil {
		t.Fatal(err)
	}
	return "sha256:" + digest
}

// writeOllamaManifest writes manifests/<registry>/<namespace>/<model>/<tag>.
func writeOllamaManifest(t *testing.T, root, registry, namespace, model, tag string, layers []map[string]any) {
	t.Helper()
	dir := filepath.Join(root, "manifests", registry, namespace, model)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"schemaVersion": 2,
		"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
		"layers":        layers,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, tag), body, 0o600); err != nil {
		t.Fatal(err)
	}
}

func modelLayer(digest string) map[string]any {
	return map[string]any{"mediaType": ollamaModelMediaType, "digest": digest, "size": 1}
}

func TestDiscoverOllamaModelsNamesAndResolvesBlobs(t *testing.T) {
	root := t.TempDir()
	gguf := buildTinyLlamaGGUF()
	digest := writeOllamaBlob(t, root, gguf)

	// A library model renders as "name:tag"; a namespaced one keeps its owner.
	writeOllamaManifest(t, root, "registry.ollama.ai", "library", "tinyllama", "latest",
		[]map[string]any{modelLayer(digest)})
	writeOllamaManifest(t, root, "registry.ollama.ai", "someuser", "custom", "q4",
		[]map[string]any{modelLayer(digest)})

	entries, err := DiscoverOllamaModels(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("discovered %d models, want 2: %+v", len(entries), entries)
	}
	byID := map[string]ModelEntry{}
	for _, e := range entries {
		byID[e.ID] = e
	}
	if _, ok := byID["tinyllama:latest"]; !ok {
		t.Fatalf("missing library model tinyllama:latest, got %v", byID)
	}
	if _, ok := byID["someuser/custom:q4"]; !ok {
		t.Fatalf("missing namespaced model someuser/custom:q4, got %v", byID)
	}

	e := byID["tinyllama:latest"]
	if e.Architecture != "llama" {
		t.Fatalf("architecture = %q, want llama (header parse did not run)", e.Architecture)
	}
	if !e.IsSupported {
		t.Fatal("tinyllama should be supported")
	}
	if e.Repository != "ollama" {
		t.Fatalf("repository = %q, want ollama", e.Repository)
	}
	// The path must point at the real blob so the model can actually be loaded.
	if got, err := os.ReadFile(e.Path); err != nil || !bytes.Equal(got, gguf) {
		t.Fatalf("entry path %q does not resolve to the model blob (err=%v)", e.Path, err)
	}
}

// A half-pulled model (manifest present, blob not yet fetched) is routine with
// Ollama and must not hide the rest of the store.
func TestDiscoverOllamaModelsSkipsIncompleteEntries(t *testing.T) {
	root := t.TempDir()
	good := writeOllamaBlob(t, root, buildTinyLlamaGGUF())

	writeOllamaManifest(t, root, "registry.ollama.ai", "library", "good", "latest",
		[]map[string]any{modelLayer(good)})
	// Blob never written.
	writeOllamaManifest(t, root, "registry.ollama.ai", "library", "halfpulled", "latest",
		[]map[string]any{modelLayer("sha256:" + hexRepeat("a", 64))})
	// Manifest with no model layer at all.
	writeOllamaManifest(t, root, "registry.ollama.ai", "library", "nomodel", "latest",
		[]map[string]any{{"mediaType": "application/vnd.ollama.image.template", "digest": good, "size": 1}})
	// Not JSON.
	junkDir := filepath.Join(root, "manifests", "registry.ollama.ai", "library", "junk")
	if err := os.MkdirAll(junkDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(junkDir, "latest"), []byte("{{{"), 0o600); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	entries, err := DiscoverOllamaModels(root, &logs)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].ID != "good:latest" {
		t.Fatalf("entries = %+v, want only good:latest", entries)
	}
	if logs.Len() == 0 {
		t.Fatal("skipped manifests should be reported")
	}
}

// A manifest is data on disk, not a trusted path fragment: a digest containing
// traversal must not read outside the blobs directory.
func TestOllamaBlobPathRejectsMalformedDigests(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "blobs"), 0o700); err != nil {
		t.Fatal(err)
	}
	for _, digest := range []string{
		"sha256:../../etc/passwd",
		"sha256:",
		":abcdef",
		"nocolon",
		"sha256:zzzz",
		"../evil:abcdef",
	} {
		if _, err := ollamaBlobPath(root, digest); err == nil {
			t.Fatalf("digest %q should have been rejected", digest)
		}
	}
}

func TestDiscoverOllamaModelsRejectsNonStore(t *testing.T) {
	if _, err := DiscoverOllamaModels(t.TempDir(), nil); err == nil {
		t.Fatal("a directory without manifests/ is not an Ollama store")
	}
}

func TestOllamaModelsRootHonoursEnvironment(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", dir)
	if got := OllamaModelsRoot(); got != dir {
		t.Fatalf("OllamaModelsRoot() = %q, want %q", got, dir)
	}
	// A configured-but-absent directory reports "" rather than falling back to
	// the home default, so an explicit override is never silently ignored.
	t.Setenv("OLLAMA_MODELS", filepath.Join(dir, "does-not-exist"))
	if got := OllamaModelsRoot(); got != "" {
		t.Fatalf("OllamaModelsRoot() = %q, want empty for a missing override", got)
	}
}

func TestDiscoverOllamaModelsDefaultIsQuietWhenAbsent(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", filepath.Join(t.TempDir(), "absent"))
	entries, err := DiscoverOllamaModelsDefault(nil)
	if err != nil {
		t.Fatalf("missing Ollama store should not be an error: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("entries = %+v, want none", entries)
	}
}

func hexRepeat(s string, n int) string {
	out := make([]byte, 0, n)
	for len(out) < n {
		out = append(out, s...)
	}
	return string(out[:n])
}
