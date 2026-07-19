package gopherllm

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testEntry(id, arch string, projector bool) ModelEntry {
	return ModelEntry{
		ID:           id,
		Repository:   "repo",
		FileName:     id + ".gguf",
		Path:         "/models/" + id + ".gguf",
		SizeBytes:    1024,
		Architecture: arch,
		IsProjector:  projector,
		IsSupported:  ArchitectureSupported(arch),
	}
}

func TestSelectModelIgnoresProjectorMatchesWhenTextModelExists(t *testing.T) {
	entries := []ModelEntry{
		testEntry("mistral/mmproj-mistral", "clip", true),
		testEntry("mistral/mistral-Q4_K_M", "mistral3", false),
	}
	selected, err := SelectModel(entries, "mistral")
	if err != nil {
		t.Fatal(err)
	}
	if selected.ID != "mistral/mistral-Q4_K_M" {
		t.Fatalf("selected %q", selected.ID)
	}
}

func TestSelectModelReportsAmbiguousTextMatches(t *testing.T) {
	entries := []ModelEntry{
		testEntry("llama-3/llama-3-Q4_K_M", "llama", false),
		testEntry("llama-2/llama-2-Q4_K_M", "llama", false),
	}
	if _, err := SelectModel(entries, "llama"); err == nil {
		t.Fatal("expected ambiguous selector error")
	}
}

func TestArchitectureSupportedCoversImplementedLoaders(t *testing.T) {
	for _, arch := range []string{"llama", "llama2", "llama3", "mistral", "mistral3", "qwen2", "gpt-oss", "gemma", "gemma2", "gemma4"} {
		if !ArchitectureSupported(arch) {
			t.Fatalf("ArchitectureSupported(%q) = false, want true", arch)
		}
	}
	for _, arch := range []string{"phi3", "deepseek2", "nomic-bert"} {
		if ArchitectureSupported(arch) {
			t.Fatalf("ArchitectureSupported(%q) = true, want false", arch)
		}
	}
}

func TestPromptModelSelectionAcceptsNumber(t *testing.T) {
	entries := []ModelEntry{
		testEntry("alpha/alpha-Q4_K_M", "llama", false),
		testEntry("beta/beta-Q4_K_M", "mistral3", false),
	}
	var out bytes.Buffer

	selected, err := PromptModelSelection("/models", entries, strings.NewReader("2\n"), &out)
	if err != nil {
		t.Fatal(err)
	}
	if selected.ID != "beta/beta-Q4_K_M" {
		t.Fatalf("selected %q", selected.ID)
	}
	if !strings.Contains(out.String(), "Found 2 supported GGUF models") {
		t.Fatalf("selection menu was not printed: %q", out.String())
	}
}

func TestPromptModelSelectionAcceptsUniqueTextFilter(t *testing.T) {
	entries := []ModelEntry{
		testEntry("alpha/alpha-Q4_K_M", "llama", false),
		testEntry("beta/beta-Q4_K_M", "mistral3", false),
	}
	var out bytes.Buffer

	selected, err := PromptModelSelection("/models", entries, strings.NewReader("beta\n"), &out)
	if err != nil {
		t.Fatal(err)
	}
	if selected.ID != "beta/beta-Q4_K_M" {
		t.Fatalf("selected %q", selected.ID)
	}
}

func TestPromptModelSelectionCanAbort(t *testing.T) {
	entries := []ModelEntry{
		testEntry("alpha/alpha-Q4_K_M", "llama", false),
		testEntry("beta/beta-Q4_K_M", "mistral3", false),
	}

	var out bytes.Buffer
	if _, err := PromptModelSelection("/models", entries, strings.NewReader("q\n"), &out); err == nil {
		t.Fatal("expected abort error")
	}
}

func TestDefaultModelDirPrefersEnvironment(t *testing.T) {
	t.Setenv("GOPHERLLM_MODEL_DIR", "  /models/custom  ")
	t.Setenv("RUSTY_LLM_MODEL_DIR", "")
	if got := DefaultModelDir(); got != "/models/custom" {
		t.Fatalf("DefaultModelDir() = %q", got)
	}
	// The deprecated pre-rename spelling still works as a fallback...
	t.Setenv("GOPHERLLM_MODEL_DIR", "")
	t.Setenv("RUSTY_LLM_MODEL_DIR", "/models/legacy")
	if got := DefaultModelDir(); got != "/models/legacy" {
		t.Fatalf("DefaultModelDir() deprecated fallback = %q", got)
	}
	// ...but the preferred GOPHERLLM_MODEL_DIR wins when both are set.
	t.Setenv("GOPHERLLM_MODEL_DIR", "/models/new")
	if got := DefaultModelDir(); got != "/models/new" {
		t.Fatalf("DefaultModelDir() with both set = %q", got)
	}
	t.Setenv("GOPHERLLM_MODEL_DIR", "")
	t.Setenv("RUSTY_LLM_MODEL_DIR", "")
	t.Setenv("HOME", "/home/tester")
	if got := DefaultModelDir(); got != filepath.Join("/home/tester", lmStudioCommunitySubdir) {
		t.Fatalf("DefaultModelDir() = %q", got)
	}
}

func TestCatalogHelpers(t *testing.T) {
	if got := (ModelEntry{IsProjector: true, IsSupported: true}).Status(); got != "projector" {
		t.Fatalf("projector status = %q", got)
	}
	if got := (ModelEntry{IsSupported: true}).Status(); got != "supported" {
		t.Fatalf("supported status = %q", got)
	}
	if got := (ModelEntry{}).Status(); got != "unsupported" {
		t.Fatalf("unsupported status = %q", got)
	}
	if got := truncate("abcdef", 4); got != "abc~" {
		t.Fatalf("truncate = %q", got)
	}
	if got := truncate("abcdef", 1); got != "~" {
		t.Fatalf("truncate short = %q", got)
	}
	entries := []ModelEntry{{ID: "first", Path: "/models/first.gguf"}, {ID: "second", Path: "/models/second.gguf"}}
	if got := modelMenuIndex(entries, entries[1]); got != 2 {
		t.Fatalf("modelMenuIndex = %d", got)
	}
	if got := modelMenuIndex(entries, ModelEntry{ID: "missing"}); got != 0 {
		t.Fatalf("missing modelMenuIndex = %d", got)
	}
	if want, got := filepath.Dir(entries[0].Path), modelDirFromEntries(entries); got != want {
		t.Fatalf("modelDirFromEntries = %q, want %q", got, want)
	}
	if got := modelDirFromEntries(nil); got != "the model directory" {
		t.Fatalf("empty modelDirFromEntries = %q", got)
	}
}

func TestDiscoverModelsAndResolveModelPath(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "repo", "tiny.gguf")
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath, buildTinyLlamaGGUF(), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "broken.gguf"), []byte("not a GGUF"), 0o600); err != nil {
		t.Fatal(err)
	}

	var logs bytes.Buffer
	entries, err := DiscoverModels(root, &logs)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].ID != filepath.Join("repo", "tiny") || entries[0].ModelName != "tiny" || !entries[0].IsSupported {
		t.Fatalf("entries = %+v", entries)
	}
	if !strings.Contains(logs.String(), "Skipping") {
		t.Fatalf("expected broken model diagnostic, got %q", logs.String())
	}

	selector := "tiny"
	if got, err := ResolveModelPath(&selector, root); err != nil || got != modelPath {
		t.Fatalf("ResolveModelPath selector = %q, %v", got, err)
	}
	if got, err := ResolveModelPath(&modelPath, root); err != nil || got != modelPath {
		t.Fatalf("ResolveModelPath path = %q, %v", got, err)
	}
	if got, err := chooseFromDirectory(root, nil, nil, nil); err != nil || got != modelPath {
		t.Fatalf("chooseFromDirectory = %q, %v", got, err)
	}
}
