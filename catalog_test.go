package main

import (
	"bytes"
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
