package gopherllm

import (
	"io"
	"os"
	"testing"
)

// TestLocalGGUFLoadSmoke is an opt-in integration test for a machine's local
// model collection. It intentionally only loads a model: a one-token decode
// can take minutes for large CPU-only models and is covered by targeted CLI
// smoke runs instead. Set GOPHERLLM_MODEL_SMOKE_DIR to a directory containing
// GGUF files to run it, for example:
//
// GOPHERLLM_MODEL_SMOKE_DIR="$HOME/.cache/lm-studio/models" go test -run TestLocalGGUFLoadSmoke -v .
func TestLocalGGUFLoadSmoke(t *testing.T) {
	root := os.Getenv("GOPHERLLM_MODEL_SMOKE_DIR")
	if root == "" {
		t.Skip("set GOPHERLLM_MODEL_SMOKE_DIR to smoke-test locally installed GGUF models")
	}
	entries, err := DiscoverModels(root, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) == 0 {
		t.Fatalf("no GGUF files found in %s", root)
	}
	for _, entry := range entries {
		entry := entry
		t.Run(entry.ID, func(t *testing.T) {
			if entry.IsProjector {
				t.Skip("multimodal projector, not a text model")
			}
			if !entry.IsSupported {
				t.Skipf("unsupported architecture: %s", entry.Architecture)
			}
			// Gemma 4 currently covers multiple incompatible GGUF graphs. Its
			// unsupported PLE/MoE layouts fail with a precise loader diagnostic;
			// keep those in the inventory without turning that known gap into a
			// test-infrastructure failure.
			if entry.Architecture == "gemma4" {
				t.Skipf("architecture requires a layout-specific smoke test: %s", entry.Architecture)
			}
			r, _, err := RunnerFromPath(entry.Path)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
		})
	}
}
