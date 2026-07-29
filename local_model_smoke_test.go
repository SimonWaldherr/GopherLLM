package gopherllm

import (
	"io"
	"os"
	"testing"
)

// TestLocalGGUFLoadSmoke is an opt-in integration test for a machine's complete
// local model collection. It validates every supported non-projector layout
// in out-of-core mode; a one-token decode can take minutes for large CPU-only
// dense models and is covered by targeted CLI smoke runs instead. Set
// GOPHERLLM_MODEL_SMOKE_DIR to a directory containing GGUF files, for example:
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
			r, _, err := RunnerFromPathWithOptions(entry.Path, LoadOptions{OutOfCore: true})
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
		})
	}
}
