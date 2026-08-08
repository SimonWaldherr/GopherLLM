package gopherllm

import (
	"os"
	"path/filepath"
	"testing"
)

// TestGenerateWasmFixture writes the tiny synthetic mistral3 GGUF already
// used by forward_batch_test.go (buildTinyStandardGGUF, gguf_synth_test.go)
// out to cmd/gopherllm-wasm/testdata/harness/tiny-model.gguf, for the
// browser wasm harness to fetch. It's opt-in (skipped by default) since it
// writes a file as a side effect rather than asserting anything itself;
// regenerate with:
//
//	GOPHERLLM_GEN_WASM_FIXTURE=1 go test -run TestGenerateWasmFixture -v .
func TestGenerateWasmFixture(t *testing.T) {
	if os.Getenv("GOPHERLLM_GEN_WASM_FIXTURE") != "1" {
		t.Skip("set GOPHERLLM_GEN_WASM_FIXTURE=1 to (re)generate cmd/gopherllm-wasm/testdata/harness/tiny-model.gguf")
	}

	data := buildTinyStandardGGUF("mistral3")

	out := filepath.Join("cmd", "gopherllm-wasm", "testdata", "harness", "tiny-model.gguf")
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, data, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %d bytes to %s", len(data), out)
}
