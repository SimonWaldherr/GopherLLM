package gopherllm

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestExternalConsumerBuilds compiles testdata/consumer — a separate Go
// module that imports github.com/SimonWaldherr/GopherLLM through a replace
// directive — proving the module path is correct and the public API is
// usable by an external `go get` consumer. (testdata/ is invisible to the go
// tool, so the nested module never interferes with normal builds.)
func TestExternalConsumerBuilds(t *testing.T) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go binary not in PATH")
	}
	dir, err := filepath.Abs(filepath.Join("testdata", "consumer"))
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "consumer.exe")
	cmd := exec.Command(goBin, "build", "-o", out, ".")
	cmd.Dir = dir
	cmd.Env = os.Environ()
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("external consumer failed to build: %v\n%s", err, output)
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatalf("consumer binary missing: %v", err)
	}
}
