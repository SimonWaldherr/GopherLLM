package laya_test

import (
	"os"
	"path/filepath"
	"testing"
)

// layaTinyFixture returns the tiny Laya checkpoint that
// scripts/generate_laya_fixture.py writes (it needs PyTorch and an upstream
// Laya checkout, so it is not committed) and skips the test when it is absent.
func layaTinyFixture(t testing.TB) string {
	t.Helper()
	const dir = "../../testdata/laya-tiny"
	if _, err := os.Stat(filepath.Join(dir, "rl_agent_config.json")); err != nil {
		t.Skipf("Laya fixture not generated (run scripts/generate_laya_fixture.py): %v", err)
	}
	return dir
}
