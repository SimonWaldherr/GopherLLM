package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLastServerModelRoundTrip(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "state", "last-model.json")
	t.Setenv(serverStatePathEnv, statePath)
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("test"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := saveLastServerModel(modelPath); err != nil {
		t.Fatal(err)
	}
	got, err := loadLastServerModel()
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("saved model = %q, want %q", got, want)
	}
	info, err := os.Stat(statePath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("state file permissions = %o, want 600", info.Mode().Perm())
	}
}

func TestLoadLastServerModelIgnoresRemovedModel(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "last-model.json")
	t.Setenv(serverStatePathEnv, statePath)
	if err := os.WriteFile(statePath, []byte(`{"model_path":"/no/such/model.gguf"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := loadLastServerModel()
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Fatalf("removed model = %q, want empty", got)
	}
}
