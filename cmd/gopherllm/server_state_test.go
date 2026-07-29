package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestLastModelRoundTrip(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "state", "last-model.json")
	t.Setenv(modelStatePathEnv, statePath)
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("test"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := saveLastModel(modelPath); err != nil {
		t.Fatal(err)
	}
	got, err := loadLastModel()
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

func TestLoadLastModelIgnoresRemovedModel(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "last-model.json")
	t.Setenv(modelStatePathEnv, statePath)
	if err := os.WriteFile(statePath, []byte(`{"model_path":"/no/such/model.gguf"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := loadLastModel()
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Fatalf("removed model = %q, want empty", got)
	}
}

func TestResumeLastModelAppliesToAnyCLIMode(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "last-model.json")
	t.Setenv(modelStatePathEnv, statePath)
	modelPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(modelPath, []byte("test"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := saveLastModel(modelPath); err != nil {
		t.Fatal(err)
	}
	cfg, err := parseCLI([]string{"--prompt", "hello"})
	if err != nil {
		t.Fatal(err)
	}
	var logs bytes.Buffer
	if !resumeLastModel(&cfg, &logs) {
		t.Fatal("remembered model was not resumed")
	}
	if cfg.modelSelector == nil || *cfg.modelSelector != modelPath {
		t.Fatalf("selector = %v, want %q", cfg.modelSelector, modelPath)
	}
	if logs.Len() == 0 {
		t.Fatal("resume was not logged")
	}

	explicit := "other.gguf"
	cfg.modelSelector = &explicit
	if resumeLastModel(&cfg, &logs) {
		t.Fatal("remembered model overrode an explicit selector")
	}
	if *cfg.modelSelector != explicit {
		t.Fatalf("explicit selector = %q, want %q", *cfg.modelSelector, explicit)
	}
}
