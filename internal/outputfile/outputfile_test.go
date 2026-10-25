package outputfile

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestWritePreservesExistingOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "output")
	if err := CheckNew(path); err != nil {
		t.Fatal(err)
	}
	// The destination can appear after the initial check, during generation.
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CheckNew(path); err == nil {
		t.Fatal("existing output passed the early check")
	}
	called := false
	err := Write(path, func(io.Writer) error {
		called = true
		return nil
	})
	if !errors.Is(err, os.ErrExist) || called {
		t.Fatalf("Write = %v, encoder called = %v", err, called)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "original" {
		t.Fatalf("existing output changed: %q, %v", data, err)
	}
}

func TestWriteRemovesIncompleteOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "output")
	want := errors.New("encoding failed")
	err := Write(path, func(w io.Writer) error {
		if _, err := io.WriteString(w, "partial"); err != nil {
			t.Fatal(err)
		}
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("Write = %v, want %v", err, want)
	}
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		t.Fatalf("incomplete output remains: %v", err)
	}
	if err := Write(path, func(w io.Writer) error {
		_, err := io.WriteString(w, "complete")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "complete" {
		t.Fatalf("completed output = %q, %v", data, err)
	}
}

func TestWriteRejectsDanglingSymlink(t *testing.T) {
	dir := t.TempDir()
	target, path := filepath.Join(dir, "target"), filepath.Join(dir, "output")
	if err := os.Symlink(target, path); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := CheckNew(path); err == nil {
		t.Fatal("dangling symlink passed the early check")
	}
	if err := Write(path, func(w io.Writer) error {
		_, err := io.WriteString(w, "data")
		return err
	}); !errors.Is(err, os.ErrExist) {
		t.Fatalf("Write = %v, want ErrExist", err)
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("symlink target was created: %v", err)
	}
	if _, err := os.Lstat(path); err != nil {
		t.Fatalf("existing symlink was removed: %v", err)
	}
}
