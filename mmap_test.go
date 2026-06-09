package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenMmapReadsBytesAndCloses(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sample.bin")
	want := []byte("gopherllm")
	if err := os.WriteFile(path, want, 0o600); err != nil {
		t.Fatal(err)
	}

	mapped, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(mapped.Bytes()) != string(want) {
		t.Fatalf("mapped bytes = %q, want %q", mapped.Bytes(), want)
	}
	if mapped.Len() != len(want) {
		t.Fatalf("len = %d, want %d", mapped.Len(), len(want))
	}
	if err := mapped.Close(); err != nil {
		t.Fatal(err)
	}
	if mapped.Len() != 0 {
		t.Fatalf("len after close = %d, want 0", mapped.Len())
	}
}
