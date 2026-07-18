package gopherllm

import (
	"os"
	"path/filepath"
	"runtime"
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
	switch runtime.GOOS {
	case "windows", "linux", "darwin":
		if !mapped.mmap {
			t.Fatal("expected a real OS mapping on this platform, got the read-copy fallback")
		}
	}
	if err := mapped.Close(); err != nil {
		t.Fatal(err)
	}
	if mapped.Len() != 0 {
		t.Fatalf("len after close = %d, want 0", mapped.Len())
	}
	if err := mapped.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}

func TestOpenMmapEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.bin")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	mapped, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	if mapped.Len() != 0 || len(mapped.Bytes()) != 0 {
		t.Fatalf("empty file: len=%d bytes=%d, want zero", mapped.Len(), len(mapped.Bytes()))
	}
}

func TestOpenMmapMissingFile(t *testing.T) {
	_, err := OpenMmap(filepath.Join(t.TempDir(), "missing.bin"))
	if !os.IsNotExist(err) {
		t.Fatalf("OpenMmap missing file error = %v, want not exist", err)
	}
}
