//go:build !js && !wasip1

package mmapfile

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

	mapped, err := Open(path)
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
		if !mapped.IsMapped() {
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

	mapped, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	if mapped.Len() != 0 || len(mapped.Bytes()) != 0 {
		t.Fatalf("empty file: len=%d bytes=%d, want zero", mapped.Len(), len(mapped.Bytes()))
	}
}

func TestOpenMmapMissingFile(t *testing.T) {
	_, err := Open(filepath.Join(t.TempDir(), "missing.bin"))
	if !os.IsNotExist(err) {
		t.Fatalf("OpenMmap missing file error = %v, want not exist", err)
	}
}

func TestOpenMmapHugepageAdvice(t *testing.T) {
	switch runtime.GOOS {
	case "linux":
		// proceed
	case "darwin":
		t.Skip("transparent hugepage advice is Linux-specific")
	default:
		t.Skipf("hugepage advice not wired for %s", runtime.GOOS)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "big.bin")
	data := make([]byte, 4<<20)
	for i := range data {
		data[i] = byte(i)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	mapped, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	if !mapped.IsMapped() {
		t.Fatal("expected a real OS mapping for hugepage advice test")
	}
	// Advice is best-effort and not directly observable without parsing
	// /proc/self/smaps, which is racy and permission-sensitive. Just verify the
	// mapping is valid and the bytes round-trip.
	if mapped.Len() != len(data) {
		t.Fatalf("len = %d, want %d", mapped.Len(), len(data))
	}
	got := mapped.Bytes()
	for i := range data {
		if got[i] != data[i] {
			t.Fatalf("byte mismatch at %d: got %d, want %d", i, got[i], data[i])
		}
	}

	// Opt-out must not break Open.
	t.Setenv("GOPHERLLM_NO_HUGEPAGE", "1")
	mapped2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := mapped2.Close(); err != nil {
		t.Fatal(err)
	}
}
