//go:build js || wasip1

package mmapfile

import "testing"

func TestOpenAlwaysErrors(t *testing.T) {
	f, err := Open("anything.gguf")
	if err == nil {
		t.Fatal("Open: want error under GOOS=js/wasip1, got nil")
	}
	if f != nil {
		t.Fatalf("Open: want nil *File on error, got %+v", f)
	}
}

func TestZeroValueFileIsSafe(t *testing.T) {
	var f File
	if f.Bytes() != nil {
		t.Fatalf("Bytes() = %v, want nil", f.Bytes())
	}
	if f.Len() != 0 {
		t.Fatalf("Len() = %d, want 0", f.Len())
	}
	if f.IsMapped() {
		t.Fatal("IsMapped() = true, want false")
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close() = %v, want nil", err)
	}
}
