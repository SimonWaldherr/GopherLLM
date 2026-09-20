package mobile

import "testing"

func TestEngineLifecycleWithoutModel(t *testing.T) {
	e := NewEngine()
	if _, err := e.Generate("hello", ""); err != ErrNoModel {
		t.Fatalf("Generate = %v, want %v", err, ErrNoModel)
	}
	e.Cancel()
	e.Cancel()
	if err := e.Unload(); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.Load("missing.gguf", ""); err == nil {
		t.Fatal("Load after Close/missing path should fail")
	}
	if e.IsLoaded() {
		t.Fatal("closed engine reports loaded")
	}
}

func TestEngineClosedErrorWinsForExistingPath(t *testing.T) {
	e := NewEngine()
	_ = e.Close()
	// A real path avoids depending on a GGUF fixture and reaches state handling.
	if err := e.Load("../go.mod", ""); err != ErrClosed {
		t.Fatalf("Load = %v, want %v", err, ErrClosed)
	}
}
