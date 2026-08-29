package rag

import (
	"bytes"
	"context"
	"encoding/gob"
	"testing"
)

func TestIndexSaveLoadRoundTrip(t *testing.T) {
	embed := fakeEmbedder{vec: func(text string) []float32 { return []float32{1, 0} }}
	ix := New(Options{Embedder: embed, EmbedderID: "test-embedder-v1"})
	if err := ix.Add(context.Background(),
		Doc{ID: "d1", Title: "First", Text: "The quick brown fox jumps over the lazy dog."},
		Doc{ID: "d2", Title: "Second", Text: "A second document with different content entirely."},
	); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := ix.Save(&buf); err != nil {
		t.Fatal(err)
	}

	loaded, err := Load(&buf, Options{Embedder: embed, EmbedderID: "test-embedder-v1"})
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != ix.Len() || loaded.Docs() != ix.Docs() {
		t.Fatalf("loaded Len/Docs = %d/%d, want %d/%d", loaded.Len(), loaded.Docs(), ix.Len(), ix.Docs())
	}

	want, err := ix.Search(context.Background(), "quick fox", Query{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Search(context.Background(), "quick fox", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(want) == 0 || len(got) != len(want) {
		t.Fatalf("got %d hits after reload, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Chunk.DocID != want[i].Chunk.DocID || got[i].Chunk.Text != want[i].Chunk.Text {
			t.Fatalf("hit %d differs after reload: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestLoadRefusesEmbedderIDMismatch(t *testing.T) {
	ix := New(Options{EmbedderID: "embedder-a"})
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "content"}); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := ix.Save(&buf); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(&buf, Options{EmbedderID: "embedder-b"}); err == nil {
		t.Fatal("expected Load to refuse a mismatched EmbedderID")
	}
}

func TestLoadRefusesUnknownVersion(t *testing.T) {
	var buf bytes.Buffer
	badSnap := snapshot{Version: snapshotVersion + 1}
	if err := gob.NewEncoder(&buf).Encode(&badSnap); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(&buf, Options{}); err == nil {
		t.Fatal("expected Load to refuse an unknown snapshot version")
	}
}

func TestIndexEmptyRoundTrip(t *testing.T) {
	ix := New(Options{})
	var buf bytes.Buffer
	if err := ix.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(&buf, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != 0 || loaded.Docs() != 0 {
		t.Fatalf("loaded empty index has Len=%d Docs=%d, want 0,0", loaded.Len(), loaded.Docs())
	}
}
