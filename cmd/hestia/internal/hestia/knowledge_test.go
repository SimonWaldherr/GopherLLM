package hestia

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/testmodel"
	tinysql "github.com/SimonWaldherr/tinySQL"
)

// stubEmbedder gives every text the same fixed vector, sufficient to prove
// the search/ingest wiring works without needing a real embedding model.
type stubEmbedder struct{}

func (stubEmbedder) EmbedSingle(text string) ([]float64, error) { return []float64{1, 0}, nil }
func (stubEmbedder) Embed(texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i := range out {
		out[i] = []float64{1, 0}
	}
	return out, nil
}

func testKnowledgeBase(t *testing.T) *KnowledgeBase {
	t.Helper()
	kb, err := newKnowledgeBaseFromEmbedder(stubEmbedder{}, "stub-embed", "", tinysql.ModeMemory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = kb.Close() })
	return kb
}

func TestKnowledgeBaseAddDocumentThenSearchFindsIt(t *testing.T) {
	kb := testKnowledgeBase(t)
	if _, err := kb.AddDocument("Wärmepumpen-Handbuch", "Die Wärmepumpe sollte jährlich gewartet werden."); err != nil {
		t.Fatal(err)
	}
	hits, err := kb.Search(context.Background(), "Wärmepumpe", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %d, want 1", len(hits))
	}
	if hits[0].Title != "Wärmepumpen-Handbuch" {
		t.Fatalf("hit title = %q, want the ingested document's title", hits[0].Title)
	}
}

func TestKnowledgeBaseSearchWithNoDocumentsReturnsEmpty(t *testing.T) {
	kb := testKnowledgeBase(t)
	hits, err := kb.Search(context.Background(), "irgendwas", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("hits = %d, want 0 on an empty knowledge base", len(hits))
	}
}

func TestKnowledgeBaseReopensWALDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "knowledge.db")
	open := func() *KnowledgeBase {
		t.Helper()
		kb, err := newKnowledgeBaseFromEmbedder(stubEmbedder{}, "stub-embed", path, tinysql.ModeWAL)
		if err != nil {
			t.Fatal(err)
		}
		return kb
	}
	kb := open()
	_, ingestErr := kb.AddDocument("Wartung", "Die Wärmepumpe wird jährlich gewartet.")
	if err := kb.Close(); err != nil {
		t.Fatal(err)
	}
	if ingestErr != nil {
		t.Fatal(ingestErr)
	}
	kb = open()
	t.Cleanup(func() {
		if err := kb.Close(); err != nil {
			t.Error(err)
		}
	})
	hits, err := kb.Search(t.Context(), "Wärmepumpe", 3)
	if err != nil || len(hits) != 1 || hits[0].Title != "Wartung" {
		t.Fatalf("reopened search = %v, %v, want the persisted Wartung document", hits, err)
	}
}

func TestKnowledgeBaseOwnsLoadedEmbeddingModel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "embedding.gguf")
	if err := testmodel.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	kb, err := OpenKnowledgeBase(path, filepath.Join(dir, "knowledge.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = kb.Close() })
	if kb.embedding == nil || kb.embedModel != "llama" {
		t.Fatalf("knowledge base did not retain the expected model: %q", kb.embedModel)
	}
	if _, err := kb.AddDocument("Licht", "Das Licht im Wohnzimmer ist aus."); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := kb.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

// TestAnswerKnowledgeQueryUsesTheSameSharedPrincipalForIngestAndSearch is a
// regression test for a bug caught during development: AddDocument used to
// pass tinyRAG an empty roles list, which falls back to whatever role
// happens to be globally active in tinyRAG (not sharedPrincipal), so a
// document could be ingested under a different role than Search always
// queries with and never be found.
func TestAnswerKnowledgeQueryUsesTheSameSharedPrincipalForIngestAndSearch(t *testing.T) {
	app := testApp(t)
	app.Knowledge = testKnowledgeBase(t)

	if _, err := app.Knowledge.AddDocument("Urlaubsordnung", "Der Resturlaub verfaellt am 31. Maerz des Folgejahres."); err != nil {
		t.Fatal(err)
	}

	turn, err := app.ProcessText(context.Background(), "conv1", "Was sagen meine Unterlagen über Resturlaub?")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCompleted {
		t.Fatalf("status = %s, want completed (result: %s)", turn.Status, turn.Result)
	}
	if turn.Result == "Dazu wurde nichts in den Dokumenten gefunden." {
		t.Fatalf("expected the ingested document to be found, got: %s", turn.Result)
	}
}

func TestAnswerKnowledgeQueryWithoutKnowledgeBaseConfigured(t *testing.T) {
	app := testApp(t) // Knowledge is nil
	turn, err := app.ProcessText(context.Background(), "conv1", "Was sagen meine Unterlagen über Resturlaub?")
	if err != nil {
		t.Fatal(err)
	}
	if turn.Status != TurnCompleted {
		t.Fatalf("status = %s, want completed (result: %s)", turn.Status, turn.Result)
	}
}
