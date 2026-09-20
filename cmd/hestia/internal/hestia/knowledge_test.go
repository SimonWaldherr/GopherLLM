package hestia

import (
	"context"
	"testing"

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
