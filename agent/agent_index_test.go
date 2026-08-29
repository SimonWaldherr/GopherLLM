package agent

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

func noopGenerator(t *testing.T) gopherllm.ChatGenerator {
	t.Helper()
	return func([]gopherllm.ChatMessage, gopherllm.GenerationOptions, func(string) bool) (gopherllm.GenerationResult, error) {
		return gopherllm.GenerationResult{Text: "ok", FinishReason: "stop"}, nil
	}
}

func TestWithIndexFileSavesOnCloseAndLoadsOnReopen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.ragidx")

	a, err := newAgentWithGenerator(context.Background(), noopGenerator(t), []Option{
		WithIndexFile(path),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.index.Add(context.Background(), rag.Doc{ID: "d1", Text: "content about xylophones"}); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected an index file at %s: %v", path, err)
	}

	reopened, err := newAgentWithGenerator(context.Background(), noopGenerator(t), []Option{
		WithIndexFile(path),
	})
	if err != nil {
		t.Fatal(err)
	}
	if reopened.index.Docs() != 1 {
		t.Fatalf("reopened index has %d docs, want 1", reopened.index.Docs())
	}
	hits, err := reopened.index.Search(context.Background(), "xylophones", rag.Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want the reloaded document to be searchable", hits)
	}
}

func TestWithIndexFileMissingFileStartsEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "does-not-exist-yet.ragidx")
	a, err := newAgentWithGenerator(context.Background(), noopGenerator(t), []Option{WithIndexFile(path)})
	if err != nil {
		t.Fatal(err)
	}
	if a.index.Docs() != 0 {
		t.Fatalf("Docs() = %d, want 0 for a fresh index", a.index.Docs())
	}
}

func TestLearnIndexesADirectory(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("content about aardvarks"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "b.md"), []byte("# Notes\n\ncontent about badgers"), 0o600); err != nil {
		t.Fatal(err)
	}
	a, err := newAgentWithGenerator(context.Background(), noopGenerator(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Learn(context.Background(), dir); err != nil {
		t.Fatal(err)
	}
	if a.index.Docs() != 2 {
		t.Fatalf("Docs() = %d, want 2", a.index.Docs())
	}
}

func TestLearnIndexesASingleFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(path, []byte("content about capybaras"), 0o600); err != nil {
		t.Fatal(err)
	}
	a, err := newAgentWithGenerator(context.Background(), noopGenerator(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Learn(context.Background(), path); err != nil {
		t.Fatal(err)
	}
	hits, err := a.index.Search(context.Background(), "capybaras", rag.Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v", hits)
	}
}

func TestLearnFailsOnMissingPath(t *testing.T) {
	a, err := newAgentWithGenerator(context.Background(), noopGenerator(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Learn(context.Background(), "/does/not/exist/at/all"); err == nil {
		t.Fatal("expected an error for a missing path")
	}
}

func TestWithDocumentsQueuesLearnAtConstruction(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("content about dolphins"), 0o600); err != nil {
		t.Fatal(err)
	}
	a, err := newAgentWithGenerator(context.Background(), noopGenerator(t), []Option{WithDocuments(dir)})
	if err != nil {
		t.Fatal(err)
	}
	if a.index.Docs() != 1 {
		t.Fatalf("Docs() = %d, want 1 (WithDocuments should learn during construction)", a.index.Docs())
	}
}

func TestWithIndexTakesPrecedenceOverWithDocuments(t *testing.T) {
	provided := rag.New(rag.Options{})
	if err := provided.Add(context.Background(), rag.Doc{ID: "pre-existing", Text: "already here"}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("should be ignored"), 0o600); err != nil {
		t.Fatal(err)
	}
	a, err := newAgentWithGenerator(context.Background(), noopGenerator(t), []Option{
		WithIndex(provided), WithDocuments(dir),
	})
	if err != nil {
		t.Fatal(err)
	}
	if a.index != provided {
		t.Fatal("WithIndex should make the Agent use exactly the provided *rag.Index")
	}
}
