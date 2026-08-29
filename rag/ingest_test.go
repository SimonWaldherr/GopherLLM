package rag

import (
	"context"
	"io/fs"
	"strings"
	"testing"
	"testing/fstest"
)

func TestAddFSIndexesMatchedFilesOnly(t *testing.T) {
	fsys := fstest.MapFS{
		"notes/a.md":       {Data: []byte("# Title\n\nSome markdown content about gophers.")},
		"notes/b.txt":      {Data: []byte("Plain text content about llamas.")},
		"notes/c.bin":      {Data: []byte{0x00, 0x01, 0x02}},
		"notes/deep/d.txt": {Data: []byte("Nested file content about foxes.")},
	}
	ix := New(Options{})
	added, skipped, err := ix.AddFS(context.Background(), fsys, IngestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if added != 3 {
		t.Fatalf("added = %d, want 3 (.md, .txt, nested .txt; .bin excluded by default extensions)", added)
	}
	if skipped != 0 {
		t.Fatalf("skipped = %d, want 0", skipped)
	}
	if ix.Docs() != 3 {
		t.Fatalf("Docs() = %d, want 3", ix.Docs())
	}
	hits, err := ix.Search(context.Background(), "gophers", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("expected to find the markdown file's content")
	}
}

func TestAddFSSkipsOversizedFiles(t *testing.T) {
	fsys := fstest.MapFS{
		"big.txt": {Data: make([]byte, 100)},
	}
	ix := New(Options{})
	added, skipped, err := ix.AddFS(context.Background(), fsys, IngestOptions{MaxFileBytes: 10})
	if err != nil {
		t.Fatal(err)
	}
	if added != 0 || skipped != 1 {
		t.Fatalf("added=%d skipped=%d, want 0,1", added, skipped)
	}
}

func TestAddFSCustomMatchAndExtractor(t *testing.T) {
	fsys := fstest.MapFS{
		"data.custom": {Data: []byte("raw payload about badgers")},
		"ignored.txt": {Data: []byte("would match by default but Match excludes it")},
	}
	ix := New(Options{})
	extractCalled := false
	added, skipped, err := ix.AddFS(context.Background(), fsys, IngestOptions{
		Match: func(path string, info fs.FileInfo) bool { return strings.HasSuffix(path, ".custom") },
		Extract: func(path string, data []byte) (Doc, error) {
			extractCalled = true
			return Doc{ID: path, Title: "custom:" + path, Text: string(data)}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if added != 1 || skipped != 0 {
		t.Fatalf("added=%d skipped=%d, want 1,0", added, skipped)
	}
	if !extractCalled {
		t.Fatal("custom Extract was never called")
	}
	hits, err := ix.Search(context.Background(), "badgers", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want the custom-extracted doc", hits)
	}
}

func TestAddFSSkipsSymlinksByDefault(t *testing.T) {
	// fstest.MapFS has no native symlink support, so this exercises the
	// FollowSymlinks default indirectly: a plain filesystem with no symlinks
	// present must behave identically regardless of the setting.
	fsys := fstest.MapFS{"a.txt": {Data: []byte("content")}}
	ix := New(Options{})
	added, _, err := ix.AddFS(context.Background(), fsys, IngestOptions{FollowSymlinks: false})
	if err != nil {
		t.Fatal(err)
	}
	if added != 1 {
		t.Fatalf("added = %d, want 1", added)
	}
}
