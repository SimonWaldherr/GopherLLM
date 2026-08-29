package rag

import "testing"

func TestTokenizeSplitsAndJoinsIdentifiers(t *testing.T) {
	got := tokenize("Order GX-1180 shipped")
	want := map[string]bool{"order": true, "gx": true, "1180": true, "gx-1180": true, "shipped": true}
	if len(got) != len(want) {
		t.Fatalf("got %v, want exactly %v", got, want)
	}
	for _, term := range got {
		if !want[term] {
			t.Fatalf("unexpected term %q in %v", term, got)
		}
	}
}

func TestTokenizeCJKIsOneTerm(t *testing.T) {
	// Documents, does not fix, the non-goal: an unbroken CJK run has no
	// internal ASCII boundary for this tokenizer to split on. If this ever
	// changes it must be a deliberate, visible diff to this test.
	got := tokenize("日本語のテキスト")
	if len(got) != 1 || got[0] != "日本語のテキスト" {
		t.Fatalf("got %v, want one term covering the whole run", got)
	}
}

func TestTokenizeLowercasesAndSplitsOnPunctuation(t *testing.T) {
	got := tokenize("Hello, World! Testing 123.")
	want := []string{"hello", "world", "testing", "123"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func newTestBM25(docs map[string]string) *bm25Index {
	b := newBM25Index(nil)
	for key, text := range docs {
		b.add(key, text)
	}
	return b
}

func TestBM25FavorsExactTermMatch(t *testing.T) {
	b := newTestBM25(map[string]string{
		"a": "the quick brown fox jumps over the lazy dog",
		"b": "a completely unrelated sentence about cooking pasta",
	})
	scores := b.scores("fox", LexicalAny)
	if scores["a"] <= 0 {
		t.Fatalf("expected a positive score for the matching doc, got %v", scores)
	}
	if _, ok := scores["b"]; ok {
		t.Fatalf("non-matching doc should not appear in scores: %v", scores)
	}
}

func TestBM25LexicalAllRequiresEveryTerm(t *testing.T) {
	b := newTestBM25(map[string]string{
		"both":     "alpha beta",
		"only-one": "alpha gamma",
	})
	any := b.scores("alpha beta", LexicalAny)
	if len(any) != 2 {
		t.Fatalf("LexicalAny: got %d matches, want 2: %v", len(any), any)
	}
	all := b.scores("alpha beta", LexicalAll)
	if len(all) != 1 || all["both"] <= 0 {
		t.Fatalf("LexicalAll: got %v, want only \"both\"", all)
	}
}

func TestBM25TitleTermsAreIndexedRegardlessOfTitlePrefix(t *testing.T) {
	ix := New(Options{})
	if err := ix.Add(t.Context(), Doc{ID: "d1", Title: "Xylophone Handbook", Text: "This document has nothing to do with the title term."}); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(t.Context(), "xylophone", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("expected the title term to be searchable even though TitlePrefix is false and the body never mentions it")
	}
}

func TestBM25RemoveDeletesEmptyPostingLists(t *testing.T) {
	b := newBM25Index(nil)
	b.add("a", "unique-term-xyz shared")
	b.add("b", "shared")
	if b.df["unique-term-xyz"] != 1 {
		t.Fatalf("df = %d, want 1", b.df["unique-term-xyz"])
	}
	b.remove("a")
	if _, exists := b.postings["unique-term-xyz"]; exists {
		t.Fatal("posting list for a term with no remaining chunks must be deleted, not left empty")
	}
	if _, exists := b.df["unique-term-xyz"]; exists {
		t.Fatal("df entry for a fully-removed term must be deleted")
	}
	if b.df["shared"] != 1 {
		t.Fatalf("df[shared] = %d, want 1 (only b remains)", b.df["shared"])
	}
}
