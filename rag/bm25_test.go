package rag

import (
	"fmt"
	"maps"
	"strings"
	"testing"
)

// Score each chunk independently as a reference for posting-list traversal.
func referenceBM25Scores(b *bm25Index, query string, mode LexicalMode) map[string]float32 {
	var terms []string
	seen := map[string]bool{}
	for _, term := range tokenize(query) {
		if !b.stopwords[term] && !seen[term] {
			terms = append(terms, term)
			seen[term] = true
		}
	}
	out := map[string]float32{}
	for key, length := range b.length {
		var score float32
		matched := 0
		for _, term := range terms {
			f := b.postings[term][key]
			if f == 0 {
				continue
			}
			matched++
			denom := float64(f) + bm25K1*(1-bm25B+bm25B*float64(length)/b.avgdl())
			score += float32(bm25IDF(len(b.length), b.df[term]) * (float64(f) * (bm25K1 + 1)) / denom)
		}
		if matched > 0 && (mode != LexicalAll || matched == len(terms)) {
			out[key] = score
		}
	}
	return out
}

func TestBM25ScoresMatchReference(t *testing.T) {
	b := newBM25Index(map[string]bool{"the": true})
	for i := 0; i < 100; i++ {
		text := "common " + strings.Repeat("alpha ", i%5)
		if i%7 == 0 {
			text += "rare GX-1180 日本語"
		}
		b.add(fmt.Sprint(i), text)
	}
	b.add("empty", "the")
	queries := []string{"", "the", "missing", "common", "alpha alpha the", "common rare", "rare common", "common missing", "missing common", "alpha rare common", "rare replacement", "GX-1180", "日本語 alpha"}
	check := func(stage string) {
		for _, query := range queries {
			for _, mode := range []LexicalMode{LexicalAny, LexicalAll} {
				got, want := b.scores(query, mode), referenceBM25Scores(b, query, mode)
				if !maps.Equal(got, want) {
					t.Fatalf("%s query=%q mode=%d: got %v, want %v", stage, query, mode, got, want)
				}
			}
		}
	}
	check("initial")
	b.remove("0")
	b.remove("7")
	b.add("7", "common replacement")
	check("replaced")
	b = newBM25Index(nil)
	check("empty")
}

func BenchmarkBM25Scores(b *testing.B) {
	ix := newBM25Index(nil)
	for i := 0; i < 10000; i++ {
		text := "common alpha beta document text"
		if i%100 == 0 {
			text += " rare"
		}
		ix.add(fmt.Sprint(i), text)
	}
	for _, tc := range []struct {
		name, query string
		mode        LexicalMode
	}{
		{"any", "common alpha rare", LexicalAny},
		{"all_selective", "common alpha rare", LexicalAll},
		{"all_common", "common alpha", LexicalAll},
		{"all_missing", "common missing", LexicalAll},
		{"all_single", "common", LexicalAll},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				ix.scores(tc.query, tc.mode)
			}
		})
	}
}

func BenchmarkBM25Replace(b *testing.B) {
	ix := newBM25Index(nil)
	for i := 0; i < 10000; i++ {
		ix.add(fmt.Sprint(i), fmt.Sprintf("common document unique%d", i))
	}
	b.ReportAllocs()
	for b.Loop() {
		ix.remove("0")
		ix.add("0", "common document unique0")
	}
}

func TestBM25RemoveMatchesFreshIndex(t *testing.T) {
	b := newTestBM25(map[string]string{"a": "shared shared unique", "b": "shared remaining", "empty": ""})
	b.remove("a")
	b.remove("a")
	b.remove("empty")
	b.remove("absent")
	want := newTestBM25(map[string]string{"b": "shared remaining"})
	if len(b.terms) != 1 || len(b.terms["b"]) != 2 {
		t.Fatalf("removal left stale reverse postings: %v", b.terms)
	}
	if b.totalLength != want.totalLength || !maps.Equal(b.length, want.length) || !maps.Equal(b.df, want.df) || !maps.EqualFunc(b.postings, want.postings, func(a, b map[string]int) bool { return maps.Equal(a, b) }) {
		t.Fatalf("removal left stale corpus statistics or postings: %+v", b)
	}
}

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
