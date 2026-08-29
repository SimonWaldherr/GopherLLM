package rag

import (
	"context"
	"testing"
)

// goldenCorpus is fixed so a ranking-formula change (a tokenizer tweak, a
// weight default, an IDF variant) shows up as an explicit, reviewed diff to
// this test rather than silently retuning search quality for every user of
// the package.
func goldenCorpus() []Doc {
	return []Doc{
		{ID: "return-policy", Title: "Return Policy", Text: "Products may be returned within 30 days of purchase for a full refund. Items must be in original condition."},
		{ID: "shipping", Title: "Shipping Information", Text: "Orders ship within two business days via standard courier. Express shipping is available at checkout."},
		{ID: "warranty", Title: "Warranty Terms", Text: "All products carry a one-year limited warranty covering manufacturing defects. This does not cover accidental damage."},
		{ID: "gx1180", Title: "GX-1180 Specification", Text: "The GX-1180 unit operates at 240V and includes a 30-day return window separate from the standard policy."},
	}
}

func TestRankingGoldenCorpusTopKIsPinned(t *testing.T) {
	ix := New(Options{})
	if err := ix.Add(context.Background(), goldenCorpus()...); err != nil {
		t.Fatal(err)
	}

	// MinScore is disabled (-1) so this test pins RANKING ORDER, which is what
	// a golden-ranking test should protect against an accidental formula
	// change — not the absolute score threshold, which is a separate,
	// deliberately tunable cutoff already covered by the MinScore tests in
	// index_test.go.
	cases := []struct {
		query string
		want  []string
	}{
		{"return refund", []string{"return-policy", "gx1180"}},
		{"warranty defects", []string{"warranty"}},
		{"GX-1180", []string{"gx1180"}},
		{"shipping courier", []string{"shipping"}},
	}
	for _, c := range cases {
		hits, err := ix.Search(context.Background(), c.query, Query{TopK: len(c.want), MinScore: -1})
		if err != nil {
			t.Fatalf("query %q: %v", c.query, err)
		}
		if len(hits) != len(c.want) {
			t.Fatalf("query %q: got %d hits, want %d: %+v", c.query, len(hits), len(c.want), hits)
		}
		for i, want := range c.want {
			if hits[i].Chunk.DocID != want {
				t.Fatalf("query %q: hit %d = %q, want %q (full: %+v)", c.query, i, hits[i].Chunk.DocID, want, hits)
			}
		}
	}
}

// TestRankingGoldenCorpusReplaceDropsOldTermsFromRanking is the retune-safety
// companion to the pinned corpus above: replacing a document's text must
// remove its old unique terms from every later ranking, not just stop
// returning the document itself for its OWN old terms (that would already be
// caught by TestIndexAddReplaceDoesNotLeakStalePostings) — this asserts the
// corpus-wide statistics (avgdl, df) an unrelated query depends on are also
// unaffected by the ghost of the old text.
func TestRankingGoldenCorpusReplaceDropsOldTermsFromRanking(t *testing.T) {
	ix := New(Options{})
	docs := goldenCorpus()
	if err := ix.Add(context.Background(), docs...); err != nil {
		t.Fatal(err)
	}
	before, err := ix.Search(context.Background(), "warranty defects", Query{TopK: 1})
	if err != nil {
		t.Fatal(err)
	}

	// Replace the return-policy doc with unrelated text AND title: BM25
	// indexes the title alongside the body (see bm25.go), so keeping the old
	// title "Return Policy" would make "return refund" keep matching for a
	// legitimate reason unrelated to the bug this test targets.
	if err := ix.Add(context.Background(), Doc{ID: "return-policy", Title: "Unrelated", Text: "This content no longer discusses the old subject at all."}); err != nil {
		t.Fatal(err)
	}

	after, err := ix.Search(context.Background(), "warranty defects", Query{TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(before) != 1 || len(after) != 1 || before[0].Chunk.DocID != after[0].Chunk.DocID {
		t.Fatalf("an unrelated query's top hit changed after an unrelated replace: before=%+v after=%+v", before, after)
	}

	stillFinds, err := ix.Search(context.Background(), "return refund", Query{})
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range stillFinds {
		if h.Chunk.DocID == "return-policy" {
			t.Fatalf("replaced doc's old content still matches its old query: %+v", h)
		}
	}
}
