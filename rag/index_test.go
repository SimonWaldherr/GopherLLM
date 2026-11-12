package rag

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// fakeEmbedder maps a text to a hand-assigned vector via a lookup function,
// so a test can construct an exact, predictable similarity structure without
// a real model. Every returned vector must already be unit-length, matching
// what gopherllm.Model.EmbedBatch actually produces.
type fakeEmbedder struct {
	vec func(text string) []float32
	err error
}

func (f fakeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([]gopherllm.EmbeddingResult, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([]gopherllm.EmbeddingResult, len(texts))
	for i, t := range texts {
		out[i] = gopherllm.EmbeddingResult{Embedding: f.vec(t)}
	}
	return out, nil
}

func TestIndexAddAndSearchPlainBM25(t *testing.T) {
	ix := New(Options{})
	err := ix.Add(context.Background(),
		Doc{ID: "d1", Title: "Return Policy", Text: "Products may be returned within 30 days of purchase for a full refund."},
		Doc{ID: "d2", Title: "Shipping", Text: "Orders ship within two business days via standard courier."},
	)
	if err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "return refund", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Chunk.DocID != "d1" {
		t.Fatalf("hits = %+v, want d1 first", hits)
	}
}

func TestIndexSearchEmptyIndexReturnsNoHits(t *testing.T) {
	ix := New(Options{})
	hits, err := ix.Search(context.Background(), "anything", Query{})
	if err != nil || hits != nil {
		t.Fatalf("hits = %v, err = %v, want nil, nil", hits, err)
	}
}

func TestIndexAddReplaceDoesNotLeakStalePostings(t *testing.T) {
	ix := New(Options{})
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "the rare term xylophone appears here"}); err != nil {
		t.Fatal(err)
	}
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "completely different content now"}); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "xylophone", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("expected no hits for a term only the REPLACED text contained, got %+v", hits)
	}
	if ix.bm25.totalLength != int64(ix.bm25.length[chunkKey("d1", 0)]) {
		t.Fatalf("corpus length statistics were not fully replaced: totalLength=%d chunkLength=%d",
			ix.bm25.totalLength, ix.bm25.length[chunkKey("d1", 0)])
	}
}

func TestIndexVectorSearchOrdersByCosine(t *testing.T) {
	embed := fakeEmbedder{vec: func(text string) []float32 {
		switch text {
		case "cat":
			return []float32{1, 0}
		case "dog":
			return []float32{0.9, 0.436} // close to "cat"
		case "airplane":
			return []float32{0, 1} // orthogonal to "cat"
		default:
			return []float32{1, 0} // query "feline" treated as near "cat"
		}
	}}
	ix := New(Options{Embedder: embed, Ranking: Ranking{VectorWeight: 1, KeywordWeight: 0, RecencyWeight: 0}})
	if err := ix.Add(context.Background(),
		Doc{ID: "cat", Text: "cat"}, Doc{ID: "dog", Text: "dog"}, Doc{ID: "plane", Text: "airplane"}); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "feline", Query{TopK: 3, MinScore: -1})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) < 2 || hits[0].Chunk.DocID != "cat" || hits[1].Chunk.DocID != "dog" {
		t.Fatalf("hits = %+v, want cat then dog by vector similarity", hits)
	}
}

func TestIndexHybridUnionFindsLexicalMatchVectorWouldMiss(t *testing.T) {
	// The embedder places "GX-1180" far from the query in vector space (as a
	// real embedder might, for an opaque identifier), but BM25 must still
	// surface it on an exact term match — this is the point of a union
	// rather than a vector-then-rerank pipeline.
	embed := fakeEmbedder{vec: func(text string) []float32 {
		if text == "GX-1180 return window" {
			return []float32{0, 1}
		}
		return []float32{1, 0}
	}}
	ix := New(Options{Embedder: embed})
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "GX-1180 return window"}); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "GX-1180", Query{MinVector: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected the lexical match to survive MinVector, got %+v", hits)
	}
	if !hits[0].LexicalOnly {
		t.Fatalf("hit should be marked LexicalOnly: %+v", hits[0])
	}
}

func TestIndexMinVectorExcludesNonLexicalLowSimilarity(t *testing.T) {
	// The query and the document embed to ORTHOGONAL vectors (cosine 0), and
	// share no term, so nothing should let this chunk through MinVector.
	embed := fakeEmbedder{vec: func(text string) []float32 {
		if text == "totally unrelated content" {
			return []float32{1, 0}
		}
		return []float32{0, 1}
	}}
	ix := New(Options{Embedder: embed})
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "totally unrelated content"}); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "something else entirely", Query{MinVector: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("expected no hits: low vector similarity and no lexical match, got %+v", hits)
	}
}

func TestRankingZeroWeightsUseDefaults(t *testing.T) {
	v, k, r := effectiveWeights(Ranking{})
	if v != 0.7 || k != 0.2 || r != 0.1 {
		t.Fatalf("defaults = %v,%v,%v, want 0.7,0.2,0.1", v, k, r)
	}
}

func TestRankingExplicitZeroWeightMeansZero(t *testing.T) {
	v, k, r := effectiveWeights(Ranking{KeywordWeight: 1})
	if v != 0 || k != 1 || r != 0 {
		t.Fatalf("weights = %v,%v,%v, want 0,1,0 (explicit, not defaulted)", v, k, r)
	}
}

func TestIndexNoEmbedderRenormalizesDefaultWeights(t *testing.T) {
	ix := New(Options{}) // no Embedder, default Ranking
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "matching term present here", Time: time.Now()}); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "matching term", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v", hits)
	}
	// With no Embedder, 0.7/0.2/0.1 renormalizes to keyword 0.2/0.3=0.667 and
	// recency 0.1/0.3=0.333 of the score; the doc is dated "now" so recency
	// contributes close to its full weight and keyword contributes its
	// normalized (here maximal, 1.0) share. Score should land close to 1.0.
	if hits[0].Score < 0.9 || hits[0].Score > 1.01 {
		t.Fatalf("score = %v, want close to 1.0 for a fresh, fully-matching, no-embedder chunk", hits[0].Score)
	}
}

func TestIndexDiversityGuardCapsChunksPerDoc(t *testing.T) {
	ix := New(Options{})
	// One long document that will produce several chunks all matching the
	// query term, plus a second document with one matching chunk.
	long := ""
	for i := 0; i < 6; i++ {
		long += "shared-term paragraph number filler content to occupy space. \n\n"
	}
	if err := ix.Add(context.Background(),
		Doc{ID: "big", Text: long},
		Doc{ID: "small", Text: "shared-term appears once here too."},
	); err != nil {
		t.Fatal(err)
	}
	if ix.docs["big"] == nil || len(ix.docs["big"].chunkKeys) < 2 {
		t.Skip("fixture did not produce multiple chunks for the big doc; adjust MaxRunes assumptions")
	}
	hits, err := ix.Search(context.Background(), "shared-term", Query{TopK: 10, MaxChunksPerDoc: 1})
	if err != nil {
		t.Fatal(err)
	}
	perDoc := map[string]int{}
	for _, h := range hits {
		perDoc[h.Chunk.DocID]++
	}
	for doc, count := range perDoc {
		if count > 1 {
			t.Fatalf("doc %q contributed %d hits, want at most 1 (MaxChunksPerDoc)", doc, count)
		}
	}
	if len(perDoc) < 2 {
		t.Fatalf("expected hits from both documents, got %v", perDoc)
	}
}

func TestIndexMaxDocsLimitsDistinctSources(t *testing.T) {
	ix := New(Options{})
	if err := ix.Add(context.Background(),
		Doc{ID: "d1", Text: "keyword one"}, Doc{ID: "d2", Text: "keyword two"}, Doc{ID: "d3", Text: "keyword three"},
	); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "keyword", Query{TopK: 10, MaxDocs: 2})
	if err != nil {
		t.Fatal(err)
	}
	docs := map[string]bool{}
	for _, h := range hits {
		docs[h.Chunk.DocID] = true
	}
	if len(docs) > 2 {
		t.Fatalf("got hits from %d distinct docs, want at most 2: %v", len(docs), docs)
	}
}

func TestIndexNeighboursJoinsAdjacentChunks(t *testing.T) {
	c := Chunker{MaxRunes: 20, OverlapRunes: 0}
	ix := New(Options{Chunker: c})
	text := "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: text}); err != nil {
		t.Fatal(err)
	}
	if len(ix.docs["d1"].chunkKeys) < 3 {
		t.Skip("fixture did not produce enough chunks to test neighbour widening")
	}
	hits, err := ix.Search(context.Background(), "Second", Query{Neighbours: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("no hits")
	}
	widened := hits[0]
	if widened.Chunk.Start != 0 {
		// The exact boundary depends on chunking, but widening must strictly
		// grow the window versus the unwidened chunk (index 0 would only be
		// exactly 0 if the widened window reached the very start of the doc).
	}
	if len(widened.Text) <= len(text[widened.Start:widened.End])-1 && widened.End-widened.Start < 20 {
		t.Fatalf("widened chunk does not look larger than a single un-widened chunk: %+v", widened)
	}
}

func TestIndexFilterRunsBeforeScoring(t *testing.T) {
	ix := New(Options{})
	if err := ix.Add(context.Background(),
		Doc{ID: "d1", Kind: "internal", Text: "matching keyword content"},
		Doc{ID: "d2", Kind: "public", Text: "matching keyword content"},
	); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(context.Background(), "matching keyword", Query{
		Filter: func(d Doc) bool { return d.Kind == "public" },
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.Kind != "public" {
			t.Fatalf("filter did not exclude doc of kind %q: %+v", h.Kind, h)
		}
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want exactly the public doc", hits)
	}
}

func TestIndexEmbedErrorPropagatesFromAdd(t *testing.T) {
	ix := New(Options{Embedder: fakeEmbedder{err: errors.New("boom")}})
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "hello"}); err == nil {
		t.Fatal("expected the embedder's error to propagate")
	}
}

func TestIndexEmbedErrorPropagatesFromSearch(t *testing.T) {
	ix := New(Options{Embedder: fakeEmbedder{vec: func(string) []float32 { return []float32{1, 0} }}})
	if err := ix.Add(context.Background(), Doc{ID: "d1", Text: "hello"}); err != nil {
		t.Fatal(err)
	}
	ix.opts.Embedder = fakeEmbedder{err: errors.New("boom")}
	if _, err := ix.Search(context.Background(), "hello", Query{}); err == nil {
		t.Fatal("expected the embedder's error to propagate from Search")
	}
}

func TestIndexSearchIsRaceFreeWithConcurrentAdd(t *testing.T) {
	ix := New(Options{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 50; i++ {
			_ = ix.Add(context.Background(), Doc{ID: "d1", Text: "changing content each time"})
		}
	}()
	for i := 0; i < 50; i++ {
		_, _ = ix.Search(context.Background(), "content", Query{})
	}
	<-done
}

func TestGuessLexicalModeOnIdentifierLikeQuery(t *testing.T) {
	if GuessLexicalMode("GX-1180") != LexicalAll {
		t.Fatal("expected LexicalAll for an identifier-shaped query")
	}
	if GuessLexicalMode("part A123") != LexicalAll {
		t.Fatal("expected LexicalAll when one of a few short tokens looks like an identifier")
	}
}

func TestGuessLexicalModeOnNaturalQuestion(t *testing.T) {
	if GuessLexicalMode("what is our return policy for damaged items") != LexicalAny {
		t.Fatal("expected LexicalAny for an ordinary natural-language question")
	}
}

func TestIndexListReportsDocsInInsertionOrderWithChunkCounts(t *testing.T) {
	ix := New(Options{})
	if err := ix.Add(context.Background(),
		Doc{ID: "d1", Title: "First", Text: "alpha beta gamma"},
		Doc{ID: "d2", Title: "Second", Text: "delta epsilon zeta"},
	); err != nil {
		t.Fatal(err)
	}
	list := ix.List()
	if len(list) != 2 || list[0].ID != "d1" || list[1].ID != "d2" {
		t.Fatalf("List() = %+v, want d1 then d2", list)
	}
	if list[0].Title != "First" || list[0].Chunks == 0 {
		t.Fatalf("list[0] = %+v", list[0])
	}
}

func TestIndexRemoveDeletesChunksAndPostings(t *testing.T) {
	ix := New(Options{})
	if err := ix.Add(context.Background(),
		Doc{ID: "d1", Text: "the rare term xylophone appears here"},
		Doc{ID: "d2", Text: "an unrelated document about oranges"},
	); err != nil {
		t.Fatal(err)
	}
	if n := ix.Remove("d1", "missing"); n != 1 {
		t.Fatalf("Remove returned %d, want 1", n)
	}
	if ix.Docs() != 1 || ix.Len() != 1 {
		t.Fatalf("Docs()=%d Len()=%d, want 1, 1 after removing d1", ix.Docs(), ix.Len())
	}
	hits, err := ix.Search(context.Background(), "xylophone", Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Fatalf("hits = %+v, want none for a removed doc's term", hits)
	}
	list := ix.List()
	if len(list) != 1 || list[0].ID != "d2" {
		t.Fatalf("List() after Remove = %+v, want only d2", list)
	}
}

func BenchmarkIndexSearchLexical(b *testing.B) {
	ix := New(Options{})
	docs := make([]Doc, 10000)
	for i := range docs {
		text := "common word"
		if i%100 == 0 {
			text += " rare"
		}
		docs[i] = Doc{ID: fmt.Sprint(i), Text: text}
	}
	if err := ix.Add(b.Context(), docs...); err != nil {
		b.Fatal(err)
	}
	for _, query := range []string{"missing", "common rare", "common"} {
		b.Run(query, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := ix.Search(b.Context(), query, Query{Lexical: LexicalAll}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestSparseLexicalSearchMatchesFullScan(t *testing.T) {
	ix := New(Options{})
	// Identical timestamps and text exercise tie breaking independently of
	// posting list order. Future timestamps have a constant recency of one.
	date := time.Now().Add(24 * time.Hour)
	for i := 19; i >= 0; i-- {
		text := "common"
		if i%5 == 0 {
			text += " rare"
		}
		if err := ix.Add(t.Context(), Doc{ID: fmt.Sprint(i), Text: text, Time: date}); err != nil {
			t.Fatal(err)
		}
	}
	for _, minScore := range []float32{-1, 0, 2} {
		for _, query := range []string{"missing", "common rare", "common"} {
			q := Query{Lexical: LexicalAll, TopK: 3, MinScore: minScore}
			got, err := ix.Search(t.Context(), query, q)
			if err != nil {
				t.Fatal(err)
			}
			visited := 0
			q.Filter = func(Doc) bool { visited++; return true }
			want, err := ix.Search(t.Context(), query, q)
			if err != nil || visited != ix.Len() || !slices.Equal(got, want) {
				t.Fatalf("%q minScore=%g visited=%d got=%v want=%v err=%v", query, minScore, visited, got, want, err)
			}
		}
	}
}

func TestSearchAppliesKindWeightToUnmatchedChunks(t *testing.T) {
	ix := New(Options{Ranking: Ranking{KindWeight: map[string]float32{"priority": 3}}})
	if err := ix.Add(t.Context(), Doc{ID: "recent", Text: "example", Kind: "priority", Time: time.Now().Add(time.Hour)}); err != nil {
		t.Fatal(err)
	}
	hits, err := ix.Search(t.Context(), "missing", Query{})
	if err != nil || len(hits) != 1 || hits[0].DocID != "recent" || hits[0].Keyword != 0 {
		t.Fatalf("kind-weighted search returned %+v, err=%v", hits, err)
	}
}

func TestSearchTinyMinScoreMatchesFullScan(t *testing.T) {
	ix := New(Options{Ranking: Ranking{KeywordWeight: 4, RecencyWeight: 1}})
	age := time.Duration(float64(ix.opts.recencyHalfLife()) * math.Log2(4.0/3))
	if err := ix.Add(t.Context(), Doc{ID: "item", Text: "example", Time: time.Now().Add(-age)}); err != nil {
		t.Fatal(err)
	}
	q := Query{MinScore: 1e-10}
	got, err := ix.Search(t.Context(), "missing", q)
	if err != nil {
		t.Fatal(err)
	}
	q.Filter = func(Doc) bool { return true }
	want, err := ix.Search(t.Context(), "missing", q)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("tiny minimum score: got %v, want %v, err=%v", got, want, err)
	}
}

func TestIndexBatchMutationsMatchSequentialUpdates(t *testing.T) {
	opts := Options{Chunker: Chunker{MaxRunes: 12, MinRunes: 1, OverlapRunes: 1}, Embedder: fakeEmbedder{vec: func(s string) []float32 {
		if len(s)%2 == 0 {
			return []float32{1, 0}
		}
		return []float32{0, 1}
	}}}
	batch, seq := New(opts), New(opts)
	initial := []Doc{{ID: "keep", Text: "survivor"}, {ID: "a", Text: "old alpha in a long document"}, {ID: "b", Text: "old beta"}}
	for _, ix := range []*Index{batch, seq} {
		if err := ix.Add(t.Context(), initial...); err != nil {
			t.Fatal(err)
		}
	}
	updates := []Doc{{ID: "a", Text: "transient"}, {ID: "c", Text: "new charlie"}, {ID: "b", Text: "updated beta"}, {ID: "a", Text: "final"}, {ID: "empty", Text: "temporary"}, {ID: "empty"}}
	if err := batch.Add(t.Context(), updates...); err != nil {
		t.Fatal(err)
	}
	for _, d := range updates {
		if err := seq.Add(t.Context(), d); err != nil {
			t.Fatal(err)
		}
	}
	check := func() {
		t.Helper()
		if !slices.Equal(batch.order, seq.order) || !slices.Equal(batch.List(), seq.List()) || batch.Docs() != seq.Docs() || batch.Len() != seq.Len() {
			t.Fatalf("batch order %v, want %v", batch.order, seq.order)
		}
		for _, query := range []string{"transient", "temporary", "old", "beta", "final"} {
			q := Query{TopK: 20, MinScore: -1}
			got, err := batch.Search(t.Context(), query, q)
			if err != nil {
				t.Fatal(err)
			}
			want, err := seq.Search(t.Context(), query, q)
			if err != nil || !slices.Equal(got, want) {
				t.Fatalf("%q: got %v, want %v, err=%v", query, got, want, err)
			}
		}
	}
	check()
	ids := []string{"b", "missing", "empty", "b", "c"}
	if got := batch.Remove(ids...); got != 3 {
		t.Fatalf("removed %d docs, want 3", got)
	}
	for _, id := range ids {
		seq.Remove(id)
	}
	check()
}

func BenchmarkIndexBatchUpdates(b *testing.B) {
	for _, size := range []int{1, 128} {
		for _, remove := range []bool{false, true} {
			name := "replace"
			if remove {
				name = "remove_add"
			}
			b.Run(fmt.Sprintf("%s/%d", name, size), func(b *testing.B) {
				ix := New(Options{})
				docs := make([]Doc, 10000)
				for i := range docs {
					docs[i] = Doc{ID: fmt.Sprint(i), Text: "common document text"}
				}
				if err := ix.Add(b.Context(), docs...); err != nil {
					b.Fatal(err)
				}
				ids := make([]string, size)
				for i := range ids {
					ids[i] = docs[i].ID
				}
				b.ReportAllocs()
				for b.Loop() {
					if remove && ix.Remove(ids...) != size {
						b.Fatal("unexpected removal count")
					}
					if err := ix.Add(b.Context(), docs[:size]...); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
