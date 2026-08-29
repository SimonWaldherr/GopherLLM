package rag

import (
	"math"
	"strings"
	"unicode"
)

// tokenize splits s into BM25 terms, lowercased. In addition to the ordinary
// split on any rune that is neither a letter nor a digit, a hyphen,
// underscore, dot or slash that sits BETWEEN two alphanumeric runs is also
// treated as part of the token, so "GX-1180" indexes as both ["gx","1180"]
// (recall: matches a query for either half) AND "gx-1180" (precision: an
// exact-identifier query scores this chunk specifically, not any chunk that
// merely contains "1180" somewhere unrelated). This is the one cheap,
// well-justified departure from a pure letter/digit split.
//
// Not fixed here, and not pretended to be: an unbroken run of CJK characters
// (unicode.IsLetter is true for Han/Hiragana/Katakana/Hangul, and nothing in
// this run breaks it) becomes a single term. See the package doc's non-goals.
func tokenize(s string) []string {
	s = strings.ToLower(s)
	r := []rune(s)
	var out []string
	isJoiner := func(c rune) bool { return c == '-' || c == '_' || c == '.' || c == '/' }
	isWord := func(c rune) bool { return unicode.IsLetter(c) || unicode.IsDigit(c) }
	i := 0
	for i < len(r) {
		if !isWord(r[i]) {
			i++
			continue
		}
		start := i
		for i < len(r) && isWord(r[i]) {
			i++
		}
		out = append(out, string(r[start:i]))
		// Look ahead for a joiner immediately followed by another word run,
		// e.g. "-1180" after "gx", and emit the whole joined span as one
		// additional term without consuming it — the plain word run after
		// the joiner is still tokenized normally on the next outer iteration.
		if i < len(r) && isJoiner(r[i]) {
			j := i + 1
			for j < len(r) && (isJoiner(r[j]) || isWord(r[j])) {
				j++
			}
			// Only worth emitting if it actually reaches another word run
			// (not just a trailing joiner with nothing after it) and it adds
			// something beyond the plain word run already emitted.
			if j > i+1 && isWord(r[j-1]) {
				out = append(out, string(r[start:j]))
			}
		}
	}
	return out
}

// bm25Params are the standard Okapi BM25 constants. They are not exposed as a
// tunable: this package's aim is a solid default, not a search-relevance lab.
const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// bm25Index is the inverted index and corpus statistics BM25 scoring needs.
// It is embedded in Index rather than exported: a caller interacts with it
// only through Index.Search.
type bm25Index struct {
	// postings maps a term to the chunk keys containing it and their term
	// frequency within that chunk.
	postings map[string]map[string]int
	// df is document frequency: how many chunks contain each term at least
	// once.
	df map[string]int
	// length is each chunk's token count (after tokenize, including the
	// title terms folded in per chunk — see Index.Add).
	length      map[string]int
	totalLength int64
	stopwords   map[string]bool
}

func newBM25Index(stopwords map[string]bool) *bm25Index {
	return &bm25Index{
		postings:  map[string]map[string]int{},
		df:        map[string]int{},
		length:    map[string]int{},
		stopwords: stopwords,
	}
}

func (b *bm25Index) avgdl() float64 {
	if len(b.length) == 0 {
		return 0
	}
	return float64(b.totalLength) / float64(len(b.length))
}

// add indexes one chunk's terms under key. text is the term source — the
// caller folds in doc.Title alongside the chunk's own text before calling
// this, per bm25.go's package-level doc on title indexing.
func (b *bm25Index) add(key, text string) {
	freq := map[string]int{}
	total := 0
	for _, term := range tokenize(text) {
		if b.stopwords[term] {
			continue
		}
		freq[term]++
		total++
	}
	for term, f := range freq {
		if b.postings[term] == nil {
			b.postings[term] = map[string]int{}
		}
		if _, exists := b.postings[term][key]; !exists {
			b.df[term]++
		}
		b.postings[term][key] = f
	}
	b.length[key] = total
	b.totalLength += int64(total)
}

// remove deletes key's postings and length entry, and deletes any term whose
// posting list becomes empty as a result (rather than leaving a phantom
// zero-df entry). Called before Index.Add reindexes a Doc.ID that already
// exists, so a replace never leaks stale postings or corpus statistics into
// later rankings.
func (b *bm25Index) remove(key string) {
	length, ok := b.length[key]
	if !ok {
		return
	}
	for term, chunks := range b.postings {
		if _, has := chunks[key]; !has {
			continue
		}
		delete(chunks, key)
		b.df[term]--
		if len(chunks) == 0 {
			delete(b.postings, term)
			delete(b.df, term)
		}
	}
	delete(b.length, key)
	b.totalLength -= int64(length)
}

// scores returns the BM25 score for every chunk key containing at least one
// query term. mode controls whether a chunk must contain ALL query terms
// (LexicalAll) or just one (LexicalAny).
func (b *bm25Index) scores(query string, mode LexicalMode) map[string]float32 {
	terms := make([]string, 0)
	seen := map[string]bool{}
	for _, term := range tokenize(query) {
		if b.stopwords[term] || seen[term] {
			continue
		}
		seen[term] = true
		terms = append(terms, term)
	}
	if len(terms) == 0 {
		return nil
	}
	n := len(b.length)
	if n == 0 {
		return nil
	}
	avgdl := b.avgdl()
	out := map[string]float32{}
	matchedTerms := map[string]int{}
	for _, term := range terms {
		chunks, ok := b.postings[term]
		if !ok {
			continue
		}
		df := b.df[term]
		idf := bm25IDF(n, df)
		for key, f := range chunks {
			dl := float64(b.length[key])
			denom := float64(f) + bm25K1*(1-bm25B+bm25B*dl/avgdl)
			out[key] += float32(idf * (float64(f) * (bm25K1 + 1)) / denom)
			matchedTerms[key]++
		}
	}
	if mode == LexicalAll {
		for key, count := range matchedTerms {
			if count < len(terms) {
				delete(out, key)
			}
		}
	}
	return out
}

// bm25IDF is the BM25+ inverse document frequency: adding 1 inside the log
// keeps the value non-negative even for a term present in more than half the
// corpus, where the classic Robertson-Sparck-Jones IDF goes negative and
// would otherwise penalize a chunk for containing a common word.
func bm25IDF(n, df int) float64 {
	return math.Log(1 + (float64(n)-float64(df)+0.5)/(float64(df)+0.5))
}
