package rag

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

// LexicalMode selects how a Query's BM25 half treats multiple query terms.
type LexicalMode int

const (
	// LexicalAny scores a chunk if it contains at least one query term (an OR
	// match) — the default, and the right choice for an ordinary question.
	LexicalAny LexicalMode = iota
	// LexicalAll requires every query term to appear in a chunk (an AND
	// match) — the right choice for a query that IS an identifier or a short
	// exact phrase, where a partial match is usually noise.
	LexicalAll
)

// GuessLexicalMode is a heuristic a caller MAY use when building a Query, not
// something Search applies automatically: the zero value of LexicalMode is
// LexicalAny, indistinguishable from "the caller wants OR-recall", so
// applying this automatically would reintroduce exactly the zero-value
// ambiguity Ranking's weight rule (see effectiveWeights) exists to avoid.
//
// It returns LexicalAll for a short query (at most 3 whitespace-separated
// tokens) containing a token that mixes letters and digits, or that joins two
// alphanumeric runs with a hyphen, underscore, dot or slash — the shape of an
// exact identifier lookup — and LexicalAny otherwise.
func GuessLexicalMode(query string) LexicalMode {
	fields := strings.Fields(query)
	if len(fields) == 0 || len(fields) > 3 {
		return LexicalAny
	}
	for _, f := range fields {
		if looksLikeIdentifier(f) {
			return LexicalAll
		}
	}
	return LexicalAny
}

func looksLikeIdentifier(tok string) bool {
	hasLetter, hasDigit, hasJoiner := false, false, false
	for _, c := range tok {
		switch {
		case unicode.IsLetter(c):
			hasLetter = true
		case unicode.IsDigit(c):
			hasDigit = true
		case c == '-' || c == '_' || c == '.' || c == '/':
			hasJoiner = true
		}
	}
	return (hasLetter && hasDigit) || (hasJoiner && (hasLetter || hasDigit))
}

// Ranking configures the blend Search uses to order results.
type Ranking struct {
	// VectorWeight, KeywordWeight and RecencyWeight blend the three signals.
	//
	// If ALL THREE are zero (the zero-value Ranking{} case), the documented
	// defaults apply: 0.7 / 0.2 / 0.1, renormalized over whichever signals
	// are actually present in this Index (an index with no Embedder scores
	// 0.67 keyword / 0.33 recency, not capped at 0.3 keyword).
	//
	// If ANY ONE of the three is non-zero, all three are taken EXACTLY as
	// given — including a zero among them, which then means exactly zero
	// weight for that signal. This is the whole rule: it is what lets
	// Ranking{KeywordWeight: 1} mean pure lexical search, rather than the
	// zero VectorWeight being silently reinterpreted as "use the default
	// 0.7" the way it would be under a naive per-field default.
	VectorWeight, KeywordWeight, RecencyWeight float32
	// RecencyHalfLife is the exponential decay half-life for the recency
	// signal. Zero means 180 days. A chunk whose Doc.Time is zero scores 0
	// for recency, regardless of this value.
	RecencyHalfLife time.Duration
	// KindWeight multiplies a chunk's final score by Doc.Kind. A Kind with no
	// entry here scores ×1.0, never ×0 — an unconfigured Kind is not
	// silently excluded.
	KindWeight map[string]float32
	// Stopwords are excluded from BM25 entirely: never indexed, never
	// counted in a query. Empty by default; this package ships no built-in
	// list for any language, so it embeds none.
	Stopwords map[string]bool
}

func (r Ranking) recencyHalfLife() time.Duration {
	if r.RecencyHalfLife > 0 {
		return r.RecencyHalfLife
	}
	return 180 * 24 * time.Hour
}

// effectiveWeights resolves Ranking's zero-vs-unset rule: default 0.7/0.2/0.1
// when all three fields are zero, or the literal configured values otherwise.
func effectiveWeights(r Ranking) (vector, keyword, recency float32) {
	if r.VectorWeight == 0 && r.KeywordWeight == 0 && r.RecencyWeight == 0 {
		return 0.7, 0.2, 0.1
	}
	return r.VectorWeight, r.KeywordWeight, r.RecencyWeight
}

// Options configures a new Index.
type Options struct {
	// Embedder produces the vector half of retrieval. Nil (the default)
	// means pure BM25: no model is required, and Search never calls
	// anything. This is deliberately not the always-on default — a chat
	// model's mean-pooled hidden states are not a trained embedding head,
	// every indexed chunk clears the Runner's KV prefix cache, and plain
	// BM25 already wins the queries people actually type at their own
	// documents: part numbers, error codes, invoice ids, surnames. Configure
	// a dedicated encoder (BGE/E5/Nomic/Granite) here when semantic recall is
	// worth those two costs.
	Embedder Embedder
	// EmbedderID identifies which embedder produced this Index's vectors.
	// Load refuses to open a snapshot whose recorded EmbedderID does not
	// match, since comparing vectors from two different embedders as if they
	// shared a space would silently return meaningless neighbors.
	EmbedderID string
	// QueryPrefix and DocPrefix are prepended before embedding a query or a
	// document chunk, for models trained with an instruction convention
	// (E5's "query: "/"passage: ", Nomic's "search_query: "/
	// "search_document: "). Empty by default — GopherLLM embeds no
	// third-party convention for you to opt into by accident.
	QueryPrefix, DocPrefix string
	Chunker                Chunker
	Ranking
}

// Query configures one Search call.
type Query struct {
	// TopK bounds the number of hits returned. Zero means 5.
	TopK int
	// MinScore is the minimum blended Score (see Ranking) a hit must reach.
	// Zero means 0.25. For a chunk with no keyword match at all, the check is
	// made against the score with its recency contribution stripped out, so
	// a merely-recent, otherwise-irrelevant chunk cannot pass the bar on
	// freshness alone.
	MinScore float32
	// MinVector is the minimum cosine similarity a chunk needs to be
	// considered AT ALL when an Embedder is configured — except a chunk that
	// is a genuine BM25 term match, which is exempt (see Hit.LexicalOnly).
	// Zero (the default) disables this filter.
	MinVector float32
	// MaxChunksPerDoc caps how many of TopK's slots one Doc may occupy — a
	// diversity guard against one long document crowding out every other
	// source. Zero means 2; a negative value disables the cap.
	MaxChunksPerDoc int
	// MaxDocs caps the number of distinct documents represented across all
	// hits. Zero means unlimited.
	MaxDocs int
	// Neighbours widens each hit to include this many chunks before and
	// after the matched one, from the same Doc, using the ORIGINAL
	// contiguous text between their offsets (not a concatenation of the
	// individual chunk strings, which may overlap per Chunker.OverlapRunes).
	Neighbours int
	// Filter, when set, excludes a Doc from consideration before any scoring
	// happens.
	Filter func(Doc) bool
	// Lexical selects BM25's OR/AND behavior. See GuessLexicalMode for a
	// heuristic a caller may apply when building a Query.
	Lexical LexicalMode
}

func (q Query) topK() int {
	if q.TopK > 0 {
		return q.TopK
	}
	return 5
}

func (q Query) minScore() float32 {
	if q.MinScore != 0 {
		return q.MinScore
	}
	return 0.25
}

func (q Query) maxChunksPerDoc() int {
	if q.MaxChunksPerDoc != 0 {
		return q.MaxChunksPerDoc
	}
	return 2
}

// docEntry is one indexed Doc's metadata plus the keys of its chunks, in
// Chunker.Split order.
type docEntry struct {
	doc       Doc
	chunkKeys []string
}

// chunkEntry is one indexed chunk: its Chunk plus denormalized Doc fields a
// Hit needs (so Search need not re-look-up the parent Doc for every result)
// and its vector, if this Index has an Embedder.
type chunkEntry struct {
	chunk  Chunk
	title  string
	path   string
	kind   string
	time   time.Time
	vector []float32
}

// Index is an in-memory hybrid index. Add, Load and Search are all safe for
// concurrent use — Search takes the read lock, Add/Load take the write lock.
// A real sync.RWMutex, not merely a documented convention: the brute-force
// scan this Index performs is a few milliseconds at most, so the lock's
// overhead is not a measurable cost.
type Index struct {
	mu sync.RWMutex

	opts Options
	bm25 *bm25Index
	docs map[string]*docEntry
	// chunks and order together give a stable, insertion-ordered view of
	// every indexed chunk; chunks is keyed for O(1) lookup by chunkKey, order
	// exists because Go map iteration order is randomized and Search's
	// full scan should not depend on incidental map bucket layout for its own
	// reasoning (the final result order is always the explicit sort below).
	chunks map[string]*chunkEntry
	order  []string
	dim    int
}

// New returns an empty Index configured by opts.
func New(opts Options) *Index {
	return &Index{
		opts:   opts,
		bm25:   newBM25Index(opts.Stopwords),
		docs:   map[string]*docEntry{},
		chunks: map[string]*chunkEntry{},
	}
}

func chunkKey(docID string, index int) string {
	return docID + "\x00" + strconv.Itoa(index)
}

// Add chunks and indexes docs. Re-adding an existing Doc.ID replaces its
// chunks (and their BM25 postings) entirely rather than duplicating them.
// Every document's chunks are embedded in one batch (when an Embedder is
// configured), so adding a corpus's worth of documents in one Add call is
// materially cheaper than one call per document.
func (ix *Index) Add(ctx context.Context, docs ...Doc) error {
	if len(docs) == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	type pending struct {
		doc    Doc
		chunks []Chunk
	}
	plan := make([]pending, 0, len(docs))
	var texts []string
	for _, doc := range docs {
		chunks := ix.opts.Chunker.Split(doc)
		plan = append(plan, pending{doc: doc, chunks: chunks})
		if ix.opts.Embedder != nil {
			for _, c := range chunks {
				embedText := c.Text
				if ix.opts.Chunker.TitlePrefix && doc.Title != "" {
					embedText = doc.Title + " > " + embedText
				}
				texts = append(texts, ix.opts.DocPrefix+embedText)
			}
		}
	}

	var vectors [][]float32
	if ix.opts.Embedder != nil && len(texts) > 0 {
		results, err := ix.opts.Embedder.EmbedBatch(ctx, texts)
		if err != nil {
			return fmt.Errorf("rag: embed documents: %w", err)
		}
		if len(results) != len(texts) {
			return fmt.Errorf("rag: embedder returned %d vectors for %d texts", len(results), len(texts))
		}
		vectors = make([][]float32, len(results))
		for i, r := range results {
			vectors[i] = r.Embedding
		}
	}

	ix.mu.Lock()
	defer ix.mu.Unlock()

	vi := 0
	for _, p := range plan {
		if existing, ok := ix.docs[p.doc.ID]; ok {
			for _, key := range existing.chunkKeys {
				ix.bm25.remove(key)
				delete(ix.chunks, key)
			}
			ix.order = removeAll(ix.order, existing.chunkKeys)
		}
		entry := &docEntry{doc: p.doc, chunkKeys: make([]string, 0, len(p.chunks))}
		for _, c := range p.chunks {
			key := chunkKey(p.doc.ID, c.Index)
			ix.bm25.add(key, p.doc.Title+"\n"+c.Text)
			ce := &chunkEntry{chunk: c, title: p.doc.Title, path: p.doc.Path, kind: p.doc.Kind, time: p.doc.Time}
			if ix.opts.Embedder != nil {
				ce.vector = vectors[vi]
				vi++
				if ix.dim == 0 {
					ix.dim = len(ce.vector)
				}
			}
			ix.chunks[key] = ce
			ix.order = append(ix.order, key)
			entry.chunkKeys = append(entry.chunkKeys, key)
		}
		ix.docs[p.doc.ID] = entry
	}
	return nil
}

func removeAll(order []string, remove []string) []string {
	if len(remove) == 0 {
		return order
	}
	drop := make(map[string]bool, len(remove))
	for _, k := range remove {
		drop[k] = true
	}
	out := order[:0]
	for _, k := range order {
		if !drop[k] {
			out = append(out, k)
		}
	}
	return out
}

// Len reports the total number of indexed chunks.
func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.chunks)
}

// Docs reports the total number of indexed documents.
func (ix *Index) Docs() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.docs)
}

type scored struct {
	key                string
	vecRaw, kwRaw, rec float32
	score              float32
	lexicalOnly        bool
}

// Search ranks the index against query, blending BM25 keyword matching,
// cosine vector similarity (when this Index has an Embedder), and recency,
// per q and the Index's Ranking. See the package doc for why this always
// scores the whole index rather than a top-N candidate shortlist: at this
// package's target scale, the union IS the brute-force scan.
func (ix *Index) Search(ctx context.Context, query string, q Query) ([]Hit, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	if len(ix.chunks) == 0 {
		return nil, nil
	}

	kwScores := ix.bm25.scores(query, q.Lexical)
	var maxKw float32
	for _, s := range kwScores {
		if s > maxKw {
			maxKw = s
		}
	}

	var qvec []float32
	if ix.opts.Embedder != nil {
		results, err := ix.opts.Embedder.EmbedBatch(ctx, []string{ix.opts.QueryPrefix + query})
		if err != nil {
			return nil, fmt.Errorf("rag: embed query: %w", err)
		}
		if len(results) == 1 {
			qvec = results[0].Embedding
		}
	}

	vw, kwWeight, recWeight := effectiveWeights(ix.opts.Ranking)
	if ix.opts.Embedder == nil {
		total := kwWeight + recWeight
		if total > 0 {
			kwWeight, recWeight = kwWeight/total, recWeight/total
		} else {
			kwWeight, recWeight = 1, 0
		}
		vw = 0
	}
	halfLife := ix.opts.Ranking.recencyHalfLife()

	candidates := make([]scored, 0, len(ix.chunks))
	for _, key := range ix.order {
		ce := ix.chunks[key]
		doc := ix.docs[ce.chunk.DocID]
		if q.Filter != nil && doc != nil && !q.Filter(doc.doc) {
			continue
		}
		kwRaw := kwScores[key]
		var vecRaw float32
		if ix.opts.Embedder != nil && len(qvec) > 0 && len(ce.vector) > 0 {
			vecRaw = dot(qvec, ce.vector)
		}
		if ix.opts.Embedder != nil && vecRaw < q.MinVector && kwRaw <= 0 {
			continue
		}
		lexicalOnly := ix.opts.Embedder != nil && vecRaw < q.MinVector && kwRaw > 0

		kwNorm := float32(0)
		if maxKw > 0 {
			kwNorm = kwRaw / maxKw
		}
		vecNorm := (vecRaw + 1) / 2
		recNorm := recencyScore(ce.time, halfLife)

		score := vw*vecNorm + kwWeight*kwNorm + recWeight*recNorm
		if kind := ce.kind; kind != "" {
			if weight, ok := ix.opts.Ranking.KindWeight[kind]; ok {
				score *= weight
			}
		}
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			continue
		}

		threshold := score
		if kwNorm == 0 {
			threshold -= recWeight * recNorm
		}
		if threshold < q.minScore() {
			continue
		}

		candidates = append(candidates, scored{key: key, vecRaw: vecRaw, kwRaw: kwNorm, rec: recNorm, score: score, lexicalOnly: lexicalOnly})
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		ci, cj := ix.chunks[candidates[i].key].chunk, ix.chunks[candidates[j].key].chunk
		if ci.DocID != cj.DocID {
			return ci.DocID < cj.DocID
		}
		return ci.Index < cj.Index
	})

	maxPerDoc := q.maxChunksPerDoc()
	perDoc := map[string]int{}
	docsUsed := map[string]bool{}
	hits := make([]Hit, 0, q.topK())
	for _, c := range candidates {
		if len(hits) >= q.topK() {
			break
		}
		ce := ix.chunks[c.key]
		docID := ce.chunk.DocID
		if maxPerDoc > 0 && perDoc[docID] >= maxPerDoc {
			continue
		}
		if q.MaxDocs > 0 && !docsUsed[docID] && len(docsUsed) >= q.MaxDocs {
			continue
		}
		chunk := ce.chunk
		if q.Neighbours > 0 {
			chunk = ix.widen(docID, chunk, q.Neighbours)
		}
		hits = append(hits, Hit{
			Chunk: chunk, Title: ce.title, Path: ce.path, Kind: ce.kind, Time: ce.time,
			Vector: c.vecRaw, Keyword: c.kwRaw, Recency: c.rec, Score: c.score, LexicalOnly: c.lexicalOnly,
		})
		perDoc[docID]++
		docsUsed[docID] = true
	}
	return hits, nil
}

// widen expands chunk to cover its Neighbours siblings within the same Doc,
// using the ORIGINAL Doc.Text between the widened range's offsets rather than
// concatenating the individual (possibly overlapping) chunk strings.
func (ix *Index) widen(docID string, chunk Chunk, neighbours int) Chunk {
	entry, ok := ix.docs[docID]
	if !ok {
		return chunk
	}
	lo := max(0, chunk.Index-neighbours)
	hi := min(len(entry.chunkKeys)-1, chunk.Index+neighbours)
	if lo == chunk.Index && hi == chunk.Index {
		return chunk
	}
	loChunk := ix.chunks[entry.chunkKeys[lo]].chunk
	hiChunk := ix.chunks[entry.chunkKeys[hi]].chunk
	return Chunk{
		DocID: chunk.DocID, Index: chunk.Index,
		Start: loChunk.Start, End: hiChunk.End,
		Text: entry.doc.Text[loChunk.Start:hiChunk.End],
	}
}

func dot(a, b []float32) float32 {
	n := min(len(a), len(b))
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// recencyScore is exponential decay against halfLife: 1.0 for a document
// dated now, 0.5 at exactly one half-life old, approaching 0 beyond that. A
// zero Time (Doc.Time never set) scores 0 — there is nothing to decay from.
func recencyScore(t time.Time, halfLife time.Duration) float32 {
	if t.IsZero() || halfLife <= 0 {
		return 0
	}
	age := time.Since(t)
	if age < 0 {
		age = 0
	}
	return float32(math.Pow(0.5, float64(age)/float64(halfLife)))
}
