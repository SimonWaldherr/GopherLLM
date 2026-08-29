// Package rag is retrieval over the caller's own documents: a chunker, a
// hybrid BM25 + optional-vector index, and nothing else.
//
// It is a subpackage, not part of the root package, for the same reason
// server is: the root package's contract is small and stable, and retrieval
// is exactly the kind of capability that grows. Nothing here is imported by
// the root package.
//
// The Index is deliberately concrete rather than built on a pluggable Store
// interface. Everything is in-memory, and that ceiling is documented, not
// hidden: scoring a hundred thousand 768-dim normalized vectors is a few
// milliseconds of dot product, which is less time than this project spends
// decoding a single token. A caller whose corpus has genuinely outgrown that
// — who needs a real vector database, row-level ACLs pushed into a query
// planner, or a document store shared across processes — is doing something
// this package's "batteries" tier isn't for, and has a cheap way out:
// register a hand-written gopherllm.AgenticTool (or agent.RawTool) against
// their own store instead of this Index. A pluggable Store/Retriever
// interface pair was considered and rejected: it roughly doubles this
// package's exported surface to serve a persona this design does not
// otherwise target, and every interface method added here is a permanent
// compatibility promise.
//
// Non-goals, stated rather than silently unhandled:
//
//   - CJK and other non-whitespace-delimited scripts. The tokenizer below is
//     unicode.IsLetter/IsDigit-based and treats an unbroken run of such
//     characters as one term, which makes term-frequency/IDF scoring close
//     to meaningless on such content. Fixing this needs a different
//     segmentation strategy (e.g. n-gram tokenization for CJK runs), which is
//     out of scope for this package's first version.
//   - Stemming ("return policy" vs. "return policies" is a surface-form miss
//     in the lexical scorer).
//   - Phrase and proximity scoring (postings carry no position information).
//   - PDF/DOCX ingestion — IngestOptions.Extract is the hook; bring your own.
package rag

import (
	"context"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// Doc is one source document before chunking.
type Doc struct {
	// ID is the document's stable identity. Re-adding the same ID replaces
	// its chunks (and their BM25 postings) rather than duplicating them.
	ID    string
	Title string // shown in a citation, and folded into BM25 regardless of Chunker.TitlePrefix
	Path  string // where it came from; empty for an in-memory document
	Text  string
	// Time is the document's date, used for the recency signal. The zero
	// value disables recency scoring for this document only.
	Time time.Time
	Kind string // caller-defined class ("note", "email"); see Ranking.KindWeight
	Meta map[string]string
}

// Chunk is one retrievable window of a Doc, with byte offsets into the
// original Doc.Text so a citation can point at the source, not a copy.
type Chunk struct {
	DocID string
	Index int
	Text  string
	Start int
	End   int
}

// Hit is one retrieved chunk with every score component kept separately, so a
// ranking can be explained: "this came back because the words matched, not
// because it is semantically close" is a different bug report from the
// reverse.
type Hit struct {
	Chunk
	Title string
	Path  string
	Kind  string
	Time  time.Time
	// Vector is the cosine similarity in [-1,1], 0 when the Index has no
	// Embedder configured.
	Vector float32
	// Keyword is the BM25 score, batch-normalized into [0,1] for this query.
	Keyword float32
	// Recency is the exponential-decay score against Ranking.RecencyHalfLife,
	// in [0,1]; 0 when the chunk's Doc.Time is zero.
	Recency float32
	// Score is the blend actually used for ordering, in [0,1].
	Score float32
	// LexicalOnly reports that this hit survived Query.MinVector only because
	// it is a genuine BM25 term match, not a vector neighbor.
	LexicalOnly bool
}

// Embedder produces one vector per text, batched. *gopherllm.Model satisfies
// it via Model.EmbedBatch.
type Embedder interface {
	EmbedBatch(ctx context.Context, texts []string) ([]gopherllm.EmbeddingResult, error)
}
