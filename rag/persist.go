package rag

import (
	"encoding/gob"
	"fmt"
	"io"
	"time"
)

// snapshotVersion guards Load against a file written by an incompatible
// future format. There is no cross-version migration promise yet: bump this
// on any field change and Load will refuse the old file outright rather than
// silently misreading it.
const snapshotVersion = 1

// persistedDoc is a Doc without its derived chunks — those are rebuilt from
// persistedChunk's offsets into Text, so a chunk's text is never stored
// twice.
type persistedDoc struct {
	ID, Title, Path, Text, Kind string
	Time                        time.Time
	Meta                        map[string]string
}

type persistedChunk struct {
	DocID      string
	Index      int
	Start, End int
	Vector     []float32
}

type snapshot struct {
	Version    int
	EmbedderID string
	Docs       []persistedDoc
	Chunks     []persistedChunk
}

// Save writes a snapshot of the index. The format is encoding/gob over plain
// Go structs — simpler and safer than a hand-rolled binary layout, at the
// cost of some size versus a packed float32 slab; this package has made no
// promise yet that a saved index is small, only that reloading it is fast
// compared to re-embedding a whole corpus.
func (ix *Index) Save(w io.Writer) error {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	snap := snapshot{Version: snapshotVersion, EmbedderID: ix.opts.EmbedderID}
	for _, d := range ix.docs {
		snap.Docs = append(snap.Docs, persistedDoc{
			ID: d.doc.ID, Title: d.doc.Title, Path: d.doc.Path, Text: d.doc.Text,
			Kind: d.doc.Kind, Time: d.doc.Time, Meta: d.doc.Meta,
		})
	}
	for _, key := range ix.order {
		ce := ix.chunks[key]
		snap.Chunks = append(snap.Chunks, persistedChunk{
			DocID: ce.chunk.DocID, Index: ce.chunk.Index,
			Start: ce.chunk.Start, End: ce.chunk.End, Vector: ce.vector,
		})
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// Load rebuilds an Index from a snapshot written by Save. opts.EmbedderID
// must match the snapshot's recorded value (both empty counts as a match,
// meaning "no embedder either time"): comparing vectors from two different
// embedders as if they shared a space would silently return meaningless
// results, so a mismatch is refused rather than risked. opts.Embedder itself
// is used for future Add calls and Search's query embedding; it is not
// required to reload a pure-BM25 snapshot.
func Load(r io.Reader, opts Options) (*Index, error) {
	var snap snapshot
	if err := gob.NewDecoder(r).Decode(&snap); err != nil {
		return nil, fmt.Errorf("rag: decode snapshot: %w", err)
	}
	if snap.Version != snapshotVersion {
		return nil, fmt.Errorf("rag: snapshot version %d, want %d", snap.Version, snapshotVersion)
	}
	if snap.EmbedderID != opts.EmbedderID {
		return nil, fmt.Errorf("rag: snapshot embedder %q does not match %q", snap.EmbedderID, opts.EmbedderID)
	}

	ix := New(opts)
	for _, d := range snap.Docs {
		ix.docs[d.ID] = &docEntry{doc: Doc{ID: d.ID, Title: d.Title, Path: d.Path, Text: d.Text, Kind: d.Kind, Time: d.Time, Meta: d.Meta}}
	}
	for _, c := range snap.Chunks {
		entry, ok := ix.docs[c.DocID]
		if !ok {
			return nil, fmt.Errorf("rag: snapshot chunk references unknown doc %q", c.DocID)
		}
		key := chunkKey(c.DocID, c.Index)
		text := entry.doc.Text[c.Start:c.End]
		ix.bm25.add(key, entry.doc.Title+"\n"+text)
		ce := &chunkEntry{
			chunk: Chunk{DocID: c.DocID, Index: c.Index, Start: c.Start, End: c.End, Text: text},
			title: entry.doc.Title, path: entry.doc.Path, kind: entry.doc.Kind, time: entry.doc.Time,
			vector: c.Vector,
		}
		if len(ce.vector) > 0 && ix.dim == 0 {
			ix.dim = len(ce.vector)
		}
		ix.chunks[key] = ce
		ix.order = append(ix.order, key)
		entry.chunkKeys = append(entry.chunkKeys, key)
	}
	return ix, nil
}
