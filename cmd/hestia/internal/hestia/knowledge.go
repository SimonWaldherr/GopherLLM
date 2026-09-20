package hestia

import (
	"context"
	"fmt"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	tinysql "github.com/SimonWaldherr/tinySQL"
	"tinyRAG/core"
)

// GopherLLMEmbedder adapts a loaded GopherLLM model to tinyRAG/core's
// minimal Embedder interface (CONCEPT.md section 10: "Im Ein-Prozess-Betrieb
// verwendet der Kern einen Go-Embedder-Adapter auf den Modellhost.").
type GopherLLMEmbedder struct {
	Runner *gopherllm.Runner
}

func (e *GopherLLMEmbedder) EmbedSingle(text string) ([]float64, error) {
	res, err := e.Runner.Embed(text)
	if err != nil {
		return nil, fmt.Errorf("embedding text: %w", err)
	}
	return float32To64(res.Embedding), nil
}

func (e *GopherLLMEmbedder) Embed(texts []string) ([][]float64, error) {
	results, err := e.Runner.EmbedBatch(context.Background(), texts)
	if err != nil {
		return nil, fmt.Errorf("embedding batch: %w", err)
	}
	out := make([][]float64, len(results))
	for i, r := range results {
		out[i] = float32To64(r.Embedding)
	}
	return out, nil
}

func float32To64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// KnowledgeBase wraps a tinyRAG/core.RAG instance for HestiaRAG's local
// document library (CONCEPT.md section 10 / section 15 step 4). It is
// exactly the "shared, all-principals" library the concept requires until
// per-person collection boundaries are implemented (see tinyRAG's own
// Principal doc comment) -- every search here uses one process-wide role,
// not a per-person one.
type KnowledgeBase struct {
	rag        *core.RAG
	embedModel string
}

// OpenKnowledgeBase loads embedModelPath once and opens (or creates) the
// knowledge base database at dbPath.
func OpenKnowledgeBase(embedModelPath, dbPath string) (*KnowledgeBase, error) {
	mmap, err := gopherllm.OpenMmap(embedModelPath)
	if err != nil {
		return nil, fmt.Errorf("opening embedding model: %w", err)
	}
	success := false
	defer func() {
		if !success {
			mmap.Close()
		}
	}()
	data := mmap.Bytes()
	gguf, err := gopherllm.ParseGGUF(data)
	if err != nil {
		return nil, fmt.Errorf("parsing embedding model GGUF: %w", err)
	}
	arch, _ := gguf.GetString("general.architecture")
	runner, err := gopherllm.RunnerFromGGUFBytes(data)
	if err != nil {
		return nil, fmt.Errorf("loading embedding model: %w", err)
	}
	embedModel := archOrPathName(arch, embedModelPath)

	kb, err := newKnowledgeBaseFromEmbedder(&GopherLLMEmbedder{Runner: runner}, embedModel, dbPath, tinysql.ModeWAL)
	if err != nil {
		runner.Close()
		return nil, err
	}
	success = true
	return kb, nil
}

// newKnowledgeBaseFromEmbedder builds a KnowledgeBase around an
// already-constructed Embedder, split out from OpenKnowledgeBase so tests
// can exercise the search/ingest wiring with a small stub embedder and
// ephemeral in-memory storage instead of loading a real multi-hundred-MB
// GGUF and touching disk.
func newKnowledgeBaseFromEmbedder(embedder core.Embedder, embedModel, dbPath string, storageMode tinysql.StorageMode) (*KnowledgeBase, error) {
	rag, err := core.OpenRAG(embedder, embedModel, 5, dbPath, storageMode, 256)
	if err != nil {
		return nil, fmt.Errorf("opening knowledge base: %w", err)
	}
	return &KnowledgeBase{rag: rag, embedModel: embedModel}, nil
}

func archOrPathName(arch, path string) string {
	if arch != "" {
		return arch
	}
	return path
}

func (kb *KnowledgeBase) Close() error {
	if kb == nil || kb.rag == nil {
		return nil
	}
	return kb.rag.Close()
}

// KnowledgeHit is one search result, ready to render as a source citation.
type KnowledgeHit struct {
	Content    string
	DocumentID string
	Title      string
}

// sharedPrincipal is the one principal HestiaRAG currently supports: every
// document is ingested with an empty roles list (tinyRAG's "|all|"
// wildcard, visible to every role), and every search uses this same
// principal. This is deliberate, not a shortcut: CONCEPT.md section 10.1
// item 1 requires per-person private collections to wait for verified
// Principal/collection boundaries, which don't exist yet in HestiaRAG
// (there is no identity system at all in this milestone). Once identity
// lands, Search/AddDocument should take a real Principal per request
// instead of using this constant.
var sharedPrincipal = core.Principal{Role: "it"}

// Search runs a query against the knowledge base's one shared library --
// see sharedPrincipal's doc comment for why there is no per-caller role yet.
func (kb *KnowledgeBase) Search(ctx context.Context, query string, k int) ([]KnowledgeHit, error) {
	if kb == nil || kb.rag == nil {
		return nil, fmt.Errorf("hestia: knowledge base not configured")
	}
	hits, err := kb.rag.Search(ctx, sharedPrincipal, query, k)
	if err != nil {
		return nil, err
	}
	out := make([]KnowledgeHit, 0, len(hits))
	for _, h := range hits {
		if h.Score < 0 {
			continue // narrative neighbor, not an independently citable primary hit
		}
		out = append(out, KnowledgeHit{Content: h.Content, DocumentID: h.DocumentID, Title: h.Citation.Title})
	}
	return out, nil
}

// AddDocument indexes text under title, tagged with sharedPrincipal's role
// so every search against this knowledge base (which always searches as
// that same principal) can find it. An empty roles argument to tinyRAG's
// own AddDocument falls back to whatever role happens to be globally
// active -- the same ambient-global pattern already fixed elsewhere in
// this codebase -- so this passes the role explicitly instead of relying
// on that default.
func (kb *KnowledgeBase) AddDocument(title, text string) (core.IngestResult, error) {
	if kb == nil || kb.rag == nil {
		return core.IngestResult{}, fmt.Errorf("hestia: knowledge base not configured")
	}
	chunks := chunkText(text, 1200)
	return kb.rag.AddDocument(title, chunks, kb.embedModel, []string{sharedPrincipal.Role}, core.R3IngestMetadata{})
}

// chunkText splits text into roughly maxRunes-sized pieces on paragraph
// boundaries where possible, falling back to a hard cut. This is a
// deliberately simple placeholder -- see CONCEPT.md section 10 for the
// richer import/extraction pipeline (PDF/DOCX, page offsets) tinyRAG's own
// pipeline already has and this does not yet reuse.
func chunkText(text string, maxRunes int) []string {
	paras := strings.Split(strings.TrimSpace(text), "\n\n")
	var chunks []string
	var cur strings.Builder
	for _, p := range paras {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if cur.Len()+len(p)+2 > maxRunes && cur.Len() > 0 {
			chunks = append(chunks, cur.String())
			cur.Reset()
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(p)
	}
	if cur.Len() > 0 {
		chunks = append(chunks, cur.String())
	}
	if len(chunks) == 0 {
		return []string{strings.TrimSpace(text)}
	}
	return chunks
}
