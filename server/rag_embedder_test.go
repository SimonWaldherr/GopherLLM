package server

import (
	"bytes"
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

type callerRAGEmbedder struct {
	calls, closed int
}

func (e *callerRAGEmbedder) EmbedBatch(ctx context.Context, texts []string) ([]gopherllm.EmbeddingResult, error) {
	e.calls++
	out := make([]gopherllm.EmbeddingResult, len(texts))
	for i := range out {
		out[i].Embedding = []float32{1, 0}
	}
	return out, ctx.Err()
}
func (e *callerRAGEmbedder) Close() error { e.closed++; return nil }

func TestCallerRAGEmbedderTakesPrecedenceAndRemainsCallerOwned(t *testing.T) {
	embedder := &callerRAGEmbedder{}
	var logs bytes.Buffer
	h := NewHandler(nil, HandlerOptions{
		Features: Features{RAG: true}, RAGEmbedder: embedder, RAGEmbedderID: "encoder-v1",
		RAGEmbedModelPath: filepath.Join(t.TempDir(), "must-not-load.gguf"), LogWriter: &logs,
	})
	if !h.rag.index.HasEmbedder() || h.rag.embedRunner != nil {
		t.Fatal("caller-owned encoder was not installed independently of a Runner")
	}
	if err := h.rag.index.Add(context.Background(), rag.Doc{ID: "note", Text: "The telescope needs calibration."}); err != nil {
		t.Fatal(err)
	}
	if _, err := h.rag.index.Search(context.Background(), "calibration", rag.Query{}); err != nil {
		t.Fatal(err)
	}
	if embedder.calls != 2 || strings.Contains(logs.String(), "must-not-load") {
		t.Fatalf("caller encoder not used exclusively: calls=%d logs=%s", embedder.calls, logs.String())
	}
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}
	if embedder.closed != 0 {
		t.Fatal("handler closed a caller-owned embedding model")
	}
}

func TestCallerRAGEmbedderRequiresIdentityAndFeature(t *testing.T) {
	for _, tc := range []struct {
		enabled bool
		id      string
	}{{false, "encoder-v1"}, {true, ""}} {
		embedder := &callerRAGEmbedder{}
		var logs bytes.Buffer
		h := NewHandler(nil, HandlerOptions{Features: Features{RAG: tc.enabled}, RAGEmbedder: embedder, RAGEmbedderID: tc.id, LogWriter: &logs})
		if h.rag.index.HasEmbedder() || embedder.calls != 0 {
			t.Fatal("unidentified or disabled encoder was installed")
		}
		if tc.enabled && !strings.Contains(logs.String(), "RAGEmbedderID") {
			t.Fatal("missing embedding identity was not diagnosed")
		}
		_ = h.Close()
		if embedder.closed != 0 {
			t.Fatal("handler closed the unused caller-owned encoder")
		}
	}
}

func TestCallerRAGEmbedderIdentityProtectsSnapshots(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rag.gob")
	embedder := &callerRAGEmbedder{}
	opts := HandlerOptions{Features: Features{RAG: true}, RAGEmbedder: embedder, RAGEmbedderID: "encoder-v1", RAGSnapshotPath: path}
	h := NewHandler(nil, opts)
	if err := h.rag.index.Add(context.Background(), rag.Doc{ID: "note", Text: "Keep these encoder-specific vectors."}); err != nil {
		t.Fatal(err)
	}
	h.rag.save()
	_ = h.Close()
	reopened := NewHandler(nil, opts)
	if reopened.rag.index.Docs() != 1 {
		t.Fatal("matching encoder identity did not restore its snapshot")
	}
	_ = reopened.Close()
	opts.RAGEmbedderID = "encoder-v2"
	var logs bytes.Buffer
	opts.LogWriter = &logs
	different := NewHandler(nil, opts)
	defer different.Close()
	if different.rag.index.Docs() != 0 || !strings.Contains(logs.String(), "embedder") {
		t.Fatalf("different embedding space reused stored vectors: docs=%d logs=%s", different.rag.index.Docs(), logs.String())
	}
}

type boundedRAGEmbedder struct {
	tokenizer *gopherllm.Tokenizer
	limit     int
	texts     []string
}

func (e *boundedRAGEmbedder) EmbedBatch(ctx context.Context, texts []string) ([]gopherllm.EmbeddingResult, error) {
	out := make([]gopherllm.EmbeddingResult, len(texts))
	for i, text := range texts {
		tokens := e.tokenizer.Encode(text)
		if len(tokens) > e.limit {
			return nil, fmt.Errorf("encoder input has %d tokens, limit %d", len(tokens), e.limit)
		}
		e.texts = append(e.texts, text)
		out[i] = gopherllm.EmbeddingResult{Embedding: []float32{1, 0}, TokenCount: len(tokens)}
	}
	return out, ctx.Err()
}

func TestCallerRAGEmbedderReceivesBoundedCompleteChunks(t *testing.T) {
	const sequenceLength = 64
	tokenizer := &gopherllm.Tokenizer{
		Mode: gopherllm.TokenizerWordPiece, RawWordPieceVocab: true,
		AddBOS: true, AddSEP: true, BOSID: 2, SEPID: 3, UNKID: 1,
		Vocab:     []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]"},
		TokenToID: map[string]uint32{},
	}
	words := make([]string, 300)
	for i := range words {
		words[i] = fmt.Sprintf("item%d", i)
		tokenizer.Vocab = append(tokenizer.Vocab, words[i])
	}
	for i, token := range tokenizer.Vocab {
		tokenizer.TokenToID[token] = uint32(i)
	}
	embedder := &boundedRAGEmbedder{tokenizer: tokenizer, limit: sequenceLength}
	h := NewHandler(nil, HandlerOptions{
		Features: Features{RAG: true}, RAGEmbedder: embedder, RAGEmbedderID: "bounded-encoder",
		RAGChunker: rag.Chunker{
			Tokenizer: tokenizer, MaxRunes: sequenceLength - len(tokenizer.Encode("")),
			OverlapRunes: 8, MinRunes: 8,
		},
	})
	defer h.Close()
	if err := h.rag.index.Add(context.Background(), rag.Doc{ID: "long", Text: strings.Join(words, " ")}); err != nil {
		t.Fatal(err)
	}
	if len(embedder.texts) < 2 {
		t.Fatal("long document was not split for the fixed-length encoder")
	}
	seen := make(map[string]bool)
	for _, chunk := range embedder.texts {
		for _, word := range strings.Fields(chunk) {
			seen[word] = true
		}
	}
	for _, word := range words {
		if !seen[word] {
			t.Fatalf("document content %q was silently lost before embedding", word)
		}
	}
}
