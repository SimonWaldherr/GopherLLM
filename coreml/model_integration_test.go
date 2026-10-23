//go:build darwin && cgo && coreml

package coreml_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/coreml"
	"github.com/SimonWaldherr/GopherLLM/rag"
)

var _ rag.Embedder = (*coreml.Embedder)(nil)

type probeTokenizer struct{}

func (probeTokenizer) Encode(string) []uint32 { return []uint32{1, 2, 3} }

// The optional probe is an int32 [1,8] model whose pooled embedding is
// float32(input_ids * attention_mask). It exercises the native boundary.
func TestCoreMLProbe(t *testing.T) {
	path := os.Getenv("GOPHERLLM_COREML_PROBE")
	if path == "" {
		t.Skip("set GOPHERLLM_COREML_PROBE to a compiled boundary-test model")
	}
	for _, units := range []coreml.ComputeUnits{coreml.CPUOnly, coreml.CPUAndNeuralEngine} {
		e, err := coreml.OpenEmbedder(path, probeTokenizer{}, coreml.Options{ComputeUnits: units, PadToken: 7})
		if err != nil {
			t.Fatal(err)
		}
		result, err := e.EmbedBatch(context.Background(), []string{"probe", "repeat"})
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, row := range result {
			if row.TokenCount != 3 || len(row.Embedding) != 8 {
				t.Fatalf("unexpected probe output: %+v", row)
			}
			for i, value := range row.Embedding {
				want := 0.0
				if i < 3 {
					want = float64(i+1) / math.Sqrt(14)
				}
				if math.Abs(float64(value)-want) > 1e-6 {
					t.Fatalf("probe[%d]=%g, want %g", i, value, want)
				}
			}
		}
	}
}

// This opt-in check validates an exported encoder against its original GGUF
// and compares the same compiled graph with CPU-only and CPU/ANE permission.
// Device permission is not evidence that the ANE executes any operation.
func TestCoreMLConfiguredEmbeddings(t *testing.T) {
	path, gguf := os.Getenv("GOPHERLLM_COREML_MODEL"), os.Getenv("GOPHERLLM_COREML_GGUF")
	if path == "" && gguf == "" {
		t.Skip("set GOPHERLLM_COREML_MODEL and GOPHERLLM_COREML_GGUF")
	}
	if path == "" || gguf == "" {
		t.Fatal("both the compiled encoder and its source GGUF are required")
	}
	tok, err := coreml.LoadTokenizer(gguf)
	if err != nil {
		t.Fatal(err)
	}
	texts := []string{
		"Wie wird ein Computerprogramm ausgeführt?",
		"Ein Betriebssystem lädt das Programm in den Arbeitsspeicher. Der Prozessor führt seine Befehle aus.",
		"An embedding encoder can retrieve older conversation turns without replacing the chat model's key-value cache.",
		"Arbeitsspeicher",
		"The processor executes instructions.",
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	r, _, err := gopherllm.RunnerFromPathWithOptions(gguf, gopherllm.LoadOptions{Prefault: gopherllm.MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	reference, embedErr := r.EmbedBatch(ctx, texts)
	closeErr := r.Close() // Release GGUF weights before opening a Core ML model.
	if embedErr != nil || closeErr != nil {
		t.Fatalf("GGUF reference: embed=%v close=%v", embedErr, closeErr)
	}
	for _, mode := range []struct {
		name  string
		units coreml.ComputeUnits
	}{{"cpu", coreml.CPUOnly}, {"cpu_neural_engine", coreml.CPUAndNeuralEngine}} {
		t.Run(mode.name, func(t *testing.T) {
			e, err := coreml.OpenEmbedder(path, tok, coreml.Options{ComputeUnits: mode.units, PadToken: tok.PadToken})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			plan, err := coreml.Inspect(path, mode.units)
			if err != nil && !errors.Is(err, coreml.ErrUnavailable) {
				t.Fatal(err)
			}
			if plan.GPUOperations != 0 || mode.units == coreml.CPUOnly && plan.NeuralEngineOperations != 0 {
				t.Fatalf("compute plan exceeds permitted devices: %+v", plan)
			}
			if _, err := e.EmbedBatch(ctx, texts); err != nil { // Warm specialization.
				t.Fatal(err)
			}
			samples := make([]time.Duration, 5)
			minimumCosine := float32(1)
			for repeat := range samples {
				began := time.Now()
				rows, err := e.EmbedBatch(ctx, texts)
				samples[repeat] = time.Since(began)
				if err != nil {
					t.Fatal(err)
				}
				for i, row := range rows {
					if row.TokenCount != reference[i].TokenCount {
						t.Fatal("token count changed from the source tokenizer")
					}
					cosine, err := gopherllm.CosineSimilarity(row.Embedding, reference[i].Embedding)
					if err != nil || cosine < .995 {
						t.Fatalf("source GGUF parity for text %d: cosine=%g err=%v", i, cosine, err)
					}
					minimumCosine = min(minimumCosine, cosine)
				}
			}
			slices.Sort(samples)
			report, err := json.Marshal(map[string]any{
				"mode": mode.name, "texts_per_batch": len(texts), "median_seconds": samples[len(samples)/2].Seconds(),
				"minimum_gguf_cosine": minimumCosine, "info": e.Info(), "compute_plan": plan,
			})
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println("COREML_EMBEDDING_JSON " + string(report))
			if mode.units == coreml.CPUAndNeuralEngine {
				index := rag.New(rag.Options{Embedder: e, EmbedderID: "coreml-integration",
					Chunker: rag.Chunker{Tokenizer: tok.Tokenizer,
						MaxRunes: e.Info().SequenceLength - len(tok.Encode("")), OverlapRunes: 8}})
				text := strings.Repeat(texts[1]+" ", 20)
				if err := index.Add(ctx, rag.Doc{ID: "long-note", Text: text}); err != nil {
					t.Fatal("fixed-shape encoder rejected token-budgeted document:", err)
				}
				hits, err := index.Search(ctx, texts[0], rag.Query{TopK: 2})
				if err != nil || len(hits) == 0 {
					t.Fatalf("Core ML RAG retrieval: hits=%d err=%v", len(hits), err)
				}
			}
		})
	}
}

// Compare static batch exports with identical texts, retaining only one native
// model at a time. Include a single query, a full group and a partial final
// group; bulk throughput alone must not hide a single-query latency regression.
func TestCoreMLBatchComparison(t *testing.T) {
	paths, gguf := os.Getenv("GOPHERLLM_COREML_BATCH_MODELS"), os.Getenv("GOPHERLLM_COREML_GGUF")
	if paths == "" {
		t.Skip("set GOPHERLLM_COREML_BATCH_MODELS to a path-list of compiled batch exports and GOPHERLLM_COREML_GGUF")
	}
	if gguf == "" {
		t.Fatal("the source GGUF is required for tokenizer and output parity")
	}
	tok, err := coreml.LoadTokenizer(gguf)
	if err != nil {
		t.Fatal(err)
	}
	base := []string{"Wie wird ein Computerprogramm ausgeführt?", "Arbeitsspeicher",
		"The processor executes instructions.", "An embedding encoder retrieves related documents."}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	r, _, err := gopherllm.RunnerFromPathWithOptions(gguf, gopherllm.LoadOptions{Prefault: gopherllm.MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	reference, embedErr := r.EmbedBatch(ctx, base)
	closeErr := r.Close()
	if embedErr != nil || closeErr != nil {
		t.Fatalf("GGUF reference: embed=%v close=%v", embedErr, closeErr)
	}
	for _, path := range filepath.SplitList(paths) {
		if path == "" {
			t.Fatal("empty model path")
		}
		if !t.Run(filepath.Base(path), func(t *testing.T) {
			e, err := coreml.OpenEmbedder(path, tok, coreml.Options{PadToken: tok.PadToken})
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			plan, err := coreml.Inspect(path, coreml.CPUAndNeuralEngine)
			if err != nil && !errors.Is(err, coreml.ErrUnavailable) {
				t.Fatal(err)
			}
			if plan.GPUOperations != 0 {
				t.Fatalf("CPU/ANE model planned GPU work: %+v", plan)
			}
			for _, count := range []int{1, 4, 17, 32} {
				texts := make([]string, count)
				for i := range texts {
					texts[i] = base[i%len(base)]
				}
				if _, err := e.EmbedBatch(ctx, texts); err != nil { // Exclude specialization.
					t.Fatal(err)
				}
				samples := make([]time.Duration, 9)
				minimumCosine := float32(1)
				for repeat := range samples {
					began := time.Now()
					rows, err := e.EmbedBatch(ctx, texts)
					samples[repeat] = time.Since(began)
					if err != nil {
						t.Fatal(err)
					}
					if len(rows) != count {
						t.Fatalf("output count: got %d, want %d", len(rows), count)
					}
					for i, row := range rows {
						want := reference[i%len(reference)]
						cosine, err := gopherllm.CosineSimilarity(row.Embedding, want.Embedding)
						if err != nil || cosine < .995 || row.TokenCount != want.TokenCount {
							t.Fatalf("text %d source parity: cosine=%g tokens=%d/%d err=%v", i, cosine, row.TokenCount, want.TokenCount, err)
						}
						minimumCosine = min(minimumCosine, cosine)
					}
				}
				slices.Sort(samples)
				report, err := json.Marshal(map[string]any{
					"model": path, "texts": count, "info": e.Info(), "compute_plan": plan,
					"median_seconds": samples[len(samples)/2].Seconds(), "min_seconds": samples[0].Seconds(),
					"max_seconds": samples[len(samples)-1].Seconds(), "minimum_gguf_cosine": minimumCosine,
				})
				if err != nil {
					t.Fatal(err)
				}
				fmt.Println("COREML_BATCH_JSON " + string(report))
			}
		}) {
			return
		}
	}
}
