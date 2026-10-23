//go:build darwin && cgo && coreml

package coreml_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/coreml"
)

// TestCoreMLConfiguredRetrievalParity compares retrieval rankings from compiled
// exports with their source GGUF. Ranking metrics are reported, not asserted:
// small rounding changes can reorder documents with nearly equal scores.
// Recall@3 uses the source's three neighbours, not labelled relevance targets.
// Embedding parity and token counts remain hard correctness checks.
func TestCoreMLConfiguredRetrievalParity(t *testing.T) {
	paths := os.Getenv("GOPHERLLM_COREML_RETRIEVAL_MODELS")
	if paths == "" {
		t.Skip("set GOPHERLLM_COREML_RETRIEVAL_MODELS and GOPHERLLM_COREML_GGUF")
	}
	gguf := os.Getenv("GOPHERLLM_COREML_GGUF")
	if gguf == "" {
		t.Fatal("the source GGUF is required for tokenizer and embedding parity")
	}
	tok, err := coreml.LoadTokenizer(gguf)
	if err != nil {
		t.Fatal(err)
	}
	documents := []struct{ id, text string }{
		{"password", "Ein vergessenes Passwort lässt sich über einen Link per E-Mail zurücksetzen."},
		{"backup", "Eine tägliche Sicherung schützt Dateien. Das Backup liegt auf einem getrennten Laufwerk."},
		{"invoice", "Die Rechnung nennt Betrag, Zahlungsfrist und Bankverbindung."},
		{"processor", "Der Prozessor führt die Befehle eines geladenen Computerprogramms aus."},
		{"conversation", "Ein Kontextspeicher bewahrt ältere Gesprächsrunden für spätere Rückfragen auf."},
		{"recipe", "Für Pfannkuchen verrührt man Mehl, Milch und Eier und erhitzt den Teig in einer Pfanne."},
		{"travel", "Le train relie Paris et Lyon. Le billet indique la date et la place."},
		{"calendar", "Una cita en el calendario tiene fecha, hora y una lista de invitados."},
		{"certificate", "Ein abgelaufenes TLS-Zertifikat kann den Aufbau einer sicheren HTTPS-Verbindung verhindern."},
		{"memory", "Arbeitsspeicher hält aktuelle Daten bereit. Bei Speichermangel lagert das System Seiten aus."},
		{"dns", "A DNS server translates a host name into an IP address."},
		{"battery", "Der Akku eines Laptops wird über ein USB-C-Netzteil geladen."},
	}
	queries := []string{
		"Wie kann ich mein vergessenes Passwort ändern?",
		"Where should I keep a backup of my files?",
		"Welche Zahlungsfrist steht auf der Rechnung?",
		"Wie kann ich ältere Runden eines Gesprächs wiederfinden?",
		"What ingredients are needed for pancakes?",
		"Comment voyager de Paris à Lyon en train?",
		"¿Cómo puedo invitar personas a una cita?",
		"Warum verhindert ein abgelaufenes Zertifikat eine HTTPS-Verbindung?",
	}
	texts := make([]string, 0, len(documents)+len(queries))
	for _, document := range documents {
		texts = append(texts, document.text)
	}
	texts = append(texts, queries...)
	for i, text := range texts {
		if count := len(tok.Encode(text)); count < 1 || count > 64 {
			t.Fatalf("retrieval fixture item %d has %d tokens; expected 1..64", i, count)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	r, _, err := gopherllm.RunnerFromPathWithOptions(gguf, gopherllm.LoadOptions{
		Prefault: gopherllm.MmapPrefaultNone, LogWriter: io.Discard,
	})
	if err != nil {
		t.Fatal(err)
	}
	reference, embedErr := r.EmbedBatch(ctx, texts)
	closeErr := r.Close() // Release source weights before opening native exports.
	if embedErr != nil || closeErr != nil {
		t.Fatalf("GGUF reference: embed=%v close=%v", embedErr, closeErr)
	}
	if len(reference) != len(texts) {
		t.Fatalf("source returned %d vectors for %d texts", len(reference), len(texts))
	}
	sourceScores := coremlRetrievalScores(t, reference, len(documents))
	for _, path := range filepath.SplitList(paths) {
		if path == "" {
			t.Fatal("empty compiled model path")
		}
		if !t.Run(filepath.Base(path), func(t *testing.T) {
			e, err := coreml.OpenEmbedder(path, tok, coreml.Options{
				ComputeUnits: coreml.CPUAndNeuralEngine, PadToken: tok.PadToken,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := e.Close(); err != nil {
					t.Error(err)
				}
			}()
			rows, err := e.EmbedBatch(ctx, texts)
			if err != nil {
				t.Fatal(err)
			}
			if len(rows) != len(reference) {
				t.Fatalf("candidate returned %d vectors, want %d", len(rows), len(reference))
			}
			minimumCosine := float32(1)
			for i, row := range rows {
				cosine, err := gopherllm.CosineSimilarity(row.Embedding, reference[i].Embedding)
				if err != nil || !coremlRetrievalFinite(cosine) || cosine < .995 || row.TokenCount != reference[i].TokenCount {
					t.Fatalf("item %d source parity: cosine=%g tokens=%d/%d err=%v", i, cosine,
						row.TokenCount, reference[i].TokenCount, err)
				}
				minimumCosine = min(minimumCosine, cosine)
			}
			candidateScores := coremlRetrievalScores(t, rows, len(documents))
			top1Matches, recalled := 0, 0
			maximumScoreError := float64(0)
			queryReports := make([]map[string]any, 0, len(queries))
			for query, scores := range sourceScores {
				sourceTop := coremlRetrievalTopK(scores, 3)
				candidateTop := coremlRetrievalTopK(candidateScores[query], 3)
				if sourceTop[0] == candidateTop[0] {
					top1Matches++
				}
				for _, document := range sourceTop {
					if slices.Contains(candidateTop, document) {
						recalled++
					}
				}
				for document, score := range scores {
					maximumScoreError = max(maximumScoreError, math.Abs(float64(score)-float64(candidateScores[query][document])))
				}
				sourceIDs, candidateIDs := make([]string, len(sourceTop)), make([]string, len(candidateTop))
				for i := range sourceTop {
					sourceIDs[i], candidateIDs[i] = documents[sourceTop[i]].id, documents[candidateTop[i]].id
				}
				queryReports = append(queryReports, map[string]any{
					"query": queries[query], "source_top3": sourceIDs, "candidate_top3": candidateIDs,
				})
			}
			report, err := json.Marshal(map[string]any{
				"schema_version": 1, "model": path, "mode": "cpu_neural_engine", "info": e.Info(),
				"documents": len(documents), "queries": len(queries),
				"top1_agreement":               float64(top1Matches) / float64(len(queries)),
				"recall_at_3":                  float64(recalled) / float64(3*len(queries)),
				"maximum_pairwise_score_error": maximumScoreError, "score_pair_scope": "query_document",
				"minimum_embedding_cosine": minimumCosine, "rankings": queryReports,
			})
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println("COREML_RETRIEVAL_JSON " + string(report))
		}) {
			return
		}
	}
}

func coremlRetrievalScores(t *testing.T, rows []gopherllm.EmbeddingResult, documents int) [][]float32 {
	t.Helper()
	scores := make([][]float32, len(rows)-documents)
	for query := range scores {
		scores[query] = make([]float32, documents)
		for document := range documents {
			score, err := gopherllm.CosineSimilarity(rows[documents+query].Embedding, rows[document].Embedding)
			if err != nil || !coremlRetrievalFinite(score) {
				t.Fatalf("query %d/document %d cosine=%g err=%v", query, document, score, err)
			}
			scores[query][document] = score
		}
	}
	return scores
}

func coremlRetrievalTopK(scores []float32, k int) []int {
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		if scores[a] > scores[b] {
			return -1
		}
		if scores[a] < scores[b] {
			return 1
		}
		return a - b // Stable document identity resolves exact score ties.
	})
	return order[:min(k, len(order))]
}

func coremlRetrievalFinite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
