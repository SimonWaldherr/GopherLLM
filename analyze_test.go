package gopherllm

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestAnalyzeGGUFTinyModel(t *testing.T) {
	g, err := ParseGGUFQuiet(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	tok, err := TokenizerFromMetadata(g.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	a := AnalyzeGGUF(g, tok)
	if a.Architecture != "llama" || !a.Supported {
		t.Fatalf("arch = %q supported=%v", a.Architecture, a.Supported)
	}
	if a.Layers != 1 || a.Dim != 8 || a.Heads != 2 || a.VocabSize != 16 {
		t.Fatalf("geometry = %+v", a)
	}
	if a.Params <= 0 || a.FileBytes <= 0 || a.BitsPerWeight != 32 { // all-F32 model
		t.Fatalf("params=%d bytes=%d bits=%v", a.Params, a.FileBytes, a.BitsPerWeight)
	}
	if len(a.DTypes) != 1 || a.DTypes[0].Type != GGMLTypeF32 {
		t.Fatalf("dtypes = %+v", a.DTypes)
	}
	if len(a.LargestTensors) == 0 || a.LargestTensors[0].Name != "token_embd.weight" {
		t.Fatalf("largest = %+v", a.LargestTensors)
	}
	if a.KVCacheBytesAtFullContext <= 0 {
		t.Fatal("kv cache estimate missing")
	}
	var buf bytes.Buffer
	a.WriteText(&buf)
	for _, want := range []string{"architecture:   llama", "tensor types:", "token_embd.weight"} {
		if !strings.Contains(buf.String(), want) {
			t.Fatalf("report missing %q:\n%s", want, buf.String())
		}
	}
}

func TestSearchTokensExactBeatsPartial(t *testing.T) {
	tok := newInstTestTokenizer()
	matches := SearchTokens(tok, "a", 10)
	if len(matches) == 0 {
		t.Fatal("no matches")
	}
	if strings.TrimSpace(matches[0].Text) != "a" {
		t.Fatalf("first match = %q, want the exact token", matches[0].Text)
	}
	if len(SearchTokens(tok, "definitely-not-in-vocab", 10)) != 0 {
		t.Fatal("expected no matches for absent text")
	}
}

func TestNearestTokensTinyModel(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	matches, err := m.NearestTokens("3", 5) // numeric id form
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 5 {
		t.Fatalf("matches = %d, want 5", len(matches))
	}
	for i, match := range matches {
		if match.ID == 3 {
			t.Fatal("self must be excluded")
		}
		if i > 0 && match.Score > matches[i-1].Score {
			t.Fatal("scores must be descending")
		}
		if match.Score < -1.0001 || match.Score > 1.0001 {
			t.Fatalf("cosine out of range: %v", match.Score)
		}
	}
	// Text form resolution.
	if _, err := m.NearestTokens("a", 3); err != nil {
		t.Fatalf("text-form lookup: %v", err)
	}
	// Out-of-range id errors cleanly.
	if _, err := m.Runner().NearestTokens(9999, 3); err == nil {
		t.Fatal("expected error for out-of-range id")
	}
}

func TestNearestTokensUsesArchitectureSpecificEmbeddingTables(t *testing.T) {
	// The specialized architectures keep their embeddings outside
	// Runner.standard. Give that fallback a deliberately different nearest
	// neighbor so this checks the public token-neighborhood path rather than
	// merely the private selector.
	standard := Weight{F32: []float32{
		1, 0,
		1, 0,
		0, 1,
	}}
	specialized := Weight{F32: []float32{
		1, 0,
		0, 1,
		1, 1,
	}}
	tests := []struct {
		name string
		kind loadedKind
		set  func(*Runner)
	}{
		{
			name: "bert",
			kind: loadedBERT,
			set:  func(r *Runner) { r.bert.TokenEmbd = specialized },
		},
		{
			name: "nemotron_h",
			kind: loadedNemotronH,
			set:  func(r *Runner) { r.nemotronH.TokenEmbd = specialized },
		},
		{
			name: "mamba2",
			kind: loadedMamba2,
			set:  func(r *Runner) { r.mamba2.TokenEmbd = specialized },
		},
		{
			name: "qwen35",
			kind: loadedQwen35,
			set:  func(r *Runner) { r.qwen35.TokenEmbd = specialized },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Runner{
				kind:     tt.kind,
				config:   Config{Dim: 2, VocabSize: 3},
				tok:      &Tokenizer{Vocab: []string{"zero", "one", "two"}},
				standard: ModelWeights{TokenEmbd: standard},
			}
			tt.set(r)
			matches, err := r.NearestTokens(0, 1)
			if err != nil {
				t.Fatal(err)
			}
			if len(matches) != 1 || matches[0].ID != 2 {
				t.Fatalf("nearest = %#v, want token 2 from the specialized embedding table", matches)
			}
		})
	}
}

func TestAnalyzeKVCacheUsesHybridPhysicalDepth(t *testing.T) {
	g, err := ParseGGUFQuiet(buildTinyQwen35MoEGGUF(false))
	if err != nil {
		t.Fatal(err)
	}
	a := AnalyzeGGUF(g, nil)
	// The tiny fixture has one DeltaNet block and one full-attention block.
	// Only the latter reserves K/V rows; treating both as attention doubles
	// the reported footprint.
	if a.KVCacheLayers != 1 {
		t.Fatalf("KV cache layers = %d, want 1", a.KVCacheLayers)
	}
	if want := int64(1 * (4 + 4) * 32 * 4); a.KVCacheBytesAtFullContext != want {
		t.Fatalf("f32 KV cache = %d, want %d", a.KVCacheBytesAtFullContext, want)
	}
	if want := int64(1 * (4 + 4) * 32 * 2); a.KVCacheBytesAtFullContextF16 != want {
		t.Fatalf("f16 KV cache = %d, want %d", a.KVCacheBytesAtFullContextF16, want)
	}
	if a.KVCacheBytesAtFullContextI8 != 0 {
		t.Fatalf("unaligned K/V dimensions must not report an I8 cache: %d", a.KVCacheBytesAtFullContextI8)
	}
}

func TestAnalyzeMambaReportsNoKVCache(t *testing.T) {
	g, err := ParseGGUFQuiet(buildTinyMamba2GGUF(0))
	if err != nil {
		t.Fatal(err)
	}
	a := AnalyzeGGUF(g, nil)
	if a.KVCacheLayers != 0 || a.KVCacheBytesAtFullContext != 0 || a.KVCacheBytesAtFullContextF16 != 0 {
		t.Fatalf("Mamba2 should not report a K/V cache: %+v", a)
	}
}
