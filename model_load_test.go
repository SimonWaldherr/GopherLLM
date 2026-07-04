package main

import (
	"math"
	"testing"
)

func TestConfigFromTinyGGUF(t *testing.T) {
	g, err := ParseGGUFQuiet(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	cfg := ConfigFromGGUF(g)
	if cfg.Arch != "llama" {
		t.Fatalf("arch = %q", cfg.Arch)
	}
	if cfg.Dim != 8 || cfg.NLayers != 1 || cfg.NHeads != 2 || cfg.NKVHeads != 2 {
		t.Fatalf("config = %+v", cfg)
	}
	if cfg.HiddenDim != 16 || cfg.VocabSize < 16 {
		t.Fatalf("hidden=%d vocab=%d", cfg.HiddenDim, cfg.VocabSize)
	}
}

func TestLoadAndGenerateTinyModel(t *testing.T) {
	runner, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if runner.Architecture() != "llama" {
		t.Fatalf("arch = %q", runner.Architecture())
	}
	cfg := runner.Config()
	if cfg.HeadDim != 4 { // inferred from tensor shapes
		t.Fatalf("head_dim = %d, want 4", cfg.HeadDim)
	}

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 4
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	res, err := runner.Generate("a b c", opts)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if res.Stats.PromptTokens < 1 {
		t.Fatalf("prompt tokens = %d", res.Stats.PromptTokens)
	}
	if res.Stats.GeneratedTokens < 1 || res.Stats.GeneratedTokens > 4 {
		t.Fatalf("generated tokens = %d", res.Stats.GeneratedTokens)
	}
}

func TestGenerateTinyModelDeterministic(t *testing.T) {
	runner, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 6
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	a, err := runner.Generate("hello", opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := runner.Generate("hello", opts)
	if err != nil {
		t.Fatal(err)
	}
	if a.Text != b.Text {
		t.Fatalf("greedy output not deterministic: %q vs %q", a.Text, b.Text)
	}
}

func TestWeightAndForwardWrappers(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	dim := r.config.Dim

	if row := r.standard.TokenEmbd.Row(0, dim); len(row) != dim {
		t.Fatalf("Row len = %d, want %d", len(row), dim)
	}
	if rf := r.standard.TokenEmbd.RowF32(1, dim); len(rf) != dim {
		t.Fatalf("RowF32 len = %d, want %d", len(rf), dim)
	}
	xn := make([]float32, dim)
	for i := range xn {
		xn[i] = 0.1
	}
	if logits := r.standard.Output.Matvec(xn); len(logits) != r.config.VocabSize {
		t.Fatalf("Matvec len = %d, want %d", len(logits), r.config.VocabSize)
	}

	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 4)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	logits := Forward(r.config, r.standard, cache, buf, 3, 0)
	if len(logits) != r.config.VocabSize {
		t.Fatalf("Forward logits len = %d, want %d", len(logits), r.config.VocabSize)
	}
}

func TestEmbedTinyModel(t *testing.T) {
	runner, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	emb, err := runner.Embed("a b c")
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(emb.Embedding) != runner.Config().Dim {
		t.Fatalf("embedding dim = %d, want %d", len(emb.Embedding), runner.Config().Dim)
	}
	var ss float64
	for _, v := range emb.Embedding {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite embedding value %v", v)
		}
		ss += float64(v) * float64(v)
	}
	// mean-pooled then L2-normalized => unit norm (or zero if degenerate).
	if norm := math.Sqrt(ss); norm > 1e-6 && math.Abs(norm-1) > 1e-3 {
		t.Fatalf("embedding norm = %v, want ~1", norm)
	}
}
