package main

import (
	"math"
	"testing"
)

func TestTopKOneOnlyKeepsSingleBestToken(t *testing.T) {
	config := SamplerConfig{Temperature: 1, TopP: 1, TopK: 1, RepeatPenalty: 1}
	rng := NewRng(42)
	for range 64 {
		logits := []float32{1, 10, 9}
		token := Sample(logits, config, rng, nil)
		if token != 1 {
			t.Fatalf("token = %d, want 1", token)
		}
	}
}

func TestTopKIgnoresNonFiniteLogits(t *testing.T) {
	config := SamplerConfig{Temperature: 1, TopP: 1, TopK: 1, RepeatPenalty: 1}
	rng := NewRng(42)
	logits := []float32{1, float32(math.Inf(1)), 2}
	if token := Sample(logits, config, rng, nil); token != 2 {
		t.Fatalf("token = %d, want 2", token)
	}

	logits = []float32{float32(math.Inf(-1)), float32(math.Inf(1))}
	if token := Sample(logits, config, rng, nil); token != 0 {
		t.Fatalf("all non-finite token = %d, want 0", token)
	}
}

func TestEmptyLogitsReturnsZeroToken(t *testing.T) {
	config := DefaultSamplerConfig()
	rng := NewRng(7)
	token := Sample(nil, config, rng, nil)
	if token != 0 {
		t.Fatalf("token = %d, want 0", token)
	}
}

func TestTopPSlicesTopKCandidateSet(t *testing.T) {
	config := SamplerConfig{Temperature: 1, TopP: 0.6, TopK: 2, RepeatPenalty: 1}
	rng := NewRng(42)
	for range 64 {
		logits := []float32{10, 9, 0, -1}
		token := Sample(logits, config, rng, nil)
		if token != 0 {
			t.Fatalf("token = %d, want 0", token)
		}
	}
}

func TestTopPWithoutTopKUsesTopMassOnly(t *testing.T) {
	config := SamplerConfig{Temperature: 1, TopP: 0.6, TopK: 0, RepeatPenalty: 1}
	rng := NewRng(42)
	scratch := make([]TokenProb, 0, 4)
	for range 64 {
		logits := []float32{10, 9, 0, -1}
		token := SampleWithScratch(logits, config, rng, nil, &scratch)
		if token != 0 {
			t.Fatalf("token = %d, want 0", token)
		}
	}
	if cap(scratch) < 4 {
		t.Fatalf("scratch capacity = %d, want at least 4", cap(scratch))
	}
}

func TestSortTokenProbsOrdersByProbabilityThenToken(t *testing.T) {
	candidates := []TokenProb{
		{Token: 4, Prob: 0.2},
		{Token: 2, Prob: 0.7},
		{Token: 1, Prob: 0.7},
		{Token: 3, Prob: 0.4},
	}
	sortTokenProbs(candidates)
	want := []TokenProb{
		{Token: 1, Prob: 0.7},
		{Token: 2, Prob: 0.7},
		{Token: 3, Prob: 0.4},
		{Token: 4, Prob: 0.2},
	}
	for i := range want {
		if candidates[i] != want[i] {
			t.Fatalf("candidates[%d] = %+v, want %+v", i, candidates[i], want[i])
		}
	}
}

func BenchmarkSampleTopPFullVocabWithScratch(b *testing.B) {
	config := SamplerConfig{Temperature: 1, TopP: 0.9, TopK: 0, RepeatPenalty: 1}
	base := make([]float32, 32000)
	for i := range base {
		base[i] = float32(i%997) / 997
	}
	logits := make([]float32, len(base))
	scratch := make([]TokenProb, 0, len(base))
	rng := NewRng(1)
	b.ReportAllocs()
	for b.Loop() {
		copy(logits, base)
		_ = SampleWithScratch(logits, config, rng, nil, &scratch)
	}
}

func BenchmarkSampleDefaultTopKWithScratch(b *testing.B) {
	config := DefaultSamplerConfig()
	base := make([]float32, 32000)
	for i := range base {
		base[i] = float32(i%997) / 997
	}
	logits := make([]float32, len(base))
	scratch := make([]TokenProb, 0, config.TopK)
	rng := NewRng(1)
	b.ReportAllocs()
	for b.Loop() {
		copy(logits, base)
		_ = SampleWithScratch(logits, config, rng, nil, &scratch)
	}
}

func BenchmarkSampleTopKOneWithScratch(b *testing.B) {
	config := SamplerConfig{Temperature: 1, TopP: 1, TopK: 1, RepeatPenalty: 1}
	base := make([]float32, 32000)
	for i := range base {
		base[i] = float32(i%997) / 997
	}
	logits := make([]float32, len(base))
	scratch := make([]TokenProb, 0, config.TopK)
	rng := NewRng(1)
	b.ReportAllocs()
	for b.Loop() {
		copy(logits, base)
		_ = SampleWithScratch(logits, config, rng, nil, &scratch)
	}
}
