package main

import "testing"

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
