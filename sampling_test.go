package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
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

func TestMinPExcludesLowProbabilityTokens(t *testing.T) {
	// exp(9-10) ≈ 0.368 of the peak; a 0.5 threshold drops everything but token 0.
	rng := NewRng(42)
	scratch := make([]TokenProb, 0, 8)

	// Full path (top_k disabled, top_p disabled): min-p must still apply.
	full := SamplerConfig{Temperature: 1, TopP: 1, TopK: 0, MinP: 0.5, RepeatPenalty: 1}
	for range 64 {
		logits := []float32{10, 9, 1, 0}
		if token := SampleWithScratch(logits, full, rng, nil, &scratch); token != 0 {
			t.Fatalf("full-path min_p token = %d, want 0", token)
		}
	}

	// Top-k path exercises the other cutoff branch.
	topk := SamplerConfig{Temperature: 1, TopP: 1, TopK: 3, MinP: 0.5, RepeatPenalty: 1}
	for range 64 {
		logits := []float32{10, 9, 1, 0}
		if token := SampleWithScratch(logits, topk, rng, nil, &scratch); token != 0 {
			t.Fatalf("top-k min_p token = %d, want 0", token)
		}
	}

	// A gentler threshold keeps the top two but never the tail.
	gentle := SamplerConfig{Temperature: 1, TopP: 1, TopK: 0, MinP: 0.3, RepeatPenalty: 1}
	for range 128 {
		logits := []float32{10, 9, 1, 0}
		token := SampleWithScratch(logits, gentle, rng, nil, &scratch)
		if token != 0 && token != 1 {
			t.Fatalf("gentle min_p token = %d, want 0 or 1", token)
		}
	}
}

func TestMinPFullVocabMatchesFullSort(t *testing.T) {
	profiles := [][]float32{
		{1},
		{1, 1, 1, 1},
		{1, 0.05, math.Nextafter32(0.05, 0), math.Nextafter32(0.05, 1), 0},
		{0, 2, 0.6, 2, 0.000001, 0.1},
	}
	r := rand.New(rand.NewSource(42))
	for _, vocab := range []int{19, 1024} {
		weights := make([]float32, vocab)
		for i := range weights {
			weights[i] = float32(math.Exp(r.NormFloat64() * 2))
		}
		profiles = append(profiles, weights)
	}
	for profile, weights := range profiles {
		for _, minP := range []float32{1e-5, 0.05, 0.3, 1} {
			for _, topP := range []float32{0.1, 0.5, 0.9, 1} {
				t.Run(fmt.Sprintf("profile=%d/minP=%g/topP=%g", profile, minP, topP), func(t *testing.T) {
					// Independent reference: sort every token, then apply min-p
					// and nucleus cutoffs. No early candidate pruning is used.
					reference := make([]TokenProb, len(weights))
					var total float32
					for i, weight := range weights {
						reference[i] = TokenProb{i, weight}
						total += weight
					}
					slices.SortFunc(reference, func(a, b TokenProb) int {
						if a.Prob > b.Prob || a.Prob == b.Prob && a.Token < b.Token {
							return -1
						}
						if a == b {
							return 0
						}
						return 1
					})
					cutoff := 1
					threshold := minP * reference[0].Prob
					for cutoff < len(reference) && reference[cutoff].Prob >= threshold {
						cutoff++
					}
					reference = reference[:cutoff]
					var mass float32
					for _, item := range reference {
						mass += item.Prob
					}
					var kept float32
					for i, item := range reference {
						kept += item.Prob
						if kept > topP*mass {
							reference = reference[:i+1]
							break
						}
					}
					scratch := make([]TokenProb, 3, len(weights)+3)
					wantRng, rng := NewRng(42), NewRng(42)
					for drawIndex := 0; drawIndex < 128; drawIndex++ {
						draw := wantRng.NextF32() * kept
						want := uint32(reference[len(reference)-1].Token)
						var cumulative float32
						for _, item := range reference {
							cumulative += item.Prob
							if cumulative > draw {
								want = uint32(item.Token)
								break
							}
						}
						got := sampleTopPFromWeights(weights, total, topP, minP, rng, &scratch)
						if got != want || rng.state != wantRng.state {
							t.Fatalf("draw %d: token/state = %d/%x, want %d/%x", drawIndex, got, rng.state, want, wantRng.state)
						}
					}
				})
			}
		}
	}
}

func TestSampleFullDistributionPath(t *testing.T) {
	// TopK=0, TopP=1, MinP=0 routes through the normalize + sampleFromProbs path.
	cfg := SamplerConfig{Temperature: 1, TopP: 1, TopK: 0, MinP: 0, RepeatPenalty: 1}
	rng := NewRng(1)
	scratch := make([]TokenProb, 0, 4)
	for range 32 {
		logits := []float32{100, 0, 0, 0} // sharply peaked -> token 0
		if token := SampleWithScratch(logits, cfg, rng, nil, &scratch); token != 0 {
			t.Fatalf("token = %d, want 0", token)
		}
	}
}

func TestRepeatPenaltyDownweightsRecentTokens(t *testing.T) {
	// Greedy path: without penalty token 0 wins; a strong penalty on token 0
	// hands the win to token 1.
	rng := NewRng(1)
	base := SamplerConfig{Temperature: 0, TopP: 1, TopK: 0, MinP: 0, RepeatPenalty: 1}
	if tok := Sample([]float32{5, 4, 0}, base, rng, []uint32{0}); tok != 0 {
		t.Fatalf("no-penalty greedy token = %d, want 0", tok)
	}
	pen := base
	pen.RepeatPenalty = 4
	if tok := Sample([]float32{5, 4, 0}, pen, rng, []uint32{0}); tok != 1 {
		t.Fatalf("penalized greedy token = %d, want 1", tok)
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

func BenchmarkSampleMinPFullVocabWithScratch(b *testing.B) {
	for _, vocab := range []int{32000, 131072} {
		b.Run(fmt.Sprintf("vocab=%d", vocab), func(b *testing.B) {
			config := SamplerConfig{Temperature: 1, TopP: 0.9, MinP: 0.05, RepeatPenalty: 1}
			base := make([]float32, vocab)
			r := rand.New(rand.NewSource(42))
			for i := range base {
				base[i] = float32(r.NormFloat64() * 2)
			}
			logits := make([]float32, len(base))
			scratch := make([]TokenProb, 0, len(base))
			rng := NewRng(1)
			b.ReportAllocs()
			for b.Loop() {
				copy(logits, base)
				SampleWithScratch(logits, config, rng, nil, &scratch)
			}
		})
	}
}
