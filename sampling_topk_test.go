package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestTopKMatchesFullSort(t *testing.T) {
	for _, size := range []int{3, 41, 1000} {
		for _, k := range []int{2, 40, 64} {
			if k >= size {
				continue
			}
			t.Run(fmt.Sprintf("vocab=%d/k=%d", size, k), func(t *testing.T) {
				r := rand.New(rand.NewSource(42))
				for trial := 0; trial < 20; trial++ {
					logits := make([]float32, size)
					reference := make([]TokenProb, 0, size)
					for i := range logits {
						logits[i] = float32(r.Intn(20) - 10) // Deliberately exercise ties.
						if i%11 == 0 {
							logits[i] = float32(math.NaN())
						}
						if i%13 == 0 {
							logits[i] = float32(math.Inf(1))
						}
						if finiteLogit(logits[i]) {
							reference = append(reference, TokenProb{i, logits[i]})
						}
					}
					slices.SortFunc(reference, func(a, b TokenProb) int {
						if a.Prob > b.Prob || (a.Prob == b.Prob && a.Token < b.Token) {
							return -1
						}
						if a == b {
							return 0
						}
						return 1
					})
					reference = reference[:min(k, len(reference))]
					var scratch []TokenProb
					SampleWithScratch(logits, SamplerConfig{Temperature: 1, TopK: k, TopP: 1, RepeatPenalty: 1}, NewRng(1), nil, &scratch)
					if len(scratch) != len(reference) {
						t.Fatalf("candidate count: %d, want %d", len(scratch), len(reference))
					}
					for i := range reference {
						if scratch[i].Token != reference[i].Token {
							t.Fatalf("trial %d candidate %d: %d, want %d", trial, i, scratch[i].Token, reference[i].Token)
						}
					}
				}
			})
		}
	}
}

func TestTopKThresholdScanEdgeCases(t *testing.T) {
	negInf, posInf, nan := float32(math.Inf(-1)), float32(math.Inf(1)), float32(math.NaN())
	for _, tc := range []struct {
		name   string
		logits []float32
	}{
		{"ascending", []float32{-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1}},
		{"descending", []float32{9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1}},
		{"ties", []float32{-1, -1, 3, 3, 3, 2, 3, 3, 3, 2, 3}},
		{"signed_zero", []float32{0, float32(math.Copysign(0, -1)), 0, 0, 0, 0, 0, 0, 0}},
		{"nonfinite_tail", []float32{-3, -2, -1, 0, posInf, nan, negInf, 2, 3}},
		{"late_finite", []float32{nan, posInf, negInf, nan, 3, 2, 1, 0, -1}},
		{"one_finite", []float32{nan, posInf, negInf, nan, -5, negInf, nan, posInf, nan}},
		{"no_finite", []float32{nan, posInf, negInf, nan, nan, negInf, nan, posInf, nan}},
	} {
		for _, k := range []int{2, 8} {
			t.Run(fmt.Sprintf("%s/k=%d", tc.name, k), func(t *testing.T) {
				reference := make([]TokenProb, 0, len(tc.logits))
				for i, v := range tc.logits {
					if finiteLogit(v) {
						reference = append(reference, TokenProb{i, v})
					}
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
				reference = reference[:min(k, len(reference))]
				var sum float32
				if len(reference) > 0 {
					maxLogit := reference[0].Prob
					for i := range reference {
						reference[i].Prob = float32(math.Exp(float64((reference[i].Prob - maxLogit) * (1 / float32(0.7)))))
						sum += reference[i].Prob
					}
				}
				// A nonempty scratch slice must be reset and reused across calls,
				// including when there are fewer finite logits than K.
				scratch := make([]TokenProb, k, k)
				cfg := SamplerConfig{Temperature: 0.7, TopK: k, TopP: 1, RepeatPenalty: 1}
				for _, seed := range []uint64{1, 17, 999} {
					wantRng := NewRng(seed)
					var want uint32
					if len(reference) > 0 {
						r := wantRng.NextF32() * sum
						var cumulative float32
						want = uint32(reference[len(reference)-1].Token)
						for _, item := range reference {
							cumulative += item.Prob
							if cumulative > r {
								want = uint32(item.Token)
								break
							}
						}
					}
					rng := NewRng(seed)
					got := SampleWithScratch(slices.Clone(tc.logits), cfg, rng, nil, &scratch)
					if got != want || rng.state != wantRng.state {
						t.Fatalf("seed %d: token/state = %d/%x, want %d/%x", seed, got, rng.state, want, wantRng.state)
					}
					if !slices.Equal(scratch, reference) {
						t.Fatalf("candidates = %v, want %v", scratch, reference)
					}
				}
			})
		}
	}
}

func BenchmarkSampleTopKLargeVocab(b *testing.B) {
	for _, tc := range []struct{ vocab, k int }{{32000, 40}, {131072, 20}, {131072, 64}} {
		b.Run(fmt.Sprintf("vocab=%d/k=%d", tc.vocab, tc.k), func(b *testing.B) {
			logits := make([]float32, tc.vocab)
			r := rand.New(rand.NewSource(42))
			for i := range logits {
				logits[i] = float32(r.NormFloat64() * 4)
			}
			scratch := make([]TokenProb, 0, tc.k)
			config := SamplerConfig{Temperature: 0.7, TopK: tc.k, TopP: 0.9, RepeatPenalty: 1}
			rng := NewRng(1)
			b.ReportAllocs()
			for b.Loop() {
				SampleWithScratch(logits, config, rng, nil, &scratch)
			}
		})
	}
}

func BenchmarkSampleTopKAscending(b *testing.B) {
	for _, k := range []int{40, 256, 1024} {
		b.Run(fmt.Sprint(k), func(b *testing.B) {
			logits := make([]float32, 32000)
			for i := range logits {
				logits[i] = float32(i) / 32000
			}
			scratch := make([]TokenProb, 0, k)
			config := SamplerConfig{Temperature: 1, TopK: k, TopP: 0.9, RepeatPenalty: 1}
			rng := NewRng(1)
			b.ReportAllocs()
			for b.Loop() {
				SampleWithScratch(logits, config, rng, nil, &scratch)
			}
		})
	}
}
