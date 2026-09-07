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
