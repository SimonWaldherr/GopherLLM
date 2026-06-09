package main

import (
	"math"
	"slices"
)

type Rng struct{ state uint64 }

func NewRng(seed uint64) *Rng {
	if seed == 0 {
		seed = 0xDEAD_BEEF_CAFE_1337
	}
	return &Rng{state: seed}
}

func (r *Rng) NextF32() float32 {
	r.state ^= r.state << 13
	r.state ^= r.state >> 7
	r.state ^= r.state << 17
	return float32(r.state>>40) / float32(uint64(1)<<24)
}

type SamplerConfig struct {
	Temperature   float32
	TopP          float32
	TopK          int
	RepeatPenalty float32
}

func DefaultSamplerConfig() SamplerConfig {
	return SamplerConfig{Temperature: 0.7, TopP: 0.9, TopK: 40, RepeatPenalty: 1.1}
}

type TokenProb struct {
	Token int
	Prob  float32
}

func Sample(logits []float32, config SamplerConfig, rng *Rng, recent []uint32) uint32 {
	candidates := []TokenProb{}
	return SampleWithScratch(logits, config, rng, recent, &candidates)
}

func SampleWithScratch(logits []float32, config SamplerConfig, rng *Rng, recent []uint32, candidates *[]TokenProb) uint32 {
	n := len(logits)
	if n == 0 {
		return 0
	}
	if config.Temperature < 1e-6 {
		if config.RepeatPenalty != 1 {
			for _, tok := range recent {
				if int(tok) < n {
					v := logits[tok]
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						logits[tok] = float32(math.Inf(-1))
					} else if v > 0 {
						logits[tok] = v / config.RepeatPenalty
					} else {
						logits[tok] = v * config.RepeatPenalty
					}
				}
			}
		}
		return argmaxFiniteToken(logits)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			logits[i] = float32(math.Inf(-1))
		}
	}
	if config.RepeatPenalty != 1 {
		for _, tok := range recent {
			if int(tok) < n {
				if logits[tok] > 0 {
					logits[tok] /= config.RepeatPenalty
				} else {
					logits[tok] *= config.RepeatPenalty
				}
			}
		}
	}
	invTemp := 1 / config.Temperature
	for i := range logits {
		logits[i] *= invTemp
	}
	if config.TopK > 0 && config.TopK < n {
		return sampleTopK(logits, config.TopK, config.TopP, rng, candidates)
	}
	maxLogit := float32(math.Inf(-1))
	for _, v := range logits {
		maxLogit = max(maxLogit, v)
	}
	var sum float32
	for i, v := range logits {
		p := float32(math.Exp(float64(v - maxLogit)))
		logits[i] = p
		sum += p
	}
	if sum <= 0 || math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
		return argmaxToken(logits)
	}
	for i := range logits {
		logits[i] /= sum
	}
	if config.TopP < 1 {
		return sampleTopPFromProbs(logits, config.TopP, rng, candidates)
	}
	return sampleFromProbs(logits, rng)
}

func sampleFromProbs(probs []float32, rng *Rng) uint32 {
	r := rng.NextF32()
	var cumsum float32
	for i, p := range probs {
		cumsum += p
		if cumsum > r {
			return uint32(i)
		}
	}
	return uint32(len(probs) - 1)
}

func sampleTopPFromProbs(probs []float32, topP float32, rng *Rng, candidates *[]TokenProb) uint32 {
	*candidates = (*candidates)[:0]
	for i, p := range probs {
		*candidates = append(*candidates, TokenProb{i, p})
	}
	slices.SortFunc(*candidates, func(a, b TokenProb) int {
		if a.Prob > b.Prob {
			return -1
		}
		if a.Prob < b.Prob {
			return 1
		}
		if a.Token < b.Token {
			return -1
		}
		if a.Token > b.Token {
			return 1
		}
		return 0
	})
	cumsum := float32(0)
	cutoff := len(*candidates)
	for i, item := range *candidates {
		cumsum += item.Prob
		if cumsum > topP {
			cutoff = i + 1
			break
		}
	}
	var kept float32
	for _, item := range (*candidates)[:cutoff] {
		kept += item.Prob
	}
	if kept <= 0 {
		return argmaxToken(probs)
	}
	r := rng.NextF32() * kept
	cumsum = 0
	for _, item := range (*candidates)[:cutoff] {
		cumsum += item.Prob
		if cumsum > r {
			return uint32(item.Token)
		}
	}
	return uint32((*candidates)[cutoff-1].Token)
}

func sampleTopK(logits []float32, topK int, topP float32, rng *Rng, candidates *[]TokenProb) uint32 {
	*candidates = (*candidates)[:0]
	for i, logit := range logits {
		if len(*candidates) < topK {
			*candidates = append(*candidates, TokenProb{i, logit})
			bubbleUpLast(*candidates)
		} else if logit > (*candidates)[len(*candidates)-1].Prob {
			(*candidates)[len(*candidates)-1] = TokenProb{i, logit}
			bubbleUpLast(*candidates)
		}
	}
	if len(*candidates) == 0 || math.IsInf(float64((*candidates)[0].Prob), -1) {
		return argmaxToken(logits)
	}
	maxLogit := (*candidates)[0].Prob
	var sum float32
	for i := range *candidates {
		(*candidates)[i].Prob = float32(math.Exp(float64((*candidates)[i].Prob - maxLogit)))
		sum += (*candidates)[i].Prob
	}
	if sum <= 0 {
		return uint32((*candidates)[0].Token)
	}
	for i := range *candidates {
		(*candidates)[i].Prob /= sum
	}
	cutoff := len(*candidates)
	if topP < 1 {
		var cumsum float32
		for i, item := range *candidates {
			cumsum += item.Prob
			if cumsum > topP {
				cutoff = i + 1
				break
			}
		}
		var kept float32
		for _, item := range (*candidates)[:cutoff] {
			kept += item.Prob
		}
		if kept > 0 {
			for i := range (*candidates)[:cutoff] {
				(*candidates)[i].Prob /= kept
			}
		}
	}
	r := rng.NextF32()
	var cumsum float32
	for _, item := range (*candidates)[:cutoff] {
		cumsum += item.Prob
		if cumsum > r {
			return uint32(item.Token)
		}
	}
	return uint32((*candidates)[cutoff-1].Token)
}

func bubbleUpLast(c []TokenProb) {
	for i := len(c) - 1; i > 0 && c[i].Prob > c[i-1].Prob; i-- {
		c[i], c[i-1] = c[i-1], c[i]
	}
}

func argmaxToken(logits []float32) uint32 {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return uint32(best)
}

func argmaxFiniteToken(logits []float32) uint32 {
	best := 0
	bestValue := logits[0]
	if math.IsNaN(float64(bestValue)) || math.IsInf(float64(bestValue), 0) {
		bestValue = float32(math.Inf(-1))
		logits[0] = bestValue
	}
	for i := 1; i < len(logits); i++ {
		v := logits[i]
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			v = float32(math.Inf(-1))
			logits[i] = v
		}
		if v > bestValue {
			best = i
			bestValue = v
		}
	}
	return uint32(best)
}
