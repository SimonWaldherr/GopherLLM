package gopherllm

// Token sampling. The pipeline applied to each step's logits, in order:
//
//  1. repeat penalty on tokens in the recent window (divides positive
//     logits, multiplies negative ones — the llama.cpp convention);
//  2. temperature == 0 (or top-k == 1) short-circuits to greedy argmax;
//  3. otherwise softmax with temperature, restricted to the top-k
//     candidates when top-k > 0;
//  4. min-p: drop candidates below MinP * (probability of the best token);
//  5. top-p (nucleus): keep the smallest prefix of the sorted candidates
//     whose cumulative mass exceeds TopP, then sample from it.
//
// Non-finite logits (NaN/±Inf, which a numerical blow-up upstream would
// produce) are treated as -Inf everywhere so they can never be sampled.

import (
	"math"
)

var negInf32 = float32(math.Inf(-1))

// Rng is a xorshift64 generator. Generation is fully deterministic for a
// fixed seed (the CLI's --seed): the same seed replays the same tokens.
type Rng struct{ state uint64 }

// NewRng seeds an Rng; a zero seed is remapped to a fixed non-zero constant
// because xorshift has an all-zero fixed point.
func NewRng(seed uint64) *Rng {
	if seed == 0 {
		seed = 0xDEAD_BEEF_CAFE_1337
	}
	return &Rng{state: seed}
}

// NextF32 returns a uniform float in [0, 1) with 24 bits of precision.
func (r *Rng) NextF32() float32 {
	r.state ^= r.state << 13
	r.state ^= r.state >> 7
	r.state ^= r.state << 17
	return float32(r.state>>40) / float32(uint64(1)<<24)
}

// SamplerConfig holds the user-tunable sampling knobs. Disabled values:
// TopK <= 0 (consider all tokens), TopP >= 1, MinP <= 0, RepeatPenalty == 1.
// Temperature 0 means greedy decoding. See the file comment above for how
// the knobs compose, and GenerationOptions.Validate for the accepted ranges.
type SamplerConfig struct {
	Temperature   float32
	TopP          float32
	TopK          int
	MinP          float32
	RepeatPenalty float32
}

// DefaultSamplerConfig is a generic chat baseline. Model vendors publish
// family-specific recommendations that override these (e.g. Gemma wants
// temp 1.0 / top-p 0.95 / top-k 64 — see docs/INFERENCE_NOTES.md).
func DefaultSamplerConfig() SamplerConfig {
	return SamplerConfig{Temperature: 0.7, TopP: 0.9, TopK: 40, MinP: 0, RepeatPenalty: 1.1}
}

// TokenProb pairs a token id with its (possibly unnormalized) weight while a
// candidate set is being filtered and sampled.
type TokenProb struct {
	Token int
	Prob  float32
}

// Sample picks the next token id from logits. recent is the trailing token
// window the repeat penalty applies to. NOTE: logits is mutated in place
// (penalties and softmax are applied destructively) — callers must not reuse
// the slice's prior contents afterwards.
func Sample(logits []float32, config SamplerConfig, rng *Rng, recent []uint32) uint32 {
	candidates := []TokenProb{}
	return SampleWithScratch(logits, config, rng, recent, &candidates)
}

// SampleWithScratch is Sample with a caller-owned candidate scratch buffer so
// the per-token decode loop allocates nothing. Same logits-mutation caveat.
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
					if !finiteLogit(v) {
						logits[tok] = negInf32
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
	if config.RepeatPenalty != 1 {
		for _, tok := range recent {
			if int(tok) < n {
				v := logits[tok]
				if !finiteLogit(v) {
					logits[tok] = negInf32
				} else if v > 0 {
					logits[tok] = v / config.RepeatPenalty
				} else {
					logits[tok] = v * config.RepeatPenalty
				}
			}
		}
	}
	invTemp := 1 / config.Temperature
	if config.TopK > 0 && config.TopK < n {
		if config.TopK == 1 {
			return argmaxFiniteToken(logits)
		}
		return sampleTopK(logits, config.TopK, config.TopP, config.MinP, invTemp, rng, candidates)
	}
	maxLogit := negInf32
	for i, v := range logits {
		if !finiteLogit(v) {
			logits[i] = negInf32
			continue
		}
		maxLogit = max(maxLogit, v)
	}
	var sum float32
	if invTemp == 1 {
		for i, v := range logits {
			p := float32(math.Exp(float64(v - maxLogit)))
			logits[i] = p
			sum += p
		}
	} else {
		for i, v := range logits {
			p := float32(math.Exp(float64((v - maxLogit) * invTemp)))
			logits[i] = p
			sum += p
		}
	}
	if sum <= 0 || math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
		return argmaxToken(logits)
	}
	if config.TopP < 1 || config.MinP > 0 {
		return sampleTopPFromWeights(logits, sum, config.TopP, config.MinP, rng, candidates)
	}
	for i := range logits {
		logits[i] /= sum
	}
	return sampleFromProbs(logits, rng)
}

// sampleFromProbs draws from an already-normalized distribution.
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

// sampleTopPFromWeights applies min-p then top-p over the full-vocabulary
// weight array (the TopK<=0 path) and samples from what remains. weights are
// unnormalized softmax terms; total is their sum.
func sampleTopPFromWeights(weights []float32, total, topP, minP float32, rng *Rng, candidates *[]TokenProb) uint32 {
	ensureLenNoClear(candidates, len(weights))
	for i, w := range weights {
		(*candidates)[i] = TokenProb{i, w}
	}
	sortTokenProbs(*candidates)
	cutoff := len(*candidates)
	if minP > 0 && cutoff > 0 {
		threshold := minP * (*candidates)[0].Prob
		total = 0
		for i, item := range *candidates {
			if i > 0 && item.Prob < threshold {
				cutoff = i
				break
			}
			total += item.Prob
		}
	}
	target := topP * total
	var cumsum float32
	for i := 0; i < cutoff; i++ {
		cumsum += (*candidates)[i].Prob
		if cumsum > target {
			cutoff = i + 1
			break
		}
	}
	kept := cumsum
	if kept <= 0 || math.IsNaN(float64(kept)) || math.IsInf(float64(kept), 0) {
		return argmaxToken(weights)
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

// sampleTopK selects the top-K logits with a bounded insertion pass (no full
// vocab sort), softmaxes just those, then applies min-p and top-p within the
// candidate set. This is the common path: K is ~40-64 while vocabularies run
// 32K-262K, so avoiding the full-vocab sort dominates sampler cost.
func sampleTopK(logits []float32, topK int, topP, minP, invTemp float32, rng *Rng, candidates *[]TokenProb) uint32 {
	*candidates = (*candidates)[:0]
	for i, logit := range logits {
		if len(*candidates) < topK {
			if !finiteLogit(logit) {
				logit = negInf32
			}
			*candidates = append(*candidates, TokenProb{i, logit})
			bubbleUpLast(*candidates)
		} else if logit > (*candidates)[len(*candidates)-1].Prob {
			if !finiteLogit(logit) {
				continue
			}
			(*candidates)[len(*candidates)-1] = TokenProb{i, logit}
			bubbleUpLast(*candidates)
		}
	}
	if len(*candidates) == 0 || math.IsInf(float64((*candidates)[0].Prob), -1) {
		return argmaxFiniteToken(logits)
	}
	maxLogit := (*candidates)[0].Prob
	var sum float32
	for i := range *candidates {
		(*candidates)[i].Prob = float32(math.Exp(float64(((*candidates)[i].Prob - maxLogit) * invTemp)))
		sum += (*candidates)[i].Prob
	}
	if sum <= 0 {
		return uint32((*candidates)[0].Token)
	}
	cutoff := len(*candidates)
	kept := sum
	if minP > 0 {
		threshold := minP * (*candidates)[0].Prob
		var s float32
		for i, item := range *candidates {
			if i > 0 && item.Prob < threshold {
				cutoff = i
				break
			}
			s += item.Prob
		}
		sum = s
		kept = s
	}
	if topP < 1 {
		target := topP * sum
		var cumsum float32
		for i := 0; i < cutoff; i++ {
			cumsum += (*candidates)[i].Prob
			if cumsum > target {
				cutoff = i + 1
				break
			}
		}
		kept = cumsum
	}
	if kept <= 0 {
		return uint32((*candidates)[0].Token)
	}
	r := rng.NextF32() * kept
	var cumsum float32
	for _, item := range (*candidates)[:cutoff] {
		cumsum += item.Prob
		if cumsum > r {
			return uint32(item.Token)
		}
	}
	return uint32((*candidates)[cutoff-1].Token)
}

// bubbleUpLast restores descending-probability order after appending or
// replacing the last element of an otherwise-sorted candidate slice.
func bubbleUpLast(c []TokenProb) {
	for i := len(c) - 1; i > 0 && c[i].Prob > c[i-1].Prob; i-- {
		c[i], c[i-1] = c[i-1], c[i]
	}
}

// sortTokenProbs sorts by probability descending, token id ascending as the
// tie-break (a stable, deterministic order for equal probabilities). Hand-
// rolled quicksort+insertion because sort.Slice allocates its closure on a
// per-decoded-token hot path.
func sortTokenProbs(c []TokenProb) {
	if len(c) < 2 {
		return
	}
	quickSortTokenProbs(c, 0, len(c)-1)
}

func quickSortTokenProbs(c []TokenProb, lo, hi int) {
	for hi-lo > 16 {
		mid := lo + (hi-lo)/2
		if tokenProbLess(c[mid], c[lo]) {
			c[mid], c[lo] = c[lo], c[mid]
		}
		if tokenProbLess(c[hi], c[mid]) {
			c[hi], c[mid] = c[mid], c[hi]
			if tokenProbLess(c[mid], c[lo]) {
				c[mid], c[lo] = c[lo], c[mid]
			}
		}
		pivot := c[mid]
		i, j := lo, hi
		for {
			for tokenProbLess(c[i], pivot) {
				i++
			}
			for tokenProbLess(pivot, c[j]) {
				j--
			}
			if i >= j {
				break
			}
			c[i], c[j] = c[j], c[i]
			i++
			j--
		}
		if j-lo < hi-i {
			quickSortTokenProbs(c, lo, j)
			lo = i
		} else {
			quickSortTokenProbs(c, i, hi)
			hi = j
		}
	}
	insertionSortTokenProbs(c[lo : hi+1])
}

func insertionSortTokenProbs(c []TokenProb) {
	for i := 1; i < len(c); i++ {
		v := c[i]
		j := i - 1
		for ; j >= 0 && tokenProbLess(v, c[j]); j-- {
			c[j+1] = c[j]
		}
		c[j+1] = v
	}
}

func tokenProbLess(a, b TokenProb) bool {
	if a.Prob != b.Prob {
		return a.Prob > b.Prob
	}
	return a.Token < b.Token
}

func finiteLogit(v float32) bool {
	return math.Float32bits(v)&0x7f800000 != 0x7f800000
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
	if !finiteLogit(bestValue) {
		bestValue = negInf32
		logits[0] = bestValue
	}
	for i := 1; i < len(logits); i++ {
		v := logits[i]
		if !finiteLogit(v) {
			v = negInf32
			logits[i] = v
		}
		if v > bestValue {
			best = i
			bestValue = v
		}
	}
	return uint32(best)
}
