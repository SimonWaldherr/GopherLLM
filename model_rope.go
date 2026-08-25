package gopherllm

import (
	"math"
	"os"
)

func ropeScalingType(gguf *GGUFFile, p string) string {
	if os.Getenv("GOPHERLLM_DISABLE_YARN") != "" {
		return ""
	}
	if s, ok := gguf.GetString(p + ".rope.scaling.type"); ok {
		return s
	}
	return ""
}

// buildRopeInvFreq returns the per-pair inverse RoPE frequencies and the
// attention magnitude scale (mscale) applied to the rotated Q/K vectors. mscale
// is 1 except for YaRN-scaled models (e.g. Ministral) where the rotation is
// amplified to match how the model was trained.
func buildRopeInvFreq(config Config, maxHeadDim int) ([]float32, float32) {
	ropeDim := config.RopeDimensionCount
	if ropeDim <= 0 || ropeDim > maxHeadDim {
		ropeDim = maxHeadDim
	}
	pairs := ropeDim / 2
	if config.RopeScalingType == "yarn" && config.RopeScalingFactor > 1 && config.RopeOriginalContextLength > 0 {
		return buildRopeInvFreqYarn(config, ropeDim, pairs)
	}
	inv := make([]float32, pairs)
	factors := config.RopeFactorsShort
	if config.RopeOriginalContextLength > 0 && config.MaxSeqLen > config.RopeOriginalContextLength && len(config.RopeFactorsLong) >= pairs {
		factors = config.RopeFactorsLong
	}
	for pair := range pairs {
		i := float32(pair * 2)
		base := float32(math.Pow(float64(config.RopeTheta), float64(i/float32(ropeDim))))
		factor := float32(1)
		if pair < len(factors) && factors[pair] != 0 {
			factor = factors[pair]
		}
		inv[pair] = 1 / (factor * base)
	}
	return inv, 1
}

// buildRopeInvFreqYarn implements YaRN "NTK-by-parts" frequency interpolation
// and the attention magnitude scale, mirroring llama.cpp's rope_yarn. High
// frequencies (short wavelengths) are left untouched, low frequencies are
// interpolated by 1/factor, and a linear ramp blends the middle band.
func buildRopeInvFreqYarn(config Config, ropeDim, pairs int) ([]float32, float32) {
	inv := make([]float32, pairs)
	base := float64(config.RopeTheta)
	nDims := float64(ropeDim)
	nOrig := float64(config.RopeOriginalContextLength)
	factor := float64(config.RopeScalingFactor)
	freqScale := 1 / factor
	betaFast := float64(config.RopeYarnBetaFast)
	if betaFast <= 0 {
		betaFast = 32
	}
	betaSlow := float64(config.RopeYarnBetaSlow)
	if betaSlow <= 0 {
		betaSlow = 1
	}

	corrDim := func(nRot float64) float64 {
		return nDims * math.Log(nOrig/(nRot*2*math.Pi)) / (2 * math.Log(base))
	}
	low := math.Floor(corrDim(betaFast))
	high := math.Ceil(corrDim(betaSlow))
	low = math.Max(0, low)
	high = math.Min(nDims-1, high)
	denom := math.Max(0.001, high-low)

	for pair := 0; pair < pairs; pair++ {
		i := float64(pair * 2)
		freqExtrap := 1 / math.Pow(base, i/nDims)
		freqInterp := freqExtrap * freqScale
		y := (float64(pair) - low) / denom
		ramp := 1 - math.Min(1, math.Max(0, y)) // 1 => keep extrapolated, 0 => interpolate
		freq := freqInterp*(1-ramp) + freqExtrap*ramp
		inv[pair] = float32(freq)
	}
	// YaRN also defines an attention-magnitude scale (mscale = 1 + 0.1*ln(factor))
	// applied to the rotated Q/K. Enabling it measurably degraded Ministral output
	// at ordinary context lengths — attention became over-sharpened and greedy
	// decoding derailed (e.g. "Alphabet" -> "Al data"). The frequency
	// interpolation above is what actually extends usable context, so we keep it
	// and leave the magnitude scale at 1.
	return inv, 1
}

func buildRopeInvFreqGptOss(config Config) ([]float32, float32) {
	pairs := config.HeadDim / 2
	inv := make([]float32, pairs)
	concentration := float32(1)
	var low, high float32
	if config.RopeScalingFactor > 1 {
		dHalf := float32(config.HeadDim) / 2
		low = dHalf * float32(math.Log(float64(float32(config.RopeOriginalContextLength)/(32*2*math.Pi)))/math.Log(float64(config.RopeTheta)))
		high = dHalf * float32(math.Log(float64(float32(config.RopeOriginalContextLength)/(1*2*math.Pi)))/math.Log(float64(config.RopeTheta)))
		concentration = 0.1*float32(math.Log(float64(config.RopeScalingFactor))) + 1
	}
	for pair := range pairs {
		i := float32(pair * 2)
		base := float32(math.Pow(float64(config.RopeTheta), float64(i/float32(config.HeadDim))))
		if config.RopeScalingFactor > 1 && high != low {
			idx := float32(pair)
			ramp := clamp((idx-low)/(high-low), 0, 1)
			mask := 1 - ramp
			interpolation := 1 / (config.RopeScalingFactor * base)
			extrapolation := 1 / base
			inv[pair] = interpolation*(1-mask) + extrapolation*mask
		} else {
			inv[pair] = 1 / base
		}
	}
	return inv, concentration
}

// prepareRopeScratch fills sin/cos tables for one position from the
// precomputed inverse frequencies (optionally magnitude-scaled by mscale, the
// YaRN attention factor — currently always 1, see buildRopeInvFreqYarn).
// It returns the pair half-width and the number of cached pairs for
// applyPreparedRope, which rotates each head either in interleaved pair
// order (dims 2i,2i+1 — the original RoPE layout used by llama/mistral) or
// split-half order (dims i, i+half — the NeoX layout everything else uses);
// ropeInterleaved picks per architecture.
func prepareRopeScratch(pos, headDim, ropeDim int, invFreq []float32, mscale float32, sinScratch, cosScratch *[]float32) (int, int) {
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}
	ropeDim -= ropeDim % 2
	half := ropeDim / 2
	if half <= 0 {
		return 0, 0
	}

	nCache := min(half, len(invFreq))
	if nCache <= 0 {
		return half, 0
	}

	ensureLenNoClear(sinScratch, nCache)
	ensureLenNoClear(cosScratch, nCache)
	sin := *sinScratch
	cos := *cosScratch
	if mscale == 0 {
		mscale = 1
	}
	for i := range nCache {
		angle := float64(float32(pos) * invFreq[i])
		s64, c64 := math.Sincos(angle)
		sin[i] = float32(s64) * mscale
		cos[i] = float32(c64) * mscale
	}
	return half, nCache
}

func applyPreparedRope(vec []float32, headDim, nHeads, half, nCache int, sin, cos []float32, interleaved bool) {
	if nCache <= 0 {
		return
	}

	for h := range nHeads {
		off := h * headDim
		if off+headDim > len(vec) {
			break
		}
		sub := vec[off : off+headDim]
		_ = sub[headDim-1] // assert length of sub

		if interleaved {
			for i := 0; i < nCache; i++ {
				idx0, idx1 := i*2, i*2+1
				s, c := sin[i], cos[i]
				v0, v1 := sub[idx0], sub[idx1]
				sub[idx0] = v0*c - v1*s
				sub[idx1] = v0*s + v1*c
			}
		} else {
			for i := 0; i < nCache; i++ {
				idx0, idx1 := i, i+half
				s, c := sin[i], cos[i]
				v0, v1 := sub[idx0], sub[idx1]
				sub[idx0] = v0*c - v1*s
				sub[idx1] = v0*s + v1*c
			}
		}
	}
}

func ropeInterleaved(arch string) bool {
	switch arch {
	case "llama", "llama2", "llama3", "mistral", "mistral3", "mixtral", "ministral", "smollm3", "internlm2",
		"chatglm", "glm4", "gptj", "command-r", "minicpm", "granite", "granitemoe":
		return true
	default:
		return false
	}
}

// attentionTemperatureAt implements Mistral 3's position-dependent
// long-context temperature schedule. llama.cpp multiplies Q (after RoPE) by
// this value, using the original YaRN context length as the floor interval.
// A missing/zero metadata pair is deliberately inert for every other family.
func attentionTemperatureAt(config Config, pos int) float32 {
	if config.AttentionTemperatureScale == 0 || config.AttentionTemperatureFloor <= 0 || pos < 0 {
		return 1
	}
	step := math.Floor(float64(pos) / float64(config.AttentionTemperatureFloor))
	return 1 + config.AttentionTemperatureScale*float32(math.Log(step+1))
}

func clamp(v, lo, hi float32) float32 {
	return min(max(v, lo), hi)
}
