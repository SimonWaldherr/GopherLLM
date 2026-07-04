package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
)

// Config is the model's hyperparameter set, read from GGUF metadata by
// ConfigFromGGUF and then refined against actual tensor shapes by
// inferAttentionShape (GGUF metadata is frequently missing or wrong about
// head dims, so the tensor shapes are authoritative).
//
// Attention shape vocabulary used throughout the forward pass:
// HeadDim is the per-head Q/K width, ValueDim the per-head V width (usually
// equal), NKVHeads the number of K/V heads (< NHeads under grouped-query
// attention), KVMul = NHeads/NKVHeads the number of query heads sharing each
// KV head, and KVDim = NKVHeads*ValueDim the per-position V cache width.
// The scale factors (Embedding/Residual/Logit/Attention) default to 1 (or 0
// meaning "use 1/sqrt(HeadDim)" for AttentionScale) and are only non-trivial
// for architectures whose GGUFs carry them.
type Config struct {
	Arch                      string
	Dim                       int
	HiddenDim                 int
	NLayers                   int
	NHeads                    int
	NKVHeads                  int
	VocabSize                 int
	MaxSeqLen                 int
	RopeTheta                 float32
	RMSNormEps                float32
	AttentionScale            float32
	EmbeddingScale            float32
	ResidualScale             float32
	LogitScale                float32
	HeadDim                   int
	KVDim                     int
	KVMul                     int
	ValueDim                  int
	SlidingWindow             int
	ExpertCount               int
	ExpertUsedCount           int
	RopeDimensionCount        int
	RopeScalingFactor         float32
	RopeAttentionFactor       float32
	RopeOriginalContextLength int
	RopeScalingType           string
	RopeYarnBetaFast          float32
	RopeYarnBetaSlow          float32
	RopeYarnLogMultiplier     float32
	RopeFactorsLong           []float32
	RopeFactorsShort          []float32
}

func ConfigFromGGUF(gguf *GGUFFile) Config {
	arch, ok := gguf.GetString("general.architecture")
	if !ok || arch == "" {
		arch = "llama"
	}
	p := arch
	dim := int(gguf.GetU32(p+".embedding_length", 0))
	nHeads := int(gguf.GetU32(p+".attention.head_count", 0))
	nKVHeads := int(gguf.GetU32(p+".attention.head_count_kv", uint32(max(1, nHeads))))
	headDim := int(gguf.GetU32(p+".attention.key_length", 0))
	if nHeads > 0 {
		headDim = max(headDim, dim/nHeads)
	}
	valueDim := int(gguf.GetU32(p+".attention.value_length", uint32(max(1, headDim))))
	vocab := int(gguf.GetU32(p+".vocab_size", 0))
	if v, ok := gguf.Metadata["tokenizer.ggml.tokens"]; ok {
		if arr, ok := v.AsStringArray(); ok {
			vocab = max(vocab, len(arr))
		}
	}
	if nKVHeads <= 0 {
		nKVHeads = max(1, nHeads)
	}
	kvMul := 1
	if nKVHeads > 0 && nHeads > 0 {
		kvMul = max(1, nHeads/nKVHeads)
	}
	embeddingScale := gguf.GetF32(p+".embedding_scale", 1)
	if embeddingScale == 0 {
		embeddingScale = 1
	}
	residualScale := gguf.GetF32(p+".residual_scale", 1)
	if residualScale == 0 {
		residualScale = 1
	}
	logitScale := gguf.GetF32(p+".logit_scale", 1)
	if logitScale == 0 {
		logitScale = 1
	}
	return Config{
		Arch:                      p,
		Dim:                       dim,
		HiddenDim:                 int(gguf.GetU32(p+".feed_forward_length", 0)),
		NLayers:                   int(gguf.GetU32(p+".block_count", 0)),
		NHeads:                    nHeads,
		NKVHeads:                  nKVHeads,
		VocabSize:                 vocab,
		MaxSeqLen:                 int(gguf.GetU32(p+".context_length", 2048)),
		RopeTheta:                 gguf.GetF32(p+".rope.freq_base", 10000),
		RMSNormEps:                gguf.GetF32(p+".attention.layer_norm_rms_epsilon", 1e-5),
		AttentionScale:            gguf.GetF32(p+".attention.scale", 0),
		EmbeddingScale:            embeddingScale,
		ResidualScale:             residualScale,
		LogitScale:                logitScale,
		HeadDim:                   headDim,
		KVDim:                     valueDim * nKVHeads,
		KVMul:                     kvMul,
		ValueDim:                  valueDim,
		SlidingWindow:             int(gguf.GetU32(p+".attention.sliding_window", 0)),
		ExpertCount:               int(gguf.GetU32(p+".expert_count", 0)),
		ExpertUsedCount:           int(gguf.GetU32(p+".expert_used_count", 0)),
		RopeDimensionCount:        int(gguf.GetU32(p+".rope.dimension_count", uint32(max(1, headDim)))),
		RopeScalingFactor:         gguf.GetF32(p+".rope.scaling.factor", 1),
		RopeAttentionFactor:       gguf.GetF32(p+".rope.scaling.attn_factor", 1),
		RopeOriginalContextLength: int(gguf.GetU32(p+".rope.scaling.original_context_length", 0)),
		RopeScalingType:           ropeScalingType(gguf, p),
		RopeYarnBetaFast:          gguf.GetF32(p+".rope.scaling.yarn_beta_fast", 32),
		RopeYarnBetaSlow:          gguf.GetF32(p+".rope.scaling.yarn_beta_slow", 1),
		RopeYarnLogMultiplier:     gguf.GetF32(p+".rope.scaling.yarn_log_multiplier", 1),
	}
}

// Weight is one loaded tensor in exactly one of two states: F32 non-nil
// (plain floats, converted at load time from F32/F16 storage) or Raw non-nil
// (still-quantized bytes, usually borrowed zero-copy from the mmap'd file,
// dequantized on the fly inside the matvec kernels). Rows/Cols only apply to
// the quantized form; the F32 form infers rows from len(F32)/cols at the
// call site.
type Weight struct {
	F32  []float32
	Raw  []byte
	Type GGMLType
	Rows int
	Cols int
}

// Matvec computes out = W·x, allocating the result. MatvecInto is the
// allocation-free form used on the decode hot path; it dispatches to the
// quant-type-specific parallel kernel in simd.go.
func (w Weight) Matvec(x []float32) []float32 {
	out := make([]float32, max(0, w.Rows))
	w.MatvecInto(x, &out)
	return out
}

func (w Weight) MatvecInto(x []float32, out *[]float32) {
	if w.F32 != nil {
		cols := len(x)
		rows := 0
		if cols > 0 {
			rows = len(w.F32) / cols
		}
		MatvecF32Into(w.F32, x, rows, cols, out)
		return
	}
	switch w.Type {
	case GGMLTypeQ8_0:
		MatvecQ8_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_0:
		MatvecQ4_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_K:
		MatvecQ4KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ5_K:
		MatvecQ5KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ6_K:
		MatvecQ6KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeMXFP4:
		MatvecMXFP4Into(w.Raw, x, w.Rows, w.Cols, out)
	default:
		panic(fmt.Sprintf("unsupported quantized matvec: %v", w.Type))
	}
}

// Row dequantizes a single weight row (used for token-embedding lookups).
// RowInto is the allocation-free form.
func (w Weight) Row(row, cols int) []float32 {
	out := make([]float32, cols)
	w.RowInto(row, cols, &out)
	return out
}

func (w Weight) RowInto(row, cols int, out *[]float32) {
	ensureLenNoClear(out, cols)
	if w.F32 != nil {
		start := row * cols
		copy(*out, w.F32[start:min(start+cols, len(w.F32))])
		return
	}
	switch w.Type {
	case GGMLTypeQ8_0:
		rowBytes := (cols / 32) * 34
		copy(*out, DequantRowQ8_0(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols))
	case GGMLTypeQ4_0:
		rowBytes := (cols / 32) * 18
		copy(*out, DequantRowQ4_0(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols))
	case GGMLTypeQ4_K:
		rowBytes := (cols / 256) * 144
		copy(*out, DequantRowQ4K(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols))
	case GGMLTypeQ5_K:
		rowBytes := (cols / 256) * 176
		copy(*out, DequantRowQ5K(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols))
	case GGMLTypeQ6_K:
		rowBytes := (cols / 256) * 210
		copy(*out, DequantRowQ6K(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols))
	case GGMLTypeMXFP4:
		rowBytes := (cols / 32) * 17
		copy(*out, DequantRowMXFP4(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols))
	default:
		panic(fmt.Sprintf("unsupported quantized row extraction: %v", w.Type))
	}
}

func (w Weight) RowF32(row, cols int) []float32 {
	if w.F32 == nil {
		panic("expected f32 row storage")
	}
	start := row * cols
	return w.F32[start : start+cols]
}

// LayerWeights holds one transformer block. Attention is either split
// (WQ/WK/WV, with optional biases BQ/BK/BV) or fused into a single WQKV
// (HasQKV); the SwiGLU FFN is likewise either split (W1 = gate, W3 = up,
// W2 = down — llama.cpp naming) or fused gate+up in WGateUp (HasGateUp).
type LayerWeights struct {
	AttnNorm  []float32
	WQ        Weight
	BQ        []float32
	WK        Weight
	BK        []float32
	WV        Weight
	BV        []float32
	WQKV      Weight
	HasQKV    bool
	WO        Weight
	FFNNorm   []float32
	W1        Weight
	W2        Weight
	W3        Weight
	WGateUp   Weight
	HasGateUp bool
}

type ModelWeights struct {
	TokenEmbd  Weight
	OutputNorm []float32
	Output     Weight
	Layers     []LayerWeights
}

type Gemma4LayerWeights struct {
	AttnNorm   []float32
	AttnQ      Weight
	AttnK      Weight
	AttnV      Weight
	AttnOutput Weight
	FFNNorm    []float32
	FFNDown    Weight
	FFNUp      Weight
	FFNGate    Weight
	HeadDim    int
	NKVHeads   int
	ValueDim   int
	HasAttnV   bool
}

type Gemma4Weights struct {
	TokenEmbd  Weight
	OutputNorm []float32
	Output     Weight
	Layers     []Gemma4LayerWeights
	Standard   ModelWeights
}

type GptOssWeights struct {
	Standard ModelWeights
}

// KVCache stores the attention keys and values of every processed position,
// one flat slice per layer laid out position-major: position p's keys occupy
// K[layer][p*PerPosKDim : (p+1)*PerPosKDim] (all KV heads concatenated), and
// likewise for V. Sized once per generation to prompt+max_tokens (capped at
// the model context length); there is no ring/eviction — generation stops at
// MaxLen.
type KVCache struct {
	K          [][]float32
	V          [][]float32
	PerPosKDim int
	PerPosVDim int
	MaxLen     int
}

// NewKVCache allocates a cache for `layers` layers of maxLen positions with
// the given per-position K and V widths (see KVCache).
func NewKVCache(layers, kDim, vDim, maxLen int) *KVCache {
	k := make([][]float32, layers)
	v := make([][]float32, layers)
	for i := range layers {
		k[i] = make([]float32, maxLen*kDim)
		v[i] = make([]float32, maxLen*vDim)
	}
	return &KVCache{K: k, V: v, PerPosKDim: kDim, PerPosVDim: vDim, MaxLen: maxLen}
}

// DecodeBuffer is the per-generation scratch memory for the single-token
// forward pass: activation vectors (X residual stream, XN/XN2 normed views,
// Q/K/V/AttnOut/Proj attention buffers, Gate/Up/Hidden FFN buffers), the
// sampler's candidate scratch, and precomputed RoPE tables (per-pair inverse
// frequencies plus per-position sin/cos filled in prepareRopeScratch). One
// DecodeBuffer serves a whole generation, so the decode loop allocates
// nothing per token. Not safe for concurrent use — Runner.genLock serializes
// generations.
type DecodeBuffer struct {
	X                       []float32
	XN                      []float32
	XN2                     []float32
	Q                       []float32
	K                       []float32
	V                       []float32
	QKV                     []float32
	AttnOut                 []float32
	Proj                    []float32
	Gate                    []float32
	Up                      []float32
	GateUp                  []float32
	Hidden                  []float32
	MOE                     []float32
	RouterLogits            []float32
	TopExperts              []ExpertScore
	ExpertProbs             []float32
	SamplerCandidates       []TokenProb
	Q4KXSums                []float32
	RopeInvFreq             []float32
	RopeSin                 []float32
	RopeCos                 []float32
	RopeMscale              float32
	RopeGptOssInvFreq       []float32
	RopeGptOssConcentration float32
}

type ExpertScore struct {
	Index int
	Score float32
}

func NewDecodeBuffer(config Config, maxHeadDim, maxNKVHeads, maxValueDim int) *DecodeBuffer {
	inv, mscale := buildRopeInvFreq(config, maxHeadDim)
	gptInv, concentration := buildRopeInvFreqGptOss(config)
	return &DecodeBuffer{
		X:                       make([]float32, config.Dim),
		XN:                      make([]float32, config.Dim),
		XN2:                     make([]float32, config.Dim),
		Q:                       make([]float32, config.NHeads*maxHeadDim),
		K:                       make([]float32, maxNKVHeads*maxHeadDim),
		V:                       make([]float32, maxNKVHeads*maxValueDim),
		QKV:                     make([]float32, config.NHeads*maxHeadDim+maxNKVHeads*maxHeadDim+maxNKVHeads*maxValueDim),
		AttnOut:                 make([]float32, config.NHeads*maxValueDim),
		Proj:                    make([]float32, config.Dim),
		Gate:                    make([]float32, config.HiddenDim),
		Up:                      make([]float32, config.HiddenDim),
		GateUp:                  make([]float32, config.HiddenDim*2),
		Hidden:                  make([]float32, config.HiddenDim),
		MOE:                     make([]float32, config.Dim),
		RouterLogits:            make([]float32, config.ExpertCount),
		SamplerCandidates:       make([]TokenProb, 0, 64),
		Q4KXSums:                make([]float32, max(1, config.Dim/32)),
		RopeInvFreq:             inv,
		RopeSin:                 make([]float32, max(1, maxHeadDim/2)),
		RopeCos:                 make([]float32, max(1, maxHeadDim/2)),
		RopeMscale:              mscale,
		RopeGptOssInvFreq:       gptInv,
		RopeGptOssConcentration: concentration,
	}
}

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

// LoadModel loads the standard llama-style weight set from a parsed GGUF.
// With borrowQuantized set (the mmap path), quantized tensors are zero-copy
// sub-slices of data — the caller must keep data alive for the model's
// lifetime; without it they are copied into owned memory (the in-memory test
// path). Models without a separate output.weight tie the output projection to
// the token embeddings.
func LoadModel(data []byte, gguf *GGUFFile, borrowQuantized bool) (Config, ModelWeights, error) {
	config := ConfigFromGGUF(gguf)
	fmt.Fprintf(stderr(), "Config: dim=%d, layers=%d, heads=%d/%d, hidden=%d, vocab=%d, ctx=%d\n",
		config.Dim, config.NLayers, config.NHeads, config.NKVHeads, config.HiddenDim, config.VocabSize, config.MaxSeqLen)
	tensorIdx := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	inferAttentionShape(&config, tensorIdx)
	if info, ok := tensorIdx["rope_factors_long.weight"]; ok {
		config.RopeFactorsLong = loadOptionalF32Vec(data, gguf.DataOffset, "rope_factors_long.weight", tensorIdx, inferred, info.Numel())
	}
	if info, ok := tensorIdx["rope_factors_short.weight"]; ok {
		config.RopeFactorsShort = loadOptionalF32Vec(data, gguf.DataOffset, "rope_factors_short.weight", tensorIdx, inferred, info.Numel())
	}

	tokenEmbd, err := loadWeight(data, gguf.DataOffset, "token_embd.weight", tensorIdx, inferred, false, borrowQuantized)
	if err != nil {
		return config, ModelWeights{}, err
	}
	outputNorm, err := loadF32Vec(data, gguf.DataOffset, "output_norm.weight", tensorIdx, inferred)
	if err != nil {
		return config, ModelWeights{}, err
	}
	output := tokenEmbd
	if _, ok := tensorIdx["output.weight"]; ok {
		output, err = loadWeight(data, gguf.DataOffset, "output.weight", tensorIdx, inferred, false, borrowQuantized)
		if err != nil {
			return config, ModelWeights{}, err
		}
	} else {
		fmt.Fprintln(stderr(), "Note: output tied to embeddings")
	}

	layers := make([]LayerWeights, 0, config.NLayers)
	qRows := config.NHeads * config.HeadDim
	kRows := config.NKVHeads * config.HeadDim
	vRows := config.NKVHeads * config.ValueDim
	for l := range config.NLayers {
		layer, err := loadLayer(data, gguf.DataOffset, l, tensorIdx, inferred, borrowQuantized, qRows, kRows, vRows)
		if err != nil {
			return config, ModelWeights{}, err
		}
		layers = append(layers, layer)
		if l == 0 || (l+1)%8 == 0 || l+1 == config.NLayers {
			fmt.Fprintf(stderr(), "  Loaded layer %d/%d\n", l+1, config.NLayers)
		}
	}
	return config, ModelWeights{TokenEmbd: tokenEmbd, OutputNorm: outputNorm, Output: output, Layers: layers}, nil
}

func LoadGptOssModel(data []byte, gguf *GGUFFile, borrowQuantized bool) (Config, GptOssWeights, error) {
	config, weights, err := LoadModel(data, gguf, borrowQuantized)
	return config, GptOssWeights{Standard: weights}, err
}

func LoadGemma4Model(data []byte, gguf *GGUFFile, borrowQuantized bool) (Config, Gemma4Weights, error) {
	config, std, err := LoadModel(data, gguf, borrowQuantized)
	if err != nil {
		return config, Gemma4Weights{}, err
	}
	layers := make([]Gemma4LayerWeights, len(std.Layers))
	for i, l := range std.Layers {
		layers[i] = Gemma4LayerWeights{
			AttnNorm: l.AttnNorm, AttnQ: l.WQ, AttnK: l.WK, AttnV: l.WV, AttnOutput: l.WO,
			FFNNorm: l.FFNNorm, FFNDown: l.W2, FFNUp: l.W3, FFNGate: l.W1,
			HeadDim: config.HeadDim, NKVHeads: config.NKVHeads, ValueDim: config.ValueDim, HasAttnV: true,
		}
	}
	return config, Gemma4Weights{TokenEmbd: std.TokenEmbd, OutputNorm: std.OutputNorm, Output: std.Output, Layers: layers, Standard: std}, nil
}

func loadLayer(data []byte, dataOffset, l int, tensors map[string]TensorInfo, inferred map[string]int, borrow bool, qRows, kRows, vRows int) (LayerWeights, error) {
	prefix := fmt.Sprintf("blk.%d.", l)
	attnNorm, err := loadF32Vec(data, dataOffset, prefix+"attn_norm.weight", tensors, inferred)
	if err != nil {
		return LayerWeights{}, err
	}
	var wq, wk, wv, wqkv Weight
	hasQKV := false
	if _, ok := tensors[prefix+"attn_qkv.weight"]; ok {
		wqkv, err = loadWeight(data, dataOffset, prefix+"attn_qkv.weight", tensors, inferred, false, borrow)
		if err != nil {
			return LayerWeights{}, err
		}
		hasQKV = true
	} else {
		wq, err = loadWeight(data, dataOffset, prefix+"attn_q.weight", tensors, inferred, false, borrow)
		if err != nil {
			return LayerWeights{}, err
		}
		wk, err = loadWeight(data, dataOffset, prefix+"attn_k.weight", tensors, inferred, false, borrow)
		if err != nil {
			return LayerWeights{}, err
		}
		wv, err = loadWeight(data, dataOffset, prefix+"attn_v.weight", tensors, inferred, false, borrow)
		if err != nil {
			return LayerWeights{}, err
		}
	}
	wo, err := loadWeight(data, dataOffset, prefix+"attn_output.weight", tensors, inferred, false, borrow)
	if err != nil {
		return LayerWeights{}, err
	}
	ffnNorm, err := loadF32Vec(data, dataOffset, prefix+"ffn_norm.weight", tensors, inferred)
	if err != nil {
		return LayerWeights{}, err
	}
	w2, err := loadWeight(data, dataOffset, prefix+"ffn_down.weight", tensors, inferred, false, borrow)
	if err != nil {
		return LayerWeights{}, err
	}
	var w1, w3, wGateUp Weight
	hasGateUp := false
	if _, ok := tensors[prefix+"ffn_gate.weight"]; ok {
		w1, err = loadWeight(data, dataOffset, prefix+"ffn_gate.weight", tensors, inferred, false, borrow)
		if err != nil {
			return LayerWeights{}, err
		}
		w3, err = loadWeight(data, dataOffset, prefix+"ffn_up.weight", tensors, inferred, false, borrow)
		if err != nil {
			return LayerWeights{}, err
		}
	} else {
		wGateUp, err = loadWeight(data, dataOffset, prefix+"ffn_up.weight", tensors, inferred, false, borrow)
		if err != nil {
			return LayerWeights{}, err
		}
		hasGateUp = true
	}
	return LayerWeights{
		AttnNorm:  attnNorm,
		WQ:        wq,
		BQ:        loadOptionalF32Vec(data, dataOffset, prefix+"attn_q.bias", tensors, inferred, qRows),
		WK:        wk,
		BK:        loadOptionalF32Vec(data, dataOffset, prefix+"attn_k.bias", tensors, inferred, kRows),
		WV:        wv,
		BV:        loadOptionalF32Vec(data, dataOffset, prefix+"attn_v.bias", tensors, inferred, vRows),
		WQKV:      wqkv,
		HasQKV:    hasQKV,
		WO:        wo,
		FFNNorm:   ffnNorm,
		W1:        w1,
		W2:        w2,
		W3:        w3,
		WGateUp:   wGateUp,
		HasGateUp: hasGateUp,
	}, nil
}

func indexTensors(gguf *GGUFFile) map[string]TensorInfo {
	out := make(map[string]TensorInfo, len(gguf.Tensors))
	for _, t := range gguf.Tensors {
		out[t.Name] = t
	}
	return out
}

// inferTensorSizes derives each tensor's byte size from the gap to the next
// tensor's offset (last one runs to end of file). Used as the fallback when a
// tensor's dtype can't be sized analytically, and as a bounds cross-check in
// loadWeight.
func inferTensorSizes(data []byte, gguf *GGUFFile) map[string]int {
	type offIdx struct {
		off uint64
		idx int
	}
	offs := make([]offIdx, len(gguf.Tensors))
	for i, t := range gguf.Tensors {
		offs[i] = offIdx{t.Offset, i}
	}
	sort.Slice(offs, func(i, j int) bool { return offs[i].off < offs[j].off })
	out := make(map[string]int, len(offs))
	for i, oi := range offs {
		next := uint64(len(data) - gguf.DataOffset)
		if i+1 < len(offs) {
			next = offs[i+1].off
		}
		size := 0
		if next > oi.off {
			size = int(next - oi.off)
		}
		out[gguf.Tensors[oi.idx].Name] = size
	}
	return out
}

func inferAttentionShape(config *Config, tensors map[string]TensorInfo) {
	var headDimCand, valueDimCand int
	for l := range config.NLayers {
		if info, ok := tensors[fmt.Sprintf("blk.%d.attn_q.weight", l)]; ok && len(info.Dims) >= 2 {
			rows, cols := int(info.Dims[1]), int(info.Dims[0])
			if cols == config.Dim && config.NHeads > 0 && rows%config.NHeads == 0 {
				headDimCand = rows / config.NHeads
			}
		} else if info, ok := tensors[fmt.Sprintf("blk.%d.attn_qkv.weight", l)]; ok && len(info.Dims) >= 2 {
			rows, cols := int(info.Dims[1]), int(info.Dims[0])
			denom := config.NHeads + 2*config.NKVHeads
			if cols == config.Dim && denom > 0 && rows%denom == 0 {
				headDimCand = rows / denom
				valueDimCand = headDimCand
			}
		}
		if info, ok := tensors[fmt.Sprintf("blk.%d.attn_v.weight", l)]; ok && len(info.Dims) >= 2 {
			rows, cols := int(info.Dims[1]), int(info.Dims[0])
			if cols == config.Dim && config.NKVHeads > 0 && rows%config.NKVHeads == 0 {
				valueDimCand = rows / config.NKVHeads
			}
		}
		if headDimCand > 0 && valueDimCand > 0 {
			break
		}
	}
	if headDimCand > 0 {
		config.HeadDim = headDimCand
	}
	if valueDimCand > 0 {
		config.ValueDim = valueDimCand
	}
	config.KVDim = config.ValueDim * config.NKVHeads
	if config.NKVHeads > 0 {
		config.KVMul = max(1, config.NHeads/config.NKVHeads)
	}
}

// loadWeight materializes one named tensor as a Weight: F32/F16 storage is
// converted to owned float32s; supported quantized types stay in their packed
// byte form, borrowed from data when borrow is set or copied otherwise.
// forceF32 additionally dequantizes Q8_0/Q4_0 at load (used for norm vectors
// that must be plain floats).
func loadWeight(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, forceF32, borrow bool) (Weight, error) {
	info, ok := tensors[name]
	if !ok {
		return Weight{}, fmt.Errorf("missing tensor: %s", name)
	}
	numel := info.Numel()
	byteSize, ok := info.DType.DataSize(numel)
	if !ok {
		byteSize = inferred[name]
	}
	if inferredSize := inferred[name]; inferredSize > 0 {
		end := dataOffset + int(info.Offset) + byteSize
		if end > len(data) || byteSize == 0 {
			byteSize = inferredSize
		}
	}
	offset := dataOffset + int(info.Offset)
	if offset < 0 || offset > len(data) {
		return Weight{}, fmt.Errorf("tensor %s offset out of range", name)
	}
	rawEnd := min(offset+byteSize, len(data))
	raw := data[offset:rawEnd]
	if len(raw) < byteSize {
		if info.DType == GGMLTypeF32 || info.DType == GGMLTypeF16 {
			return Weight{}, fmt.Errorf("tensor %s exceeds file length", name)
		}
		padded := make([]byte, byteSize)
		copy(padded, raw)
		raw = padded
	}
	switch info.DType {
	case GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1, GGMLTypeQ8_1:
		return Weight{}, fmt.Errorf("tensor type %s is parsed but not implemented correctly yet for %s", info.DType, name)
	}
	effectiveForce := forceF32
	switch info.DType {
	case GGMLTypeF32:
		f := make([]float32, numel)
		for i := range numel {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return Weight{F32: f}, nil
	case GGMLTypeF16:
		f := make([]float32, numel)
		for i := range numel {
			f[i] = F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return Weight{F32: f}, nil
	case GGMLTypeQ8_0, GGMLTypeQ4_0, GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeMXFP4:
		if effectiveForce {
			if info.DType == GGMLTypeQ4_K || info.DType == GGMLTypeQ5_K || info.DType == GGMLTypeQ6_K || info.DType == GGMLTypeMXFP4 {
				return Weight{}, fmt.Errorf("%s force_f32 dequantization not implemented for %s", info.DType, name)
			}
			f := make([]float32, numel)
			if info.DType == GGMLTypeQ8_0 {
				copy(f, DequantRowQ8_0(raw, numel))
			} else {
				copy(f, DequantRowQ4_0(raw, numel))
			}
			return Weight{F32: f}, nil
		}
		rows, cols := 1, numel
		if len(info.Dims) >= 2 {
			rows = int(info.Dims[1])
			cols = int(info.Dims[0])
		}
		if !borrow {
			owned := make([]byte, len(raw))
			copy(owned, raw)
			raw = owned
		}
		return Weight{Raw: raw, Type: info.DType, Rows: rows, Cols: cols}, nil
	default:
		return Weight{}, fmt.Errorf("unsupported tensor type for %s: %s", name, info.DType)
	}
}

func loadF32Vec(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int) ([]float32, error) {
	w, err := loadWeight(data, dataOffset, name, tensors, inferred, true, false)
	if err != nil {
		return nil, err
	}
	if w.F32 == nil {
		return nil, fmt.Errorf("expected f32 for %s", name)
	}
	return w.F32, nil
}

func loadOptionalF32Vec(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, length int) []float32 {
	if _, ok := tensors[name]; !ok {
		return make([]float32, length)
	}
	v, err := loadF32Vec(data, dataOffset, name, tensors, inferred)
	if err != nil {
		return make([]float32, length)
	}
	return v
}

// Forward runs one token through the transformer and returns its
// next-token logits; ForwardInto is the allocation-free form. Both append the
// token's K/V to the cache at position pos as a side effect.
func Forward(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	logits := make([]float32, 0)
	ForwardInto(config, weights, cache, buf, token, pos, &logits)
	return logits
}

func ForwardInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardBodyInto(config, weights, cache, buf, token, pos)
	weights.Output.MatvecInto(buf.XN, logits)
	if config.LogitScale != 1 {
		ScaleF32(*logits, 1/config.LogitScale)
	}
}

// ForwardBodyInto is the transformer body shared by logits, prefill, and
// embedding paths: embed the token, then per layer run pre-norm attention
// (RoPE'd Q/K, K/V appended to the cache, online-softmax attention over all
// cached positions, output projection, residual add) followed by a pre-norm
// SwiGLU FFN, and finally apply the output norm, leaving the normed hidden
// state in buf.XN for the caller to project (or pool, for embeddings).
// Attention heads are spread across the worker pool once the attended span is
// long enough to amortize dispatch (see the comment at the call site); the
// Q/K/V and gate/up matvecs go through the fused multi-matrix kernels when
// the quant types allow (tryMatvec3Into/tryMatvec2Into).
func ForwardBodyInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	dim := config.Dim
	headDim := config.HeadDim
	kvMul := max(1, config.KVMul)
	ropeHalf, ropePairs := prepareRopeScratch(pos, headDim, config.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &buf.RopeSin, &buf.RopeCos)
	ropeIsInterleaved := ropeInterleaved(config.Arch)
	weights.TokenEmbd.RowInto(int(token), dim, &buf.X)
	if config.EmbeddingScale != 1 {
		ScaleF32(buf.X[:dim], config.EmbeddingScale)
	}
	for l := range config.NLayers {
		layer := weights.Layers[l]
		rmsNormInto(buf.X, layer.AttnNorm, config.RMSNormEps, &buf.XN)
		if layer.HasQKV {
			layer.WQKV.MatvecInto(buf.XN, &buf.QKV)
			qLen := config.NHeads * headDim
			kLen := config.NKVHeads * headDim
			vLen := config.NKVHeads * config.ValueDim
			ensureLenNoClear(&buf.Q, qLen)
			ensureLenNoClear(&buf.K, kLen)
			ensureLenNoClear(&buf.V, vLen)
			copy(buf.Q, buf.QKV[:qLen])
			copy(buf.K, buf.QKV[qLen:qLen+kLen])
			copy(buf.V, buf.QKV[qLen+kLen:qLen+kLen+vLen])
		} else {
			if !tryMatvec3Into(layer.WQ, layer.WK, layer.WV, buf.XN, &buf.Q4KXSums, &buf.Q, &buf.K, &buf.V) {
				layer.WQ.MatvecInto(buf.XN, &buf.Q)
				layer.WK.MatvecInto(buf.XN, &buf.K)
				layer.WV.MatvecInto(buf.XN, &buf.V)
			}
			addInPlace(buf.Q, layer.BQ)
			addInPlace(buf.K, layer.BK)
			addInPlace(buf.V, layer.BV)
		}
		applyPreparedRope(buf.Q, headDim, config.NHeads, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, ropeIsInterleaved)
		applyPreparedRope(buf.K, headDim, config.NKVHeads, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, ropeIsInterleaved)

		kStart := pos * cache.PerPosKDim
		vStart := pos * cache.PerPosVDim
		copy(cache.K[l][kStart:kStart+min(len(buf.K), cache.PerPosKDim)], buf.K)
		copy(cache.V[l][vStart:vStart+min(len(buf.V), cache.PerPosVDim)], buf.V)

		clear(buf.AttnOut)
		scale := config.AttentionScale
		if scale == 0 {
			scale = float32(1 / math.Sqrt(float64(headDim)))
		}
		attnStart := 0
		if config.SlidingWindow > 0 {
			attnStart = max(0, pos-config.SlidingWindow)
		}
		attnHeads := func(hStart, hEnd int) {
			for h := hStart; h < hEnd; h++ {
				kvH := h / kvMul
				qOff := h * headDim
				outOff := h * config.ValueDim
				onlineAttention(
					buf.Q[qOff:qOff+headDim],
					cache.K[l][kvH*config.HeadDim:],
					cache.V[l][kvH*config.ValueDim:],
					cache.PerPosKDim,
					cache.PerPosVDim,
					headDim,
					config.ValueDim,
					attnStart,
					pos,
					scale,
					buf.AttnOut[outOff:outOff+config.ValueDim],
				)
			}
		}
		// Attention cost grows with context length; spread heads across the
		// worker pool once there is enough work to amortize dispatch overhead.
		if attnLen := pos - attnStart + 1; attnLen >= 128 && config.NHeads > 1 {
			parallelChunks(config.NHeads, attnHeads)
		} else {
			attnHeads(0, config.NHeads)
		}
		layer.WO.MatvecInto(buf.AttnOut, &buf.Proj)
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		addInPlace(buf.X[:dim], buf.Proj)

		rmsNormInto(buf.X, layer.FFNNorm, config.RMSNormEps, &buf.XN2)
		if layer.HasGateUp {
			layer.WGateUp.MatvecInto(buf.XN2, &buf.GateUp)
			ensureLenNoClear(&buf.Gate, config.HiddenDim)
			ensureLenNoClear(&buf.Up, config.HiddenDim)
			copy(buf.Gate, buf.GateUp[:config.HiddenDim])
			copy(buf.Up, buf.GateUp[config.HiddenDim:2*config.HiddenDim])
		} else {
			if !tryMatvec2Into(layer.W1, layer.W3, buf.XN2, &buf.Q4KXSums, &buf.Gate, &buf.Up) {
				layer.W1.MatvecInto(buf.XN2, &buf.Gate)
				layer.W3.MatvecInto(buf.XN2, &buf.Up)
			}
		}
		hDim := config.HiddenDim
		ensureLenNoClear(&buf.Hidden, hDim)
		if hDim > 0 {
			gate := buf.Gate
			up := buf.Up
			hidden := buf.Hidden
			_ = gate[hDim-1]
			_ = up[hDim-1]
			_ = hidden[hDim-1]
			for i := 0; i < hDim; i++ {
				g := gate[i]
				hidden[i] = (g / (1 + float32(math.Exp(float64(-g))))) * up[i]
			}
		}
		layer.W2.MatvecInto(buf.Hidden, &buf.Proj)
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		addInPlace(buf.X[:dim], buf.Proj)
	}
	rmsNormInto(buf.X, weights.OutputNorm, config.RMSNormEps, &buf.XN)
}

func ForwardHidden(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	ForwardBodyInto(config, weights, cache, buf, token, pos)
	out := make([]float32, len(buf.XN))
	copy(out, buf.XN)
	return out
}

func ForwardPrefill(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	ForwardBodyInto(config, weights, cache, buf, token, pos)
}

func ForwardGptOssInto(config Config, weights GptOssWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardInto(config, weights.Standard, cache, buf, token, pos, logits)
}

func ForwardHiddenGptOss(config Config, weights GptOssWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	return ForwardHidden(config, weights.Standard, cache, buf, token, pos)
}

func ForwardGemma4Into(config Config, weights Gemma4Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardInto(config, weights.Standard, cache, buf, token, pos, logits)
}

func ForwardHiddenGemma4(config Config, weights Gemma4Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	ForwardBodyInto(config, weights.Standard, cache, buf, token, pos)
	out := make([]float32, len(buf.XN))
	copy(out, buf.XN)
	return out
}

// rmsNormInto writes out[i] = x[i] / rms(x) * weight[i] where
// rms(x) = sqrt(mean(x²) + eps) — RMSNorm as used by all supported
// architectures (no mean subtraction, no bias).
func rmsNormInto(x, weight []float32, eps float32, out *[]float32) {
	n := len(x)
	ensureLenNoClear(out, n)
	if n == 0 {
		return
	}
	ss := DotF32(x, x)
	scale := float32(1 / math.Sqrt(float64(ss/float32(n)+eps)))

	o := *out
	_ = o[n-1]
	_ = x[n-1]

	if len(weight) >= n {
		mulScaleF32(x[:n], weight[:n], scale, o[:n])
	} else {
		for i := 0; i < n; i++ {
			w := float32(1)
			if i < len(weight) {
				w = weight[i]
			}
			o[i] = x[i] * scale * w
		}
	}
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
	case "llama", "llama2", "llama3", "mistral", "mistral3", "mixtral", "ministral":
		return true
	default:
		return false
	}
}

// tryMatvec3Into / tryMatvec2Into route same-typed Q4_K or Q6_K weight groups
// (Q/K/V projections; FFN gate+up) through the fused kernels that share one
// activation-sums pass and one worker-pool dispatch. They return false —
// having written nothing — whenever types or shapes don't line up, and the
// caller falls back to independent matvecs.
func tryMatvec3Into(wq, wk, wv Weight, x []float32, q4kXSums *[]float32, q, k, v *[]float32) bool {
	if wq.Type != wk.Type || wq.Type != wv.Type || wq.Cols != wk.Cols || wq.Cols != wv.Cols || wq.Cols != len(x) || wq.F32 != nil || wk.F32 != nil || wv.F32 != nil {
		return false
	}
	switch wq.Type {
	case GGMLTypeQ4_K:
		return Q4KMatvec3IntoWithXSums(
			Q4KMatrix{Data: wq.Raw, Rows: wq.Rows, Cols: wq.Cols},
			Q4KMatrix{Data: wk.Raw, Rows: wk.Rows, Cols: wk.Cols},
			Q4KMatrix{Data: wv.Raw, Rows: wv.Rows, Cols: wv.Cols},
			x,
			q4kXSums,
			q,
			k,
			v,
		)
	case GGMLTypeQ6_K:
		return MatvecQ6K3Into(wq.Raw, wq.Rows, wq.Cols, wk.Raw, wk.Rows, wk.Cols, wv.Raw, wv.Rows, wv.Cols, x, q, k, v)
	default:
		return false
	}
}

func tryMatvec2Into(a, b Weight, x []float32, q4kXSums *[]float32, aOut, bOut *[]float32) bool {
	if a.Type != b.Type || a.Cols != b.Cols || a.Cols != len(x) || a.F32 != nil || b.F32 != nil {
		return false
	}
	switch a.Type {
	case GGMLTypeQ4_K:
		return MatvecQ4K2IntoWithXSums(a.Raw, a.Rows, a.Cols, b.Raw, b.Rows, b.Cols, x, q4kXSums, aOut, bOut)
	case GGMLTypeQ6_K:
		return MatvecQ6K2Into(a.Raw, a.Rows, a.Cols, b.Raw, b.Rows, b.Cols, x, aOut, bOut)
	default:
		return false
	}
}

// onlineAttention computes softmax(q·K^T * scale)·V for one head over cached
// positions [startT, endT] using the online-softmax algorithm: a single pass
// that tracks the running max score and rescales the accumulated output
// whenever a new maximum appears (out *= exp(oldMax-newMax)), so no
// per-position score array is materialized regardless of context length.
// keys/values are the head's slices into the position-major KV cache, striding
// by keyStride/valueStride per position. out must be zeroed by the caller and
// receives the already-normalized weighted value sum.
func onlineAttention(query, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale float32, out []float32) {
	maxScore := float32(math.Inf(-1))
	denom := float32(0)
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		vOff := t * valueStride
		if kOff+keyHeadDim > len(keys) || vOff+valueHeadDim > len(values) {
			break
		}
		score := DotF32(query, keys[kOff:kOff+keyHeadDim]) * scale
		valueRow := values[vOff : vOff+valueHeadDim]
		if score > maxScore {
			oldScale := float32(0)
			if !math.IsInf(float64(maxScore), 0) {
				oldScale = float32(math.Exp(float64(maxScore - score)))
			}
			ScaleAddF32(out[:valueHeadDim], oldScale, valueRow)
			denom = denom*oldScale + 1
			maxScore = score
		} else {
			weight := float32(math.Exp(float64(score - maxScore)))
			AxpyF32(out[:valueHeadDim], weight, valueRow)
			denom += weight
		}
	}
	if denom > 0 {
		ScaleF32(out[:valueHeadDim], 1/denom)
	}
}

func addInPlace(dst, src []float32) {
	AxpyF32(dst, 1.0, src)
}

func clamp(v, lo, hi float32) float32 {
	return min(max(v, lo), hi)
}
