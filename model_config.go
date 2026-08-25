package gopherllm

import (
	"math"
	"strings"
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
	Arch      string
	Dim       int
	HiddenDim int
	NLayers   int
	NHeads    int
	NKVHeads  int
	VocabSize int
	MaxSeqLen int
	RopeTheta float32
	// RopeThetaSWA is an optional second frequency base for local-attention
	// layers. OLMo 3 uses it with unscaled RoPE while its global layers retain
	// the model's ordinary (potentially YaRN-scaled) RoPE table.
	RopeThetaSWA   float32
	RMSNormEps     float32
	AttentionScale float32
	// AttentionTemperatureScale is Mistral 3's long-context Q scaling
	// coefficient. It remains separate from AttentionScale, which is the
	// ordinary static QK score multiplier.
	AttentionTemperatureScale float32
	AttentionTemperatureFloor int
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
	// StableLM uses mean-and-variance LayerNorm and can calculate attention
	// and FFN from the same input before adding both residuals.
	UseLayerNorm     bool
	ParallelResidual bool
	// Gemma-family mechanics (all inert at their zero values; see
	// docs/INFERENCE_NOTES.md for the researched semantics):
	// UseGELU switches the FFN activation from SiLU to GELU. Gemma uses the
	// tanh approximation in its gated MLP; Phi-2 uses exact GELU in a
	// sequential, ungated MLP selected by its architecture.
	// AttnLogitSoftcap/FinalLogitSoftcap apply cap*tanh(v/cap) to attention
	// scores / final logits (Gemma 2: 50.0 / 30.0). SWAPattern, when non-nil,
	// restricts the sliding window to layers whose entry is true (Gemma 4
	// ships it as bool-array metadata; Gemma 2's alternating pattern is
	// synthesized); nil means SlidingWindow applies to every layer.
	UseGELU bool
	// UseExactGELU selects erf-based exact GELU over the tanh approximation
	// for architectures whose FFN is plain (see usesPlainMLP): Phi-2 uses
	// exact GELU; GPT-2/GPT-NeoX/GPT-J/BLOOM/MPT/StarCoder/StarCoder2 use
	// the tanh approximation ("gelu_new"/"gelu_pytorch_tanh") instead.
	UseExactGELU bool
	// ALiBiMaxBias configures BLOOM/MPT's additive per-head linear
	// attention-score bias (see Config.usesALiBi); zero means disabled.
	ALiBiMaxBias      float32
	AttnLogitSoftcap  float32
	FinalLogitSoftcap float32
	SWAPattern        []bool
	// Mamba-2 configuration used by the pure Mamba2 and hybrid Nemotron-H
	// paths. These fields are unused by the standard transformer path. A
	// zero-valued per-layer entry means that the layer does not expose that
	// component.
	LayerHeads   []int
	LayerKVHeads []int
	LayerFFNDim  []int
	SSMConv      int
	SSMInner     int
	SSMState     int
	SSMHeads     int
	SSMGroups    int
	// QwenRecurrentLayers, when supplied by a Qwen3.5/3.6/3.8 GGUF, is the
	// authoritative per-layer hybrid schedule: true denotes a Gated DeltaNet
	// layer and false denotes ordinary full attention.  Older exports expose
	// only FullAttentionInterval, for which the loader retains the documented
	// every-Nth-layer fallback.
	QwenRecurrentLayers []bool
	// NextNPredictLayers is the number of MTP/NextN draft blocks appended to
	// the Qwen3.5/3.6/3.8 decoder stack. They are not part of normal autoregressive
	// inference unless speculative decoding is enabled.
	NextNPredictLayers int
	// FullAttentionInterval is Qwen3.5/3.6/3.8's hybrid schedule: layer il (0
	// indexed) keeps ordinary self-attention when (il+1)%FullAttentionInterval
	// == 0, and uses the Gated DeltaNet linear-recurrent mixer otherwise. Zero
	// means the architecture does not use this schedule.
	FullAttentionInterval int
	ExpertWeightsNorm     bool
	ExpertWeightsScale    float32
	ExpertWeightsNormClip float32
	// DeepSeek-V2/V3/Kimi-K2 Multi-head Latent Attention (MLA).  HeadDim and
	// ValueDim retain the compact cache widths from the GGUF header; the
	// decompressed per-head Q/K and V widths live in MLAKeyDim/MLAValueDim.
	// Keeping both representations explicit prevents an MLA model from being
	// accidentally interpreted as ordinary GQA.
	UsesMLA                bool
	MLAQueryLoRARank       int
	MLAKVLoRARank          int
	MLAKeyDim              int
	MLAValueDim            int
	LeadingDenseBlockCount int
	ExpertFeedForwardDim   int
	ExpertSharedCount      int
	ExpertGroupCount       int
	ExpertGroupUsedCount   int
	ExpertGatingFunc       int
}

// gemmaFamily reports whether arch is a Gemma generation, all of which share
// the hardcoded sqrt(dim) embedding scaling and GELU FFN.
func gemmaFamily(arch string) bool {
	switch arch {
	case "gemma", "gemma2", "gemma3", "gemma4":
		return true
	default:
		return false
	}
}

// defaultExpertWeightsNorm follows the reference graphs for the sparse-MoE
// families handled by the standard decoder. Mixtral/Llama and Qwen3 normalize
// the selected weights; Qwen2-MoE intentionally retains their mass from the
// full router softmax. Nemotron-H has a separate graph and keeps its historic
// metadata-driven default.
func defaultExpertWeightsNorm(arch string) bool {
	switch arch {
	case "qwen2moe", "nemotron_h", "nemotron_h_moe", "mamba2":
		return false
	default:
		return true
	}
}

// deepSeek2Family covers the GGUF architecture emitted for DeepSeek-V2/V3
// and Kimi K2.  Official Kimi GGUF conversions declare `deepseek2`; the
// kimi_k2 alias is accepted for converters which retain the HF model type.
func deepSeek2Family(arch string) bool {
	return arch == "deepseek2" || arch == "kimi_k2"
}

// usesUngatedSharedExpert reports architectures whose sparse-MoE layers have
// an always-on shared expert added directly to the routed output, with no
// sigmoid/softmax gate of its own — as opposed to Qwen2-MoE's shared expert,
// which is gated by its own router logit (ffn_gate_inp_shexp). GraniteMoE
// shares this mechanism with the DeepSeek2 family despite being otherwise
// unrelated to it, so this is deliberately its own helper rather than folded
// into deepSeek2Family.
func usesUngatedSharedExpert(arch string) bool {
	return deepSeek2Family(arch) || arch == "granitemoe"
}

// layerUsesSWA reports whether layer il attends with the sliding window (true)
// or over the full context (false, or when no window is configured).
func (c Config) layerUsesSWA(il int) bool {
	if c.SlidingWindow <= 0 {
		return false
	}
	if c.SWAPattern == nil {
		return true
	}
	if il < 0 || il >= len(c.SWAPattern) {
		return false
	}
	return c.SWAPattern[il]
}

// layerUsesRoPE reports whether layer il rotates Q/K. Most decoder families
// apply RoPE in every block. SmolLM3 deliberately leaves every fourth block
// unrotated; EXAONE 4's 32B graph rotates only its local-attention blocks
// (the dense 1.2B model has no sliding window and rotates every block).
func (c Config) layerUsesRoPE(il int) bool {
	switch c.Arch {
	case "smollm3":
		return (il+1)%4 != 0
	case "exaone4":
		return c.SlidingWindow <= 0 || c.layerUsesSWA(il)
	case "gpt2", "bloom", "mpt", "starcoder":
		// These position tokens via a learned absolute embedding (GPT-2/
		// StarCoder) or ALiBi (BLOOM/MPT) instead of RoPE.
		return false
	default:
		return true
	}
}

// usesPostNormOnly identifies decoder blocks whose residual branches consume
// the unnormalized hidden state and normalize each projected branch before it
// is added back. EXAONE 4 is the first supported family with this layout.
func (c Config) usesPostNormOnly() bool {
	return c.Arch == "exaone4" || c.Arch == "olmo2"
}

// sharesParallelBranchNorm identifies architectures whose attention and FFN
// branches consume the same normalized block input. Parallel StableLM variants
// use a learned LayerNorm for both branches; Phi-2 and GPT-J do the same with
// their ungated MLP. Falcon's parallel-residual variants deliberately are NOT
// included here even though 7B checkpoints also share one norm: Falcon's
// tensor loader aliases AttnNorm/FFNNorm to the same weights when there is
// only one, so the ordinary independent-norm path already produces the
// identical (if slightly redundant) result for both Falcon sizes.
func (c Config) sharesParallelBranchNorm() bool {
	return c.ParallelResidual && (c.Arch == "phi2" || c.Arch == "stablelm" || c.Arch == "gptj" || c.Arch == "command-r")
}

// usesLayerNorm reports whether this architecture normalizes with
// mean/variance LayerNorm instead of RMSNorm. This single list is the source
// of truth for both Config.UseLayerNorm and which architectures read their
// epsilon from the "*.attention.layer_norm_epsilon" key instead of the
// generic RMS one (see the switch in ConfigFromGGUF) — the two questions
// have the same answer for every architecture GopherLLM supports, so a
// single helper keeps them from silently drifting apart.
func usesLayerNorm(arch string) bool {
	switch arch {
	case "stablelm", "phi2", "gpt2", "gptneox", "gptj", "bloom", "mpt", "falcon", "starcoder", "starcoder2", "command-r":
		return true
	default:
		return false
	}
}

// forcesParallelResidual reports architectures whose parallel-residual block
// structure (attention and FFN both computed from the same input, summed
// into one residual add) is an architectural constant rather than a
// per-checkpoint GGUF flag. Contrast with the generic
// "{arch}.use_parallel_residual" key read earlier in ConfigFromGGUF, which
// covers GPT-NeoX/StableLM-style architectures where it varies by file.
func forcesParallelResidual(arch string) bool {
	switch arch {
	case "phi2", "gptj", "falcon", "command-r":
		return true
	default:
		return false
	}
}

// usesAbsolutePositionEmbd reports whether this architecture positions
// tokens with a learned absolute position-embedding table (gathered by
// sequence position and added to the token embedding once, before layer 0)
// instead of RoPE or ALiBi. Only GPT-2 and StarCoder v1 do this among
// supported architectures.
func (c Config) usesAbsolutePositionEmbd() bool {
	return c.Arch == "gpt2" || c.Arch == "starcoder"
}

// usesALiBi reports whether this architecture positions tokens with an
// additive per-head linear attention-score bias (Attention with Linear
// Biases) instead of RoPE or a learned position embedding. BLOOM hardcodes
// its max-bias to 8.0; MPT reads it from a GGUF key (0 means disabled, but
// every released checkpoint enables it).
func (c Config) usesALiBi() bool {
	return c.Arch == "bloom" || (c.Arch == "mpt" && c.ALiBiMaxBias > 0)
}

// aLiBiSlope returns head h's linear-bias slope for BLOOM/MPT's ALiBi
// positional mechanism. Heads are split at n2, the largest power of two not
// exceeding nHeads: heads below n2 follow one geometric sequence (m0), and
// any remaining heads (only when nHeads is not itself a power of two) follow
// a second, twice-as-fine sequence (m1) starting from every other exponent.
// For power-of-two head counts this collapses to the textbook
// slope_h = 2^(-maxBias*(h+1)/nHeads).
func aLiBiSlope(h, nHeads int, maxBias float32) float32 {
	n2 := 1
	for n2*2 <= nHeads {
		n2 *= 2
	}
	if h < n2 {
		m0 := math.Pow(2, -float64(maxBias)/float64(n2))
		return float32(math.Pow(m0, float64(h+1)))
	}
	m1 := math.Pow(2, -float64(maxBias)/float64(2*n2))
	return float32(math.Pow(m1, float64(2*(h-n2)+1)))
}

// usesPlainMLP reports whether this architecture's FFN is a plain
// (non-gated) up-projection -> activation -> down-projection MLP using only
// W3 (up) and W2 (down), with no W1/gate tensor and no elementwise gate
// multiply. UseExactGELU distinguishes Phi-2's exact GELU from the rest of
// this family's tanh-approximation GELU.
func (c Config) usesPlainMLP() bool {
	switch c.Arch {
	case "phi2", "gpt2", "gptneox", "gptj", "bloom", "mpt", "starcoder", "starcoder2":
		return true
	default:
		return false
	}
}

// usesFullProjectionQKNorm distinguishes OLMo 2/3's one RMSNorm over the
// complete Q or K projection from the shared per-head norm used by Qwen3 and
// EXAONE 4.
func (c Config) usesFullProjectionQKNorm() bool {
	return c.Arch == "olmo2"
}

func ConfigFromGGUF(gguf *GGUFFile) Config {
	// arch is the canonical architecture label — it drives loader, graph, and
	// template selection. p is the metadata namespace the hyperparameters
	// actually live under. They differ for alias labels that keep their
	// identity (kimi_k2 with deepseek2.* hparams, llama2/llama3 with llama.*)
	// and for files whose declared label was missing or unknown and had to be
	// detected — see ResolveArchitecture. Behavior decisions below must key on
	// arch; metadata reads must go through p.
	arch, p := ResolveArchitecture(gguf)
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
	embeddingScale := gguf.GetF32(p+".embedding_scale", 0)
	if embeddingScale == 0 {
		if gemmaFamily(arch) && dim > 0 {
			// Gemma scales input embeddings by sqrt(dim); reference
			// implementations hardcode this — it is NOT in GGUF metadata
			// (verified against llama.cpp's gemma graphs).
			embeddingScale = float32(math.Sqrt(float64(dim)))
		} else {
			embeddingScale = 1
		}
	}
	residualScale := gguf.GetF32(p+".residual_scale", 1)
	if residualScale == 0 {
		residualScale = 1
	}
	logitScale := gguf.GetF32(p+".logit_scale", 1)
	if logitScale == 0 {
		logitScale = 1
	}
	parallelResidual := false
	if v, ok := gguf.Metadata[p+".use_parallel_residual"]; ok {
		parallelResidual, _ = v.AsBool()
	}
	if arch == "stablelm" {
		// StableLM converters commonly retain use_parallel_residual=true even
		// for Stable-Code checkpoints that contain a separate FFN LayerNorm.
		// The tensor layout is authoritative: llama.cpp also selects the
		// sequential branch whenever ffn_norm is present.
		parallelResidual = true
		for _, tensor := range gguf.Tensors {
			if strings.HasSuffix(tensor.Name, ".ffn_norm.weight") {
				parallelResidual = false
				break
			}
		}
	}
	normEps := gguf.GetF32(p+".attention.layer_norm_rms_epsilon", 1e-5)
	if usesLayerNorm(arch) {
		// LayerNorm architectures carry an explicit epsilon key of their own
		// rather than the generic RMSNorm one.
		normEps = gguf.GetF32(p+".attention.layer_norm_epsilon", normEps)
	}
	hiddenDim := int(gguf.GetU32(p+".feed_forward_length", 0))
	// Gemma 4 E2B serializes feed_forward_length as one entry per layer. The
	// native loader still validates tensor shapes, but retaining its maximum in
	// Config makes header-only analysis and early workspace sizing truthful.
	if perLayerFFN, ok := gguf.GetU32Array(p + ".feed_forward_length"); ok {
		for _, width := range perLayerFFN {
			hiddenDim = max(hiddenDim, int(width))
		}
	}
	cfg := Config{
		Arch:                      arch,
		Dim:                       dim,
		HiddenDim:                 hiddenDim,
		NLayers:                   int(gguf.GetU32(p+".block_count", 0)),
		NHeads:                    nHeads,
		NKVHeads:                  nKVHeads,
		VocabSize:                 vocab,
		MaxSeqLen:                 int(gguf.GetU32(p+".context_length", 2048)),
		RopeTheta:                 gguf.GetF32(p+".rope.freq_base", 10000),
		RMSNormEps:                normEps,
		AttentionScale:            gguf.GetF32(p+".attention.scale", 0),
		AttentionTemperatureScale: gguf.GetF32(p+".attention.temperature_scale", 0),
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
		ExpertWeightsNorm:         defaultExpertWeightsNorm(arch),
		ExpertWeightsScale:        gguf.GetF32(p+".expert_weights_scale", 1),
		ExpertWeightsNormClip:     gguf.GetF32(p+".expert_weights_norm_clip", 0),
		MLAQueryLoRARank:          int(gguf.GetU32(p+".attention.q_lora_rank", 0)),
		MLAKVLoRARank:             int(gguf.GetU32(p+".attention.kv_lora_rank", 0)),
		MLAKeyDim:                 int(gguf.GetU32(p+".attention.key_length_mla", 0)),
		MLAValueDim:               int(gguf.GetU32(p+".attention.value_length_mla", 0)),
		LeadingDenseBlockCount:    int(gguf.GetU32(p+".leading_dense_block_count", 0)),
		ExpertFeedForwardDim:      int(gguf.GetU32(p+".expert_feed_forward_length", 0)),
		ExpertSharedCount:         int(gguf.GetU32(p+".expert_shared_count", 0)),
		ExpertGroupCount:          int(gguf.GetU32(p+".expert_group_count", 0)),
		ExpertGroupUsedCount:      int(gguf.GetU32(p+".expert_group_used_count", 0)),
		ExpertGatingFunc:          int(gguf.GetU32(p+".expert_gating_func", 0)),
		RopeDimensionCount:        int(gguf.GetU32(p+".rope.dimension_count", uint32(max(1, headDim)))),
		RopeScalingFactor:         gguf.GetF32(p+".rope.scaling.factor", 1),
		RopeAttentionFactor:       gguf.GetF32(p+".rope.scaling.attn_factor", 1),
		RopeOriginalContextLength: int(gguf.GetU32(p+".rope.scaling.original_context_length", 0)),
		RopeScalingType:           ropeScalingType(gguf, p),
		RopeYarnBetaFast:          gguf.GetF32(p+".rope.scaling.yarn_beta_fast", 32),
		RopeYarnBetaSlow:          gguf.GetF32(p+".rope.scaling.yarn_beta_slow", 1),
		RopeYarnLogMultiplier:     gguf.GetF32(p+".rope.scaling.yarn_log_multiplier", 1),
		UseLayerNorm:              usesLayerNorm(arch),
		ParallelResidual:          parallelResidual || forcesParallelResidual(arch),
		UseGELU:                   gemmaFamily(arch) || arch == "phi2",
		UseExactGELU:              arch == "phi2",
		ALiBiMaxBias:              aLiBiMaxBias(gguf, arch, p),
		AttnLogitSoftcap:          gguf.GetF32(p+".attn_logit_softcapping", 0),
		FinalLogitSoftcap:         gguf.GetF32(p+".final_logit_softcapping", 0),
		SWAPattern:                swaPattern(gguf, arch, p, int(gguf.GetU32(p+".block_count", 0))),
	}
	// Mistral 3's temperature schedule uses the original YaRN context length
	// as its floor (llama.cpp's n_attn_temp_floor_scale).
	cfg.AttentionTemperatureFloor = cfg.RopeOriginalContextLength
	if deepSeek2Family(arch) {
		// The compact K/V cache is only valid when all MLA dimensions are
		// present.  The tensor loader performs the stricter per-layer shape
		// validation and returns a useful diagnostic for malformed GGUFs.
		cfg.UsesMLA = cfg.MLAKVLoRARank > 0 && cfg.MLAKeyDim > 0 && cfg.MLAValueDim > 0
	}
	if arch == "nemotron_h" || arch == "nemotron_h_moe" || arch == "mamba2" {
		cfg.LayerHeads = u32ArrayAsInts(gguf, p+".attention.head_count")
		cfg.LayerKVHeads = u32ArrayAsInts(gguf, p+".attention.head_count_kv")
		cfg.LayerFFNDim = u32ArrayAsInts(gguf, p+".feed_forward_length")
		cfg.SSMConv = int(gguf.GetU32(p+".ssm.conv_kernel", 0))
		cfg.SSMInner = int(gguf.GetU32(p+".ssm.inner_size", 0))
		cfg.SSMState = int(gguf.GetU32(p+".ssm.state_size", 0))
		cfg.SSMHeads = int(gguf.GetU32(p+".ssm.time_step_rank", 0))
		cfg.SSMGroups = int(gguf.GetU32(p+".ssm.group_count", 0))
	}
	if arch == "qwen35" || arch == "qwen35moe" {
		cfg.SSMConv = int(gguf.GetU32(p+".ssm.conv_kernel", 0))
		cfg.SSMInner = int(gguf.GetU32(p+".ssm.inner_size", 0))
		cfg.SSMState = int(gguf.GetU32(p+".ssm.state_size", 0))
		cfg.SSMHeads = int(gguf.GetU32(p+".ssm.time_step_rank", 0))
		cfg.SSMGroups = int(gguf.GetU32(p+".ssm.group_count", 0))
		cfg.FullAttentionInterval = int(gguf.GetU32(p+".full_attention_interval", 0))
		cfg.NextNPredictLayers = int(gguf.GetU32(p+".nextn_predict_layers", 0))
		if v, ok := gguf.Metadata[p+".attention.recurrent_layers"]; ok {
			cfg.QwenRecurrentLayers, _ = v.AsBoolArray()
		}
	}
	switch arch {
	case "command-r":
		applyCommandRLogitScale(&cfg)
	case "minicpm":
		applyMiniCPMScaleDefaults(&cfg, gguf, p, dim)
	case "minicpm3":
		applyMiniCPM3HardcodedScales(&cfg, dim)
	}
	if arch == "exaone4" {
		// EXAONE 4 rotates only its SWA blocks, so the optional SWA-specific
		// base is the one the active RoPE table must use.
		cfg.RopeTheta = gguf.GetF32(p+".rope.freq_base_swa", cfg.RopeTheta)
		cfg.NextNPredictLayers = int(gguf.GetU32(p+".nextn_predict_layers", 0))
	}
	if arch == "olmo2" && cfg.SlidingWindow > 0 {
		cfg.RopeThetaSWA = gguf.GetF32(p+".rope.freq_base_swa", cfg.RopeTheta)
	}
	if v, ok := gguf.Metadata[p+".expert_weights_norm"]; ok {
		if norm, ok := v.AsBool(); ok {
			cfg.ExpertWeightsNorm = norm
		}
	}
	return cfg
}

// applyCommandRLogitScale corrects the direction of Command-R's logit_scale.
// llama.cpp's graph multiplies logits by the raw GGUF value directly; the
// generic path in ProjectLogitsInto divides by Config.LogitScale, so this
// stores the reciprocal to reproduce the same multiply.
func applyCommandRLogitScale(cfg *Config) {
	if cfg.LogitScale != 0 {
		cfg.LogitScale = 1 / cfg.LogitScale
	}
}

// applyMiniCPMScaleDefaults fills MiniCPM's three multiplier fields with
// llama.cpp's exact hardcoded defaults whenever the corresponding GGUF key
// is absent — older MiniCPM 1B/2B exports omit them entirely rather than
// writing an explicit 1/0.
func applyMiniCPMScaleDefaults(cfg *Config, gguf *GGUFFile, p string, dim int) {
	if _, ok := gguf.Metadata[p+".embedding_scale"]; !ok {
		cfg.EmbeddingScale = 12.0
	}
	if _, ok := gguf.Metadata[p+".residual_scale"]; !ok && cfg.NLayers > 0 {
		cfg.ResidualScale = float32(1.4 / math.Sqrt(float64(cfg.NLayers)))
	}
	if _, ok := gguf.Metadata[p+".logit_scale"]; !ok && dim > 0 {
		cfg.LogitScale = 256.0 / float32(dim)
	}
}

// applyMiniCPM3HardcodedScales sets MiniCPM3's three multiplier fields
// unconditionally: unlike MiniCPM, its GGUF never writes these keys at all,
// so llama.cpp hardcodes them directly in its graph (its own source carries
// a "TODO: if the model varies, these need to be read from the model"
// comment acknowledging this is a one-checkpoint-family simplification).
// The logit multiply is stored as its reciprocal so the generic
// 1/Config.LogitScale apply in ProjectLogitsInto reproduces it.
func applyMiniCPM3HardcodedScales(cfg *Config, dim int) {
	cfg.EmbeddingScale = 12.0
	if cfg.NLayers > 0 {
		cfg.ResidualScale = float32(1.4 / math.Sqrt(float64(cfg.NLayers)))
	}
	if dim > 0 {
		cfg.LogitScale = float32(dim) / 256.0
	}
}

// aLiBiMaxBias returns the ALiBi max-bias value (0 disables ALiBi). BLOOM
// hardcodes 8.0 unconditionally (llama.cpp does not read it from any GGUF
// key for this arch); MPT reads it from an explicit key that is 0 whenever
// the source checkpoint had ALiBi disabled. The decision keys on the
// canonical architecture label; the key read goes through the metadata
// namespace p, which can differ when the label was resolved from an alias.
func aLiBiMaxBias(gguf *GGUFFile, arch, p string) float32 {
	switch arch {
	case "bloom":
		return 8.0
	case "mpt":
		return gguf.GetF32(p+".attention.max_alibi_bias", 0)
	default:
		return 0
	}
}

func u32ArrayAsInts(gguf *GGUFFile, key string) []int {
	v, ok := gguf.GetU32Array(key)
	if !ok {
		return nil
	}
	out := make([]int, len(v))
	for i, n := range v {
		out[i] = int(n)
	}
	return out
}

// swaPattern determines which layers use the sliding window. Priority:
// explicit bool-array metadata ({arch}.attention.sliding_window_pattern, one
// entry per layer — the Gemma 4 convention), an explicit scalar period (the
// EXAONE 4 convention), then known per-architecture defaults expressed as
// "every Nth layer is global". nil means every layer uses the window (the
// pre-Gemma behavior, correct for e.g. old Mistral). The per-architecture
// defaults key on the canonical label arch; the metadata read goes through
// the namespace p.
func swaPattern(gguf *GGUFFile, arch, p string, nLayers int) []bool {
	period := 0
	if v, ok := gguf.Metadata[p+".attention.sliding_window_pattern"]; ok {
		if arr, ok := v.AsBoolArray(); ok && len(arr) > 0 {
			return arr
		}
		if n, ok := v.AsU32(); ok {
			period = int(n)
		}
	}
	if period == 0 {
		switch arch {
		case "gemma2":
			period = 2
		case "gemma3", "gemma4":
			period = 6
		case "gpt-oss":
			// GPT-OSS alternates local attention on even layers with full
			// attention on odd layers. The generic pattern below marks the first
			// period-1 layer as local, which is exactly that 1:1 schedule.
			period = 2
		case "exaone4":
			// EXAONE 4 32B uses three local layers followed by one global layer.
			// The 1.2B model has no sliding window, so this pattern remains inert.
			period = 4
		case "olmo2":
			// OLMo 3 GGUFs retain the olmo2 architecture label and use the
			// same three-local/one-global pattern.
			period = 4
		}
	}
	if period == 0 || nLayers <= 0 {
		return nil
	}
	pattern := make([]bool, nLayers)
	for il := range pattern {
		pattern[il] = il%period < period-1
	}
	return pattern
}
