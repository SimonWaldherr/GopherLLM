package gopherllm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
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
	UseGELU           bool
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
	// QwenRecurrentLayers, when supplied by a Qwen3.5/3.6 GGUF, is the
	// authoritative per-layer hybrid schedule: true denotes a Gated DeltaNet
	// layer and false denotes ordinary full attention.  Older exports expose
	// only FullAttentionInterval, for which the loader retains the documented
	// every-Nth-layer fallback.
	QwenRecurrentLayers []bool
	// NextNPredictLayers is the number of MTP/NextN draft blocks appended to
	// the Qwen3.5/3.6 decoder stack. They are not part of normal autoregressive
	// inference unless speculative decoding is enabled.
	NextNPredictLayers int
	// FullAttentionInterval is Qwen3.5/3.6's hybrid schedule: layer il (0
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
// use a learned LayerNorm for both branches; Phi-2 does the same with its
// ungated-GELU MLP.
func (c Config) sharesParallelBranchNorm() bool {
	return c.ParallelResidual && (c.Arch == "phi2" || c.Arch == "stablelm")
}

// usesFullProjectionQKNorm distinguishes OLMo 2/3's one RMSNorm over the
// complete Q or K projection from the shared per-head norm used by Qwen3 and
// EXAONE 4.
func (c Config) usesFullProjectionQKNorm() bool {
	return c.Arch == "olmo2"
}

func ConfigFromGGUF(gguf *GGUFFile) Config {
	arch, ok := gguf.GetString("general.architecture")
	if !ok || arch == "" {
		arch = "llama"
	}
	// A few Kimi converters preserve Hugging Face's `kimi_k2` architecture
	// label but retain llama.cpp's `deepseek2.*` hparam namespace. Keep the
	// public architecture label (it drives loader/template selection) while
	// reading the namespace that is actually present in the GGUF.
	p := arch
	// `llama2`/`llama3` are compatibility labels used by a few third-party
	// converters. Canonical GGUF metadata still lives under `llama.*`; prefer
	// an alias-specific namespace when it exists, otherwise read the canonical
	// keys while preserving the public architecture label.
	if arch == "llama2" || arch == "llama3" {
		if _, hasAliasPrefix := gguf.Metadata[p+".embedding_length"]; !hasAliasPrefix {
			if _, hasLlamaPrefix := gguf.Metadata["llama.embedding_length"]; hasLlamaPrefix {
				p = "llama"
			}
		}
	}
	if arch == "kimi_k2" {
		if _, hasKimiPrefix := gguf.Metadata[p+".embedding_length"]; !hasKimiPrefix {
			if _, hasDeepSeekPrefix := gguf.Metadata["deepseek2.embedding_length"]; hasDeepSeekPrefix {
				p = "deepseek2"
			}
		}
	}
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
		if gemmaFamily(p) && dim > 0 {
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
	if p == "stablelm" || p == "phi2" {
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
		UseLayerNorm:              arch == "stablelm" || arch == "phi2",
		ParallelResidual:          parallelResidual || arch == "phi2",
		UseGELU:                   gemmaFamily(arch) || arch == "phi2",
		AttnLogitSoftcap:          gguf.GetF32(p+".attn_logit_softcapping", 0),
		FinalLogitSoftcap:         gguf.GetF32(p+".final_logit_softcapping", 0),
		SWAPattern:                swaPattern(gguf, p, int(gguf.GetU32(p+".block_count", 0))),
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
	if p == "nemotron_h" || p == "nemotron_h_moe" || p == "mamba2" {
		cfg.LayerHeads = u32ArrayAsInts(gguf, p+".attention.head_count")
		cfg.LayerKVHeads = u32ArrayAsInts(gguf, p+".attention.head_count_kv")
		cfg.LayerFFNDim = u32ArrayAsInts(gguf, p+".feed_forward_length")
		cfg.SSMConv = int(gguf.GetU32(p+".ssm.conv_kernel", 0))
		cfg.SSMInner = int(gguf.GetU32(p+".ssm.inner_size", 0))
		cfg.SSMState = int(gguf.GetU32(p+".ssm.state_size", 0))
		cfg.SSMHeads = int(gguf.GetU32(p+".ssm.time_step_rank", 0))
		cfg.SSMGroups = int(gguf.GetU32(p+".ssm.group_count", 0))
	}
	if p == "qwen35" || p == "qwen35moe" {
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
	if p == "exaone4" {
		// EXAONE 4 rotates only its SWA blocks, so the optional SWA-specific
		// base is the one the active RoPE table must use.
		cfg.RopeTheta = gguf.GetF32(p+".rope.freq_base_swa", cfg.RopeTheta)
		cfg.NextNPredictLayers = int(gguf.GetU32(p+".nextn_predict_layers", 0))
	}
	if p == "olmo2" && cfg.SlidingWindow > 0 {
		cfg.RopeThetaSWA = gguf.GetF32(p+".rope.freq_base_swa", cfg.RopeTheta)
	}
	if v, ok := gguf.Metadata[p+".expert_weights_norm"]; ok {
		if norm, ok := v.AsBool(); ok {
			cfg.ExpertWeightsNorm = norm
		}
	}
	return cfg
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
// pre-Gemma behavior, correct for e.g. old Mistral).
func swaPattern(gguf *GGUFFile, p string, nLayers int) []bool {
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
		switch p {
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

// Weight is one loaded tensor in one of three states: F32 non-nil (owned
// float32 values), Raw quantized bytes, or Raw scalar F32/F16/BF16/F64 bytes.
// The last state is used by out-of-core loads: it keeps a mmap-backed matrix
// in its on-disk representation and converts only values used by a row dot.
// Rows/Cols describe every Raw form; the owned F32 form infers rows from
// len(F32)/cols at the call site for compatibility with existing callers.
type Weight struct {
	F32      []float32
	Raw      []byte
	Type     GGMLType
	Rows     int
	Cols     int
	Prepared *PreparedQuantizedWeight
	Metal    *MetalWeight
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
	if w.rawScalarMatvecInto(x, out) {
		return
	}
	switch w.Type {
	case GGMLTypeQ8_0:
		MatvecQ8_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_0:
		MatvecQ4_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeIQ4_NL:
		MatvecIQ4NLInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeIQ2_S:
		MatvecIQ2SInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeIQ3_S:
		MatvecIQ3SInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeIQ4_XS:
		MatvecIQ4XSInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_1:
		MatvecQ4_1Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ5_0:
		MatvecQ5_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ5_1:
		MatvecQ5_1Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ8_1:
		MatvecQ8_1Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ8_K:
		MatvecQ8KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ2_K:
		MatvecQ2KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ3_K:
		MatvecQ3KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_K:
		if matvecMetalQ4KInto(w.Metal, x, w.Rows, w.Cols, out) {
			return
		}
		if MatvecPreparedQ4KInto(w.Raw, w.Prepared, x, w.Rows, w.Cols, out) {
			return
		}
		MatvecQ4KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ5_K:
		MatvecQ5KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ6_K:
		if matvecMetalQ6KInto(w.Metal, x, w.Rows, w.Cols, out) {
			return
		}
		if MatvecPreparedQ6KInto(w.Raw, w.Prepared, x, w.Rows, w.Cols, out) {
			return
		}
		MatvecQ6KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeMXFP4:
		MatvecMXFP4Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeTQ1_0:
		MatvecTQ1_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeTQ2_0:
		MatvecTQ2_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ1_0:
		MatvecQ1_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ2_0:
		MatvecQ2_0Into(w.Raw, x, w.Rows, w.Cols, out)
	default:
		panic(fmt.Sprintf("unsupported quantized matvec: %v", w.Type))
	}
}

// ArgmaxMatvec returns argmax(W*x) without materializing the full logits
// vector. Observed bottleneck: Ministral-3 3B Q4_K_M spends most decode time in
// the 131k-row output projection. For deterministic decoding, the sampler only
// needs the winning token, so this saves the logits writeback and second full
// vocab scan. Risk is limited by using it only for exact greedy-compatible
// sampler settings; rollback is to disable the runtime fast-path. Covers the
// quantized formats llama.cpp's own quantize tool leaves on tied/output
// embeddings: Q6_K (the "_M"/"_L" floor), and Q4_K/Q5_K/Q8_0 (the "_S"
// presets and this project's own --compress skip that floor). Other
// quantized types fall through to the general logits path below.
func (w Weight) ArgmaxMatvec(x []float32) (uint32, bool) {
	if len(x) == 0 {
		return 0, false
	}
	if w.F32 != nil {
		rows := len(w.F32) / len(x)
		if rows <= 0 || rows*len(x) > len(w.F32) {
			return 0, false
		}
		return argmaxMatvecRows(rows, func(row int) float32 {
			off := row * len(x)
			return DotF32(w.F32[off:off+len(x)], x)
		}), true
	}
	if token, ok := w.rawScalarArgmaxMatvec(x); ok {
		return token, true
	}
	if w.Rows <= 0 || w.Cols != len(x) {
		return 0, false
	}
	switch w.Type {
	case GGMLTypeQ6_K:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 256) * 210
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ6KXSums16(x, w.Cols, scratch)
		ScaleF32(xs, 32)
		tok, ok := argmaxQ6KRowsQ8(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ6KRowsWithXSums(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return tok, true
	case GGMLTypeQ4_K:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 256) * 144
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, w.Cols, scratch)
		tok, ok := argmaxQ4KRowsQ8(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ4KRowsWithXSums(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return tok, true
	case GGMLTypeQ5_K:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 256) * 176
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		// Q5_K shares Q4_K's per-sub-block scale/min structure (see
		// MatvecQ5KInto), so the same per-32-element activation sums apply.
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, w.Cols, scratch)
		tok, ok := argmaxQ5KRowsQ8(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ5KRowsFloat(w.Raw, x, w.Rows, w.Cols, rowBytes)
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return tok, true
	case GGMLTypeQ8_0:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 32) * 34
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		tok, ok := argmaxQ8_0RowsQ8(w.Raw, x, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ8_0RowsFloat(w.Raw, x, w.Rows, w.Cols, rowBytes)
		}
		return tok, true
	default:
		return 0, false
	}
}

// argmaxQ6KRowsWithXSums is the f32-activation counterpart of the amd64
// Q8-activation greedy path. It keeps the Q6_K row dot directly in the loop
// instead of passing it through argmaxMatvecRows' function value: on ARM64
// that removes an indirect call for every vocabulary row while preserving the
// exact floating-point dot product, finite-value filtering, and lowest-token
// tie behavior. xsums must already include the Q6_K offset factor (×32).
func argmaxQ6KRowsWithXSums(data []byte, x, xsums []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			var v float32
			if hasQuantSIMD {
				v = dotQ6KF32SIMDWithXSums(data[off:off+rowBytes], x, xsums, cols)
			} else {
				v = DotQ6KF32(data[off:off+rowBytes], x, cols)
			}
			if !finiteLogit(v) {
				continue
			}
			// Rows are visited in ascending order; strict > retains the
			// lowest row index within a worker. The reduction below retains it
			// globally too, matching argmaxFiniteToken.
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken, bestValue, found = localToken, localValue, true
		}
		mu.Unlock()
	})
	return uint32(bestToken)
}

// argmaxQ4KRowsWithXSums is the exact-float counterpart of argmaxQ4KRowsQ8,
// used when the int8-activation path is unavailable or disabled.
// dotQ4KF32WithXSums already picks the SIMD or scalar kernel internally.
func argmaxQ4KRowsWithXSums(data []byte, x, xsums []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			v := dotQ4KF32WithXSums(data[off:off+rowBytes], x, xsums, cols)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken, bestValue, found = localToken, localValue, true
		}
		mu.Unlock()
	})
	return uint32(bestToken)
}

// argmaxQ5KRowsFloat is the exact-float counterpart of argmaxQ5KRowsQ8. Q5_K
// has no SIMD-with-xsums float kernel (unlike Q4_K/Q6_K), so this always uses
// the plain per-row dot product, matching MatvecQ5KInto's float fallback.
func argmaxQ5KRowsFloat(data []byte, x []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			v := DotQ5KF32(data[off:off+rowBytes], x, cols)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken, bestValue, found = localToken, localValue, true
		}
		mu.Unlock()
	})
	return uint32(bestToken)
}

// argmaxQ8_0RowsFloat is the exact-float counterpart of argmaxQ8_0RowsQ8.
func argmaxQ8_0RowsFloat(data []byte, x []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			v := DotQ8_0F32(data[off:off+rowBytes], x, cols)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken, bestValue, found = localToken, localValue, true
		}
		mu.Unlock()
	})
	return uint32(bestToken)
}

func argmaxMatvecRows(rows int, dot func(row int) float32) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			v := dot(row)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue || (v == localValue && row < localToken) {
				localToken = row
				localValue = v
				localFound = true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken = localToken
			bestValue = localValue
			found = true
		}
		mu.Unlock()
	})
	return uint32(bestToken)
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
	if w.rawScalarRowInto(row, cols, out) {
		return
	}
	switch w.Type {
	case GGMLTypeQ8_0:
		rowBytes := (cols / 32) * 34
		DequantRowQ8_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ4_0:
		rowBytes := (cols / 32) * 18
		DequantRowQ4_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeIQ4_NL:
		rowBytes := (cols / 32) * 18
		DequantRowIQ4NLInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeIQ2_S:
		rowBytes := (cols / 256) * 82
		DequantRowIQ2SInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeIQ3_S:
		rowBytes := (cols / 256) * 110
		DequantRowIQ3SInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeIQ4_XS:
		rowBytes := (cols / 256) * 136
		DequantRowIQ4XSInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ4_1:
		rowBytes := (cols / 32) * 20
		DequantRowQ4_1Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ5_0:
		rowBytes := (cols / 32) * 22
		DequantRowQ5_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ5_1:
		rowBytes := (cols / 32) * 24
		DequantRowQ5_1Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ8_1:
		rowBytes := (cols / 32) * 36
		DequantRowQ8_1Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ8_K:
		rowBytes := (cols / 256) * 292
		DequantRowQ8KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ2_K:
		rowBytes := (cols / 256) * 84
		DequantRowQ2KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ3_K:
		rowBytes := (cols / 256) * 110
		DequantRowQ3KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ4_K:
		rowBytes := (cols / 256) * 144
		DequantRowQ4KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ5_K:
		rowBytes := (cols / 256) * 176
		DequantRowQ5KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ6_K:
		rowBytes := (cols / 256) * 210
		DequantRowQ6KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeMXFP4:
		rowBytes := (cols / 32) * 17
		DequantRowMXFP4Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeTQ1_0:
		rowBytes := (cols / 256) * 54
		DequantRowTQ1_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeTQ2_0:
		rowBytes := (cols / 256) * 66
		DequantRowTQ2_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ1_0:
		rowBytes := (cols / 128) * 18
		DequantRowQ1_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ2_0:
		rowBytes := (cols / 64) * 18
		DequantRowQ2_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	default:
		panic(fmt.Sprintf("unsupported quantized row extraction: %v", w.Type))
	}
}

func releaseModelMetalWeights(weights *ModelWeights) {
	if weights == nil {
		return
	}
	seen := map[*MetalWeight]bool{}
	release := func(w *Weight) {
		if w == nil || w.Metal == nil {
			return
		}
		if !seen[w.Metal] {
			releaseMetalWeight(w.Metal)
			seen[w.Metal] = true
		}
		w.Metal = nil
	}
	release(&weights.TokenEmbd)
	release(&weights.Output)
	for i := range weights.Layers {
		layer := &weights.Layers[i]
		release(&layer.WQ)
		release(&layer.WK)
		release(&layer.WV)
		release(&layer.WQKV)
		release(&layer.WO)
		release(&layer.W1)
		release(&layer.W2)
		release(&layer.W3)
		release(&layer.WGateUp)
		if layer.MLA != nil {
			release(&layer.MLA.Q)
			release(&layer.MLA.QA)
			release(&layer.MLA.QB)
			release(&layer.MLA.KVA)
			release(&layer.MLA.KB.Weight)
			release(&layer.MLA.VB.Weight)
		}
		if layer.MoE != nil {
			moe := layer.MoE
			release(&moe.Router)
			release(&moe.Gate.Weight)
			release(&moe.Up.Weight)
			release(&moe.Down.Weight)
			release(moe.SharedGateIn)
			release(moe.SharedGate)
			release(moe.SharedUp)
			release(moe.SharedDown)
		}
	}
}

func (w Weight) RowF32(row, cols int) []float32 {
	if w.F32 != nil {
		start := row * cols
		return w.F32[start : start+cols]
	}
	if rawScalarWeight(w) {
		// A raw scalar mapping has no stable []float32 view. Return a decoded
		// row instead; RowF32 is a convenience accessor, not a mutation API.
		return w.Row(row, cols)
	}
	panic("expected f32 row storage")
}

// LayerWeights holds one transformer block. Attention is either split
// (WQ/WK/WV, with optional biases BQ/BK/BV) or fused into a single WQKV
// (HasQKV); the SwiGLU FFN is likewise either split (W1 = gate, W3 = up,
// W2 = down — llama.cpp naming) or fused gate+up in WGateUp (HasGateUp).
type LayerWeights struct {
	AttnNorm     []float32
	AttnNormBias []float32
	WQ           Weight
	BQ           []float32
	WK           Weight
	BK           []float32
	WV           Weight
	BV           []float32
	WQKV         Weight
	HasQKV       bool
	WO           Weight
	BO           []float32
	FFNNorm      []float32
	FFNNormBias  []float32
	W1           Weight
	W2           Weight
	W3           Weight
	FFNUpBias    []float32
	FFNDownBias  []float32
	WGateUp      Weight
	HasGateUp    bool
	// MLA is set for DeepSeek-V2/V3 and Kimi-K2 attention blocks.  Those
	// models cache compressed K/V latents and therefore cannot use WQ/WK/WV
	// ordinary-GQA attention even though their surrounding residual/FFN graph
	// is llama-like.
	MLA *MLAAttentionWeights
	// MoE replaces the dense FFN tensors above for sparse decoder blocks.
	// It is nil for ordinary SwiGLU layers.
	MoE *SparseMoEWeights
	// Optional Gemma-family norms, nil when the tensors are absent:
	// AttnQNorm/AttnKNorm are per-head RMS norms of length HeadDim applied
	// after the Q/K projections (before RoPE); PostAttnNorm/PostFFNNorm are
	// full-width RMS norms applied to the attention/FFN outputs before their
	// residual adds.
	AttnQNorm    []float32
	AttnKNorm    []float32
	PostAttnNorm []float32
	PostFFNNorm  []float32
	// Learned no-value attention sink used by GPT-OSS; nil for ordinary
	// attention. Each entry corresponds to one query head.
	AttnSinks []float32
}

type ModelWeights struct {
	TokenEmbd      Weight
	OutputNorm     []float32
	OutputNormBias []float32
	Output         Weight
	OutputBias     []float32
	Layers         []LayerWeights
}

type Gemma4LayerWeights struct {
	// Native is true for the actual Gemma 4 graph.  Unlike Gemma 1--3, Gemma
	// 4 changes attention geometry between SWA and global blocks, may omit V
	// (then K is used as V), and applies a layer output scale after the whole
	// block.  The older fields below are retained for the generic Gemma path
	// and for API compatibility with the original wrapper.
	Native bool

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

	// Native Gemma 4-only tensors and per-layer geometry.  The names mirror
	// the GGUF tensors and llama.cpp graph: Q/K have learned per-head RMS
	// norms; V is RMS-normalized without a learned weight; post norms sit
	// before each residual add; OutputScale is applied last in the block.
	AttnQNorm    []float32
	AttnKNorm    []float32
	PostAttnNorm []float32
	PostFFNNorm  []float32
	OutputScale  float32
	IsSWA        bool
	HasKV        bool
	UsesKAsV     bool
	// KVCacheSlot compacts native Gemma 4 shared-KV layouts. Dense 12B/26B
	// layers use their own slot; E2B tail layers read the matching preceding
	// local/global slot and never write K/V of their own.
	KVCacheSlot   int
	RopeDimension int
	RopeInvFreq   []float32
	FFNHiddenDim  int
	// Per-layer embedding (PLE) tensors are present on E2B-style checkpoints.
	// They inject a token-specific gated low-rank residual after the ordinary
	// attention/FFN block and before OutputScale.
	PerLayerInputGate Weight
	PerLayerProj      Weight
	PerLayerPostNorm  []float32
}

// Gemma4PerLayerWeights holds the global part of Gemma 4 E2B's PLE graph.
// TokenEmbd stores one Dim-wide slice per decoder layer in every token row.
// ModelProj and ProjNorm mix the scaled regular input into those slices before
// each layer's own gate/projection consumes its selected slice.
type Gemma4PerLayerWeights struct {
	TokenEmbd Weight
	ModelProj Weight
	ProjNorm  []float32
	Dim       int
}

type Gemma4Weights struct {
	TokenEmbd  Weight
	OutputNorm []float32
	Output     Weight
	Layers     []Gemma4LayerWeights
	// MoE holds Gemma 4's special shared-dense-plus-expert FFN state by
	// decoder layer. It intentionally is not SparseMoEWeights: Gemma 4 routes
	// from attn_out through a scaled RMS input and combines separately
	// normalized dense/expert branches.
	MoE []*Gemma4MoEWeights
	// PerLayer is non-nil for E2B-style per-layer embeddings.
	PerLayer *Gemma4PerLayerWeights
	// Native selects the isolated Gemma 4 execution graph.  Standard remains
	// populated for Gemma/Gemma2/Gemma3 and legacy synthetic Gemma4 fixtures.
	Native   bool
	Standard ModelWeights
}

type GptOssWeights struct {
	Standard ModelWeights
}

// KVCache stores the attention keys and values of every processed position,
// one flat slice per layer laid out position-major: position p's keys occupy
// K[layer][p*PerPosKDim : (p+1)*PerPosKDim] (all KV heads concatenated), and
// likewise for V. Sized to prompt+max_tokens (capped at the model context
// length) and reused by Runner when capacity permits; there is no ring/eviction
// and generation stops at its request-local cache length.
type KVCache struct {
	K          [][]float32
	V          [][]float32
	PerPosKDim int
	PerPosVDim int
	MaxLen     int
	// F16 selects half-precision row storage: K16/V16 replace K/V entirely
	// and attention converts rows in-register (see kv_f16.go). Halves the
	// cache's memory footprint and the bytes attention streams per token.
	F16 bool
	// Nemotron is the shared recurrent cache for Mamba-2 graphs. Hybrid
	// Nemotron-H also uses K/V rows; pure Mamba2 leaves those dimensions empty.
	Nemotron *NemotronHCache
	// Qwen35 is the Gated DeltaNet recurrent state for the qwen35/qwen35moe
	// hybrid graph. Its periodic full-attention layers use the K/V rows above,
	// same dual-cache pattern as Nemotron-H.
	Qwen35 *Qwen35Cache
	K16    [][]uint16
	V16    [][]uint16
}

// NewKVCache allocates an f32 cache for `layers` layers of maxLen positions
// with the given per-position K and V widths (see KVCache).
func NewKVCache(layers, kDim, vDim, maxLen int) *KVCache {
	k := make([][]float32, layers)
	v := make([][]float32, layers)
	for i := range layers {
		k[i] = make([]float32, maxLen*kDim)
		v[i] = make([]float32, maxLen*vDim)
	}
	return &KVCache{K: k, V: v, PerPosKDim: kDim, PerPosVDim: vDim, MaxLen: maxLen}
}

// NewKVCacheF16 is NewKVCache with half-precision row storage.
func NewKVCacheF16(layers, kDim, vDim, maxLen int) *KVCache {
	k := make([][]uint16, layers)
	v := make([][]uint16, layers)
	for i := range layers {
		k[i] = make([]uint16, maxLen*kDim)
		v[i] = make([]uint16, maxLen*vDim)
	}
	return &KVCache{K16: k, V16: v, F16: true, PerPosKDim: kDim, PerPosVDim: vDim, MaxLen: maxLen}
}

// newKVCacheAuto picks f16 storage where the platform supports it (see
// useF16KVCache) and exact f32 storage everywhere else.
func newKVCacheAuto(layers, kDim, vDim, maxLen int) *KVCache {
	if useF16KVCache.Load() {
		return NewKVCacheF16(layers, kDim, vDim, maxLen)
	}
	return NewKVCache(layers, kDim, vDim, maxLen)
}

// layerCount reports the number of layers the cache was allocated for,
// regardless of element format.
func (c *KVCache) layerCount() int {
	if c.F16 {
		return len(c.K16)
	}
	return len(c.K)
}

// storeKV writes one position's K and V rows into the cache in its native
// element format.
func (c *KVCache) storeKV(l, pos int, k, v []float32) {
	kStart := pos * c.PerPosKDim
	vStart := pos * c.PerPosVDim
	if c.F16 {
		f32ToF16Row(c.K16[l][kStart:kStart+min(len(k), c.PerPosKDim)], k)
		f32ToF16Row(c.V16[l][vStart:vStart+min(len(v), c.PerPosVDim)], v)
		return
	}
	copy(c.K[l][kStart:kStart+min(len(k), c.PerPosKDim)], k)
	copy(c.V[l][vStart:vStart+min(len(v), c.PerPosVDim)], v)
}

// attendHead runs online attention for one query head against this cache's
// rows, dispatching to the storage format's kernel set.
func (c *KVCache) attendHead(l, kvH int, query []float32, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	c.attendHeadWithSink(l, kvH, query, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, 0, false, out)
}

// attendHeadWithSink is attendHead with an optional learned no-value sink
// logit. GPT-OSS appends that logit to each head's softmax denominator without
// adding a value row, which dampens attention when the learned sink wins.
func (c *KVCache) attendHeadWithSink(l, kvH int, query []float32, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap, sink float32, hasSink bool, out []float32) {
	if c.F16 {
		onlineAttentionF16WithSink(query, c.K16[l][kvH*keyHeadDim:], c.V16[l][kvH*valueHeadDim:],
			c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, sink, hasSink, out)
		return
	}
	onlineAttentionWithSink(query, c.K[l][kvH*keyHeadDim:], c.V[l][kvH*valueHeadDim:],
		c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, sink, hasSink, out)
}

// attendHeadGroup evaluates all query heads that share one GQA/MQA KV head.
// Keeping their score and value passes adjacent lets the CPU reuse each K/V
// cacheline instead of streaming the same cache once per query head.
func (c *KVCache) attendHeadGroup(l, kvH int, queries []float32, queryHeads, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	if c.F16 {
		onlineAttentionGroupF16(queries, c.K16[l][kvH*keyHeadDim:], c.V16[l][kvH*valueHeadDim:],
			queryHeads, c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim,
			startT, endT, scale, softcap, out)
		return
	}
	onlineAttentionGroup(queries, c.K[l][kvH*keyHeadDim:], c.V[l][kvH*valueHeadDim:],
		queryHeads, c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim,
		startT, endT, scale, softcap, out)
}

// DecodeBuffer is reusable request scratch for single-token decode and batched
// prefill: activation vectors (X residual stream, XN/XN2 normed views,
// Q/K/V/AttnOut/Proj attention buffers, Gate/Up/Hidden FFN buffers), the
// output logits and generation/token scratch, the sampler's candidate scratch,
// and precomputed RoPE tables (per-pair inverse frequencies plus per-position
// sin/cos filled in prepareRopeScratch). One DecodeBuffer serves successive
// requests, so decode allocates nothing per token and prefill reuses its
// activation slabs. Not safe for concurrent use; Runner.genLock serializes
// requests.
type DecodeBuffer struct {
	X        []float32
	XN       []float32
	XN2      []float32
	Q        []float32
	K        []float32
	V        []float32
	QKV      []float32
	AttnOut  []float32
	Proj     []float32
	AttnProj []float32
	Gate     []float32
	Up       []float32
	GateUp   []float32
	Hidden   []float32
	MOE      []float32
	// MoELatent keeps Nemotron-H's optional projected expert input alive
	// while each selected expert reuses MOE for its output.
	MoELatent []float32
	// MLA scratch holds the Q LoRA intermediate, the compact KV projection,
	// and a temporary RoPE/value expansion plane.  It is reused across all
	// layers so Kimi/DeepSeek decode remains allocation-free.
	MLAQ            []float32
	MLAKV           []float32
	MLATmp          []float32
	MLAValues       []float32
	RouterLogits    []float32
	RouterSelection []float32
	// RouterGroups/TopGroups are DeepSeek-V3's allocation-free scratch for
	// group-limited noaux routing. They remain small (eight groups in V3)
	// while TopExperts retains the final selected expert indices.
	RouterGroups            []float32
	TopGroups               []ExpertScore
	TopExperts              []ExpertScore
	ExpertProbs             []float32
	SamplerCandidates       []TokenProb
	Logits                  []float32
	RecentTokens            []uint32
	GeneratedTokens         []uint32
	StreamBytes             []byte
	Q4KXSums                []float32
	RopeInvFreq             []float32
	RopeSin                 []float32
	RopeCos                 []float32
	RopeMscale              float32
	RopeSWAInvFreq          []float32
	RopeSWASin              []float32
	RopeSWACos              []float32
	RopeSWAMscale           float32
	RopeGptOssInvFreq       []float32
	RopeGptOssConcentration float32
	MambaIn                 []float32
	MambaConv               []float32
	MambaZ                  []float32
	MambaX                  []float32
	MambaB                  []float32
	MambaC                  []float32
	MambaDT                 []float32
	MambaY                  []float32
	MambaKernel             []float32
	// MambaBeta/MambaRecall are Qwen3.5 Gated DeltaNet scratch: MambaBeta is
	// the per-head delta-rule mixing gate (reuses the Mamba naming scheme
	// rather than adding a parallel "DeltaNet*" set, since they play the same
	// per-head-scratch role as MambaDT etc.), MambaRecall is the single head's
	// worth of scratch for what the state currently predicts for this token's
	// key, before the delta correction.
	MambaBeta   []float32
	MambaRecall []float32
	// QGate/AttnGate are Qwen3.5's gated-attention scratch: attn_q projects
	// each head to [query(headDim) | gate(headDim)] rather than plain
	// headDim, so QGate holds the raw strided projection and AttnGate the
	// extracted, compactly-packed gate half (the query half is copied into
	// the ordinary Q buffer so every existing per-head helper still applies).
	QGate        []float32
	AttnGate     []float32
	ExpertRow    []float32
	ExpertHidden []float32
	// Gemma4PLE and Gemma4PLEInput retain a token's prepared E2B per-layer
	// embedding slices and their raw embedding-row source. They are grown only
	// when a native PLE model is used, so ordinary decoder workspaces stay the
	// same size.
	Gemma4PLE      []float32
	Gemma4PLEInput []float32
	batch          batchDecodeBuffer
}

type ExpertScore struct {
	Index int
	Score float32
}

func NewDecodeBuffer(config Config, maxHeadDim, maxNKVHeads, maxValueDim int) *DecodeBuffer {
	inv, mscale := buildRopeInvFreq(config, maxHeadDim)
	var swaInv []float32
	var swaSin, swaCos []float32
	swaMscale := float32(1)
	if config.RopeThetaSWA > 0 {
		swaConfig := config
		swaConfig.RopeTheta = config.RopeThetaSWA
		swaConfig.RopeScalingType = ""
		swaConfig.RopeScalingFactor = 1
		swaConfig.RopeOriginalContextLength = 0
		swaConfig.RopeFactorsLong = nil
		swaConfig.RopeFactorsShort = nil
		swaInv, swaMscale = buildRopeInvFreq(swaConfig, maxHeadDim)
		swaSin = make([]float32, max(1, maxHeadDim/2))
		swaCos = make([]float32, max(1, maxHeadDim/2))
	}
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
		AttnProj:                make([]float32, config.Dim),
		Gate:                    make([]float32, config.HiddenDim),
		Up:                      make([]float32, config.HiddenDim),
		GateUp:                  make([]float32, config.HiddenDim*2),
		Hidden:                  make([]float32, config.HiddenDim),
		MOE:                     make([]float32, config.Dim),
		MoELatent:               make([]float32, config.Dim),
		MLAQ:                    make([]float32, max(1, config.MLAQueryLoRARank)),
		MLAKV:                   make([]float32, max(1, config.MLAKVLoRARank+config.RopeDimensionCount)),
		MLATmp:                  make([]float32, max(1, max(config.MLAKVLoRARank, config.RopeDimensionCount))),
		MLAValues:               make([]float32, max(1, config.NHeads*config.MLAValueDim)),
		RouterLogits:            make([]float32, config.ExpertCount),
		RouterSelection:         make([]float32, config.ExpertCount),
		RouterGroups:            make([]float32, max(1, config.ExpertGroupCount)),
		TopGroups:               make([]ExpertScore, 0, max(1, config.ExpertGroupUsedCount)),
		TopExperts:              make([]ExpertScore, 0, config.ExpertUsedCount),
		ExpertProbs:             make([]float32, 0, config.ExpertUsedCount),
		SamplerCandidates:       make([]TokenProb, 0, 64),
		Logits:                  make([]float32, config.VocabSize),
		RecentTokens:            make([]uint32, 0, repeatPenaltyWindow),
		GeneratedTokens:         make([]uint32, 0, 64),
		StreamBytes:             make([]byte, 0, 256),
		Q4KXSums:                make([]float32, max(1, config.Dim/32)),
		RopeInvFreq:             inv,
		RopeSin:                 make([]float32, max(1, maxHeadDim/2)),
		RopeCos:                 make([]float32, max(1, maxHeadDim/2)),
		RopeMscale:              mscale,
		RopeSWAInvFreq:          swaInv,
		RopeSWASin:              swaSin,
		RopeSWACos:              swaCos,
		RopeSWAMscale:           swaMscale,
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
func LoadModel(data []byte, gguf *GGUFFile, borrowQuantized, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, ModelWeights, error) {
	if logw == nil {
		logw = io.Discard
	}
	config := ConfigFromGGUF(gguf)
	if deepSeek2Family(config.Arch) {
		// DeepSeek-V2/V3 and Kimi-K2 use MLA rather than the ordinary
		// Q/K/V matrices below.  Keep their loader isolated so an incomplete
		// MLA checkpoint cannot accidentally fall through to a plausible but
		// mathematically wrong GQA graph.
		return LoadDeepSeek2Model(data, gguf, borrowQuantized, prepareQuantized, useMetal, logw, outOfCore...)
	}
	if config.Dim <= 0 || config.NLayers <= 0 || config.NHeads <= 0 {
		return config, ModelWeights{}, fmt.Errorf("invalid model configuration: dim=%d layers=%d heads=%d", config.Dim, config.NLayers, config.NHeads)
	}
	lazyScalarWeights := len(outOfCore) > 0 && outOfCore[0]
	if config.Arch == "exaone4" && config.NextNPredictLayers > 0 {
		if config.NextNPredictLayers >= config.NLayers {
			return config, ModelWeights{}, fmt.Errorf("exaone4: nextn_predict_layers=%d leaves no decoder layers", config.NextNPredictLayers)
		}
		config.NLayers -= config.NextNPredictLayers
		if len(config.SWAPattern) > config.NLayers {
			config.SWAPattern = config.SWAPattern[:config.NLayers]
		}
	}
	fmt.Fprintf(logw, "Config: dim=%d, layers=%d, heads=%d/%d, hidden=%d, vocab=%d, ctx=%d\n",
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

	tokenEmbd, err := loadWeight(data, gguf.DataOffset, "token_embd.weight", tensorIdx, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
	if err != nil {
		return config, ModelWeights{}, err
	}
	outputNorm, err := loadF32Vec(data, gguf.DataOffset, "output_norm.weight", tensorIdx, inferred)
	if err != nil {
		return config, ModelWeights{}, err
	}
	output := tokenEmbd
	if _, ok := tensorIdx["output.weight"]; ok {
		output, err = loadWeight(data, gguf.DataOffset, "output.weight", tensorIdx, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, ModelWeights{}, err
		}
	} else {
		if config.Arch == "olmo2" || config.Arch == "phi2" {
			return config, ModelWeights{}, fmt.Errorf("%s requires output.weight", config.Arch)
		}
		fmt.Fprintln(logw, "Note: output tied to embeddings")
	}
	outputBias := loadOptionalF32VecNil(data, gguf.DataOffset, "output.bias", tensorIdx, inferred)
	outputNormBias := loadOptionalF32Vec(data, gguf.DataOffset, "output_norm.bias", tensorIdx, inferred, config.Dim)
	if config.Arch == "phi2" {
		if len(outputBias) != config.VocabSize {
			return config, ModelWeights{}, fmt.Errorf("phi2 requires %d-element output.bias", config.VocabSize)
		}
		outputNormBias = loadOptionalF32VecNil(data, gguf.DataOffset, "output_norm.bias", tensorIdx, inferred)
		if len(outputNormBias) != config.Dim {
			return config, ModelWeights{}, fmt.Errorf("phi2 requires %d-element output_norm.bias", config.Dim)
		}
	}

	layers := make([]LayerWeights, 0, config.NLayers)
	qRows := config.NHeads * config.HeadDim
	kRows := config.NKVHeads * config.HeadDim
	vRows := config.NKVHeads * config.ValueDim
	for l := range config.NLayers {
		layer, err := loadLayer(data, gguf.DataOffset, l, config, tensorIdx, inferred, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights, qRows, kRows, vRows)
		if err != nil {
			return config, ModelWeights{}, err
		}
		layers = append(layers, layer)
		if l == 0 || (l+1)%8 == 0 || l+1 == config.NLayers {
			fmt.Fprintf(logw, "  Loaded layer %d/%d\n", l+1, config.NLayers)
		}
	}
	return config, ModelWeights{
		TokenEmbd: tokenEmbd, OutputNorm: outputNorm,
		OutputNormBias: outputNormBias,
		Output:         output,
		OutputBias:     outputBias,
		Layers:         layers,
	}, nil
}

func LoadGptOssModel(data []byte, gguf *GGUFFile, borrowQuantized, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, GptOssWeights, error) {
	config, weights, err := LoadModel(data, gguf, borrowQuantized, prepareQuantized, useMetal, logw, outOfCore...)
	return config, GptOssWeights{Standard: weights}, err
}

func LoadGemma4Model(data []byte, gguf *GGUFFile, borrowQuantized, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, Gemma4Weights, error) {
	config := ConfigFromGGUF(gguf)
	// Real Gemma 4 exports are structurally different from the older Gemma
	// decoder graph: their per-layer output scales make a reliable marker that
	// lets us keep the tested Gemma/Gemma2/Gemma3 fallback untouched.
	if config.Arch == "gemma4" && isNativeGemma4Layout(indexTensors(gguf)) {
		return loadNativeGemma4Model(data, gguf, borrowQuantized, prepareQuantized, useMetal, logw, outOfCore...)
	}
	if config.Arch == "gemma4" {
		if err := validateGemma4DenseLayout(config, indexTensors(gguf)); err != nil {
			return config, Gemma4Weights{}, err
		}
	}
	config, std, err := LoadModel(data, gguf, borrowQuantized, prepareQuantized, useMetal, logw, outOfCore...)
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

// validateGemma4DenseLayout rejects Gemma 4 variants that look like a normal
// Gemma model in metadata but use mechanisms this runtime does not implement
// (PLE, per-layer dimensions, cross-layer KV sharing, or MoE).  Without this
// guard, an absent global feed_forward_length could reach the decode path and
// produce a slice-bounds panic instead of a useful load-time diagnostic.
func validateGemma4DenseLayout(config Config, tensors map[string]TensorInfo) error {
	if config.Dim <= 0 || config.HiddenDim <= 0 {
		return fmt.Errorf("gemma4 GGUF uses unsupported per-layer dimensions (embedding_length=%d, feed_forward_length=%d); Gemma 4 PLE/MoE layouts are not implemented", config.Dim, config.HiddenDim)
	}
	for l := range config.NLayers {
		prefix := fmt.Sprintf("blk.%d.", l)
		required := []string{
			prefix + "attn_norm.weight", prefix + "attn_output.weight",
			prefix + "ffn_norm.weight", prefix + "ffn_down.weight",
		}
		for _, name := range required {
			if _, ok := tensors[name]; !ok {
				return fmt.Errorf("gemma4 GGUF uses an unsupported layer layout (missing %s); Gemma 4 p-RoPE/PLE/MoE is not implemented", name)
			}
		}
		if _, fused := tensors[prefix+"attn_qkv.weight"]; !fused {
			for _, name := range []string{prefix + "attn_q.weight", prefix + "attn_k.weight", prefix + "attn_v.weight"} {
				if _, ok := tensors[name]; !ok {
					return fmt.Errorf("gemma4 GGUF uses an unsupported attention layout (missing %s); Gemma 4 p-RoPE/PLE/MoE is not implemented", name)
				}
			}
		}
		if _, split := tensors[prefix+"ffn_gate.weight"]; split {
			if _, ok := tensors[prefix+"ffn_up.weight"]; !ok {
				return fmt.Errorf("gemma4 GGUF uses an unsupported FFN layout (missing %s); Gemma 4 PLE/MoE is not implemented", prefix+"ffn_up.weight")
			}
		} else if _, fused := tensors[prefix+"ffn_up.weight"]; !fused {
			return fmt.Errorf("gemma4 GGUF uses an unsupported FFN layout (missing %s); Gemma 4 PLE/MoE is not implemented", prefix+"ffn_gate.weight")
		}
	}
	return nil
}

func loadLayer(data []byte, dataOffset, l int, config Config, tensors map[string]TensorInfo, inferred map[string]int, borrow, prepareQuantized, useMetal, lazyScalarWeights bool, qRows, kRows, vRows int) (LayerWeights, error) {
	prefix := fmt.Sprintf("blk.%d.", l)
	var attnNorm []float32
	var err error
	if config.usesPostNormOnly() {
		attnNorm = loadOptionalF32VecNil(data, dataOffset, prefix+"attn_norm.weight", tensors, inferred)
	} else {
		attnNorm, err = loadF32Vec(data, dataOffset, prefix+"attn_norm.weight", tensors, inferred)
		if err != nil {
			return LayerWeights{}, err
		}
	}
	var wq, wk, wv, wqkv Weight
	hasQKV := false
	if _, ok := tensors[prefix+"attn_qkv.weight"]; ok {
		wqkv, err = loadWeight(data, dataOffset, prefix+"attn_qkv.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return LayerWeights{}, err
		}
		hasQKV = true
	} else {
		wq, err = loadWeight(data, dataOffset, prefix+"attn_q.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return LayerWeights{}, err
		}
		wk, err = loadWeight(data, dataOffset, prefix+"attn_k.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return LayerWeights{}, err
		}
		wv, err = loadWeight(data, dataOffset, prefix+"attn_v.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return LayerWeights{}, err
		}
	}
	wo, err := loadWeight(data, dataOffset, prefix+"attn_output.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
	if err != nil {
		return LayerWeights{}, err
	}
	var ffnNorm []float32
	if config.usesPostNormOnly() || config.sharesParallelBranchNorm() {
		ffnNorm = loadOptionalF32VecNil(data, dataOffset, prefix+"ffn_norm.weight", tensors, inferred)
	} else {
		ffnNorm, err = loadF32Vec(data, dataOffset, prefix+"ffn_norm.weight", tensors, inferred)
		if err != nil {
			return LayerWeights{}, err
		}
	}
	var w1, w2, w3, wGateUp Weight
	hasGateUp := false
	var moe *SparseMoEWeights
	if _, isMoE := tensors[prefix+"ffn_gate_inp.weight"]; isMoE {
		moe, err = loadSparseMoEWeights(data, dataOffset, prefix, config, tensors, inferred, borrow, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return LayerWeights{}, fmt.Errorf("layer %d MoE: %w", l, err)
		}
	} else {
		w2, err = loadWeight(data, dataOffset, prefix+"ffn_down.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return LayerWeights{}, err
		}
		if config.Arch == "phi2" {
			w3, err = loadWeight(data, dataOffset, prefix+"ffn_up.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return LayerWeights{}, err
			}
		} else if _, ok := tensors[prefix+"ffn_gate.weight"]; ok {
			w1, err = loadWeight(data, dataOffset, prefix+"ffn_gate.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return LayerWeights{}, err
			}
			w3, err = loadWeight(data, dataOffset, prefix+"ffn_up.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return LayerWeights{}, err
			}
		} else {
			wGateUp, err = loadWeight(data, dataOffset, prefix+"ffn_up.weight", tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return LayerWeights{}, err
			}
			hasGateUp = true
		}
	}
	attnSinks, err := loadOptionalMoEVec(data, dataOffset, prefix+"attn_sinks.weight", tensors, inferred, config.NHeads)
	if err != nil {
		return LayerWeights{}, err
	}
	attnQNorm := loadOptionalF32VecNil(data, dataOffset, prefix+"attn_q_norm.weight", tensors, inferred)
	attnKNorm := loadOptionalF32VecNil(data, dataOffset, prefix+"attn_k_norm.weight", tensors, inferred)
	postAttnNorm := loadOptionalF32VecNil(data, dataOffset, prefix+"post_attention_norm.weight", tensors, inferred)
	postFFNNorm := loadOptionalF32VecNil(data, dataOffset, prefix+"post_ffw_norm.weight", tensors, inferred)
	bo, err := loadOptionalMoEVec(data, dataOffset, prefix+"attn_output.bias", tensors, inferred, config.Dim)
	if err != nil {
		return LayerWeights{}, err
	}
	if bo == nil {
		bo = make([]float32, config.Dim)
	}
	if (config.Arch == "qwen3" || config.Arch == "qwen3moe") && (len(attnQNorm) != config.HeadDim || len(attnKNorm) != config.HeadDim) {
		return LayerWeights{}, fmt.Errorf("%s requires %d-element attn_q_norm and attn_k_norm tensors", config.Arch, config.HeadDim)
	}
	if config.Arch == "exaone4" {
		if len(attnQNorm) != config.HeadDim || len(attnKNorm) != config.HeadDim {
			return LayerWeights{}, fmt.Errorf("exaone4 requires %d-element attn_q_norm and attn_k_norm tensors", config.HeadDim)
		}
		if len(postAttnNorm) != config.Dim || len(postFFNNorm) != config.Dim {
			return LayerWeights{}, fmt.Errorf("exaone4 requires %d-element post_attention_norm and post_ffw_norm tensors", config.Dim)
		}
	}
	if config.Arch == "olmo2" {
		if len(attnQNorm) != qRows || len(attnKNorm) != kRows {
			return LayerWeights{}, fmt.Errorf("olmo2 requires %d-element attn_q_norm and %d-element attn_k_norm tensors", qRows, kRows)
		}
		if len(postAttnNorm) != config.Dim || len(postFFNNorm) != config.Dim {
			return LayerWeights{}, fmt.Errorf("olmo2 requires %d-element post_attention_norm and post_ffw_norm tensors", config.Dim)
		}
	}
	var ffnUpBias, ffnDownBias []float32
	if config.Arch == "phi2" {
		attnNormBias := loadOptionalF32VecNil(data, dataOffset, prefix+"attn_norm.bias", tensors, inferred)
		ffnUpBias = loadOptionalF32VecNil(data, dataOffset, prefix+"ffn_up.bias", tensors, inferred)
		ffnDownBias = loadOptionalF32VecNil(data, dataOffset, prefix+"ffn_down.bias", tensors, inferred)
		if len(attnNormBias) != config.Dim {
			return LayerWeights{}, fmt.Errorf("phi2 requires %d-element %sattn_norm.bias", config.Dim, prefix)
		}
		if _, ok := tensors[prefix+"attn_output.bias"]; !ok || len(bo) != config.Dim {
			return LayerWeights{}, fmt.Errorf("phi2 requires %d-element %sattn_output.bias", config.Dim, prefix)
		}
		if len(ffnUpBias) != config.HiddenDim || len(ffnDownBias) != config.Dim {
			return LayerWeights{}, fmt.Errorf("phi2 requires %d-element ffn_up.bias and %d-element ffn_down.bias", config.HiddenDim, config.Dim)
		}
	}
	if config.Arch == "gpt-oss" {
		if len(attnSinks) != config.NHeads {
			return LayerWeights{}, fmt.Errorf("gpt-oss requires %d-element attn_sinks.weight", config.NHeads)
		}
		if _, ok := tensors[prefix+"attn_output.bias"]; !ok || len(bo) != config.Dim {
			return LayerWeights{}, fmt.Errorf("gpt-oss requires %d-element attn_output.bias", config.Dim)
		}
	}
	return LayerWeights{
		AttnNorm:     attnNorm,
		AttnNormBias: loadOptionalF32Vec(data, dataOffset, prefix+"attn_norm.bias", tensors, inferred, len(attnNorm)),
		WQ:           wq,
		BQ:           loadOptionalF32Vec(data, dataOffset, prefix+"attn_q.bias", tensors, inferred, qRows),
		WK:           wk,
		BK:           loadOptionalF32Vec(data, dataOffset, prefix+"attn_k.bias", tensors, inferred, kRows),
		WV:           wv,
		BV:           loadOptionalF32Vec(data, dataOffset, prefix+"attn_v.bias", tensors, inferred, vRows),
		WQKV:         wqkv,
		HasQKV:       hasQKV,
		WO:           wo,
		BO:           bo,
		FFNNorm:      ffnNorm,
		FFNNormBias:  loadOptionalF32Vec(data, dataOffset, prefix+"ffn_norm.bias", tensors, inferred, len(ffnNorm)),
		W1:           w1,
		W2:           w2,
		W3:           w3,
		FFNUpBias:    ffnUpBias,
		FFNDownBias:  ffnDownBias,
		WGateUp:      wGateUp,
		HasGateUp:    hasGateUp,
		MoE:          moe,
		// Unlike the biases above (where a zero-filled default is inert),
		// these must stay nil when absent: applying an all-zero norm would
		// zero the activations.
		AttnQNorm:    attnQNorm,
		AttnKNorm:    attnKNorm,
		PostAttnNorm: postAttnNorm,
		PostFFNNorm:  postFFNNorm,
		AttnSinks:    attnSinks,
	}, nil
}

// loadOptionalF32VecNil loads a float vector tensor, returning nil (not a
// zero-filled slice) when the tensor does not exist or fails to load.
func loadOptionalF32VecNil(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int) []float32 {
	if _, ok := tensors[name]; !ok {
		return nil
	}
	v, err := loadF32Vec(data, dataOffset, name, tensors, inferred)
	if err != nil {
		return nil
	}
	return v
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
// normally converted to owned float32s; with lazyScalars enabled for a borrowed
// mmap it remains in its packed scalar representation. Supported quantized
// types stay in their packed byte form, borrowed from data when borrow is set
// or copied otherwise.
// forceF32 additionally dequantizes Q8_0/Q4_0 at load (used for norm vectors
// that must be plain floats).
func loadWeight(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, forceF32, borrow, prepareQuantized, useMetal bool, lazyScalars ...bool) (Weight, error) {
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
		if info.DType == GGMLTypeF32 || info.DType == GGMLTypeF16 || info.DType == GGMLTypeBF16 {
			return Weight{}, fmt.Errorf("tensor %s exceeds file length", name)
		}
		padded := make([]byte, byteSize)
		copy(padded, raw)
		raw = padded
		borrow = false
	}
	keepRawScalars := len(lazyScalars) > 0 && lazyScalars[0] && borrow && !forceF32
	rows, cols := 1, numel
	if len(info.Dims) >= 2 {
		rows = int(info.Dims[1])
		cols = int(info.Dims[0])
	}
	switch info.DType {
	case GGMLTypeF32:
		if keepRawScalars {
			return Weight{Raw: raw, Type: info.DType, Rows: rows, Cols: cols}, nil
		}
		f := make([]float32, numel)
		for i := range numel {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return Weight{F32: f}, nil
	case GGMLTypeF16:
		if keepRawScalars {
			return Weight{Raw: raw, Type: info.DType, Rows: rows, Cols: cols}, nil
		}
		f := make([]float32, numel)
		for i := range numel {
			f[i] = F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return Weight{F32: f}, nil
	case GGMLTypeF64:
		if keepRawScalars {
			return Weight{Raw: raw, Type: info.DType, Rows: rows, Cols: cols}, nil
		}
		f := make([]float32, numel)
		for i := range numel {
			f[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return Weight{F32: f}, nil
	case GGMLTypeBF16:
		if keepRawScalars {
			return Weight{Raw: raw, Type: info.DType, Rows: rows, Cols: cols}, nil
		}
		// bfloat16 is the top 16 bits of an IEEE float32 (QAT checkpoints and
		// many recent full-precision GGUFs use it), so conversion is a shift.
		f := make([]float32, numel)
		for i := range numel {
			f[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return Weight{F32: f}, nil
	case GGMLTypeQ8_0, GGMLTypeQ4_0, GGMLTypeIQ4_NL, GGMLTypeIQ2_S, GGMLTypeIQ3_S, GGMLTypeIQ4_XS, GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1, GGMLTypeQ8_1, GGMLTypeQ8_K,
		GGMLTypeQ2_K, GGMLTypeQ3_K, GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeTQ1_0, GGMLTypeTQ2_0,
		GGMLTypeMXFP4, GGMLTypeQ1_0, GGMLTypeQ2_0:
		if forceF32 {
			f, ok := dequantTensor(info.DType, raw, numel)
			if !ok {
				return Weight{}, fmt.Errorf("%s force_f32 dequantization not implemented for %s", info.DType, name)
			}
			return Weight{F32: f}, nil
		}
		if !borrow {
			owned := make([]byte, len(raw))
			copy(owned, raw)
			raw = owned
		}
		w := Weight{Raw: raw, Type: info.DType, Rows: rows, Cols: cols}
		if useMetal {
			w.Metal = prepareMetalWeight(raw, info.DType, rows, cols, borrow)
		}
		// Direct Metal weights skip redundant prepared data. Small Q/K/V handles
		// retain prepared CPU data so fused-attention dispatch can roll back
		// without changing results if a GPU command fails.
		if !metalWeightUsesDirect(w.Metal) && prepareQuantized && (info.DType == GGMLTypeQ4_K || info.DType == GGMLTypeQ6_K) {
			w.Prepared = PrepareQuantizedWeight(raw, info.DType, rows, cols)
		}
		return w, nil
	default:
		return Weight{}, fmt.Errorf("unsupported tensor type for %s: %s", name, info.DType)
	}
}

// dequantTensor fully dequantizes a contiguous quantized tensor (treated as
// one long row, which is valid because rows are stored back to back with no
// padding). Used by the forceF32 load path for norm vectors and rope factors.
func dequantTensor(t GGMLType, raw []byte, numel int) ([]float32, bool) {
	switch t {
	case GGMLTypeQ8_0:
		return DequantRowQ8_0(raw, numel), true
	case GGMLTypeQ4_0:
		return DequantRowQ4_0(raw, numel), true
	case GGMLTypeIQ4_NL:
		return DequantRowIQ4NL(raw, numel), true
	case GGMLTypeIQ2_S:
		return DequantRowIQ2S(raw, numel), true
	case GGMLTypeIQ3_S:
		return DequantRowIQ3S(raw, numel), true
	case GGMLTypeIQ4_XS:
		return DequantRowIQ4XS(raw, numel), true
	case GGMLTypeQ4_1:
		return DequantRowQ4_1(raw, numel), true
	case GGMLTypeQ5_0:
		return DequantRowQ5_0(raw, numel), true
	case GGMLTypeQ5_1:
		return DequantRowQ5_1(raw, numel), true
	case GGMLTypeQ8_1:
		return DequantRowQ8_1(raw, numel), true
	case GGMLTypeQ8_K:
		return DequantRowQ8K(raw, numel), true
	case GGMLTypeQ2_K:
		return DequantRowQ2K(raw, numel), true
	case GGMLTypeQ3_K:
		return DequantRowQ3K(raw, numel), true
	case GGMLTypeQ4_K:
		return DequantRowQ4K(raw, numel), true
	case GGMLTypeQ5_K:
		return DequantRowQ5K(raw, numel), true
	case GGMLTypeQ6_K:
		return DequantRowQ6K(raw, numel), true
	case GGMLTypeMXFP4:
		return DequantRowMXFP4(raw, numel), true
	case GGMLTypeTQ1_0:
		return DequantRowTQ1_0(raw, numel), true
	case GGMLTypeTQ2_0:
		return DequantRowTQ2_0(raw, numel), true
	case GGMLTypeQ1_0:
		return DequantRowQ1_0(raw, numel), true
	case GGMLTypeQ2_0:
		return DequantRowQ2_0(raw, numel), true
	default:
		return nil, false
	}
}

func loadF32Vec(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int) ([]float32, error) {
	w, err := loadWeight(data, dataOffset, name, tensors, inferred, true, false, false, false)
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
	ProjectLogitsInto(config, weights, buf, logits)
}

func ProjectLogitsInto(config Config, weights ModelWeights, buf *DecodeBuffer, logits *[]float32) {
	weights.Output.MatvecInto(buf.XN, logits)
	addInPlace(*logits, weights.OutputBias)
	if config.LogitScale != 1 {
		ScaleF32(*logits, 1/config.LogitScale)
	}
	if config.FinalLogitSoftcap > 0 {
		softcapF32(*logits, config.FinalLogitSoftcap)
	}
}

func ArgmaxOutputToken(config Config, weights ModelWeights, buf *DecodeBuffer) (uint32, bool) {
	return argmaxOutputTokenInto(config, weights, buf, nil)
}

func argmaxOutputTokenInto(config Config, weights ModelWeights, buf *DecodeBuffer, logits *[]float32) (uint32, bool) {
	return argmaxOutputTokenPenalizedInto(config, weights, buf, nil, 1, logits)
}

func argmaxOutputTokenPenalizedInto(config Config, weights ModelWeights, buf *DecodeBuffer, recent []uint32, repeatPenalty float32, logits *[]float32) (uint32, bool) {
	if !finite32(config.LogitScale) || config.LogitScale <= 0 || config.FinalLogitSoftcap < 0 ||
		!finite32(repeatPenalty) || repeatPenalty <= 0 {
		return 0, false
	}
	// Positive linear logit scaling commutes with repeat penalty, but adding a
	// bias or applying the nonlinear softcap does not. Materialize those rare
	// combinations so penalty ordering stays exactly the sampler's ordering.
	if len(weights.OutputBias) > 0 || (repeatPenalty != 1 && config.FinalLogitSoftcap > 0) {
		ProjectLogitsInto(config, weights, buf, &buf.Logits)
		applyRepeatPenalty(buf.Logits, recent, repeatPenalty)
		if logits != nil {
			ensureLenNoClear(logits, len(buf.Logits))
			copy(*logits, buf.Logits)
		}
		return argmaxFiniteToken(buf.Logits), true
	}
	// Greedy decode only needs the winning token. Positive logit scaling and the
	// optional positive softcap preserve argmax ordering, so Metal can reduce its
	// Q6_K output buffer on-device and avoid a 131k-logit readback plus CPU scan.
	// Sampling and unsupported output types retain the materialized fallback.
	if token, ok := argmaxMetalQ6KPenalized(weights.Output.Metal, buf.XN, recent, repeatPenalty); ok {
		return token, true
	}
	if weights.Output.Metal != nil && logits != nil {
		ProjectLogitsInto(config, weights, buf, logits)
		if len(*logits) == 0 {
			return 0, false
		}
		applyRepeatPenalty(*logits, recent, repeatPenalty)
		return argmaxFiniteToken(*logits), true
	}
	if repeatPenalty != 1 {
		return 0, false
	}
	return weights.Output.ArgmaxMatvec(buf.XN)
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
	if config.UsesMLA {
		ForwardDeepSeek2BodyInto(config, weights, cache, buf, token, pos)
		return
	}
	dim := config.Dim
	headDim := config.HeadDim
	kvMul := max(1, config.KVMul)
	ropeInvFreq, ropeMscale := buf.RopeInvFreq, buf.RopeMscale
	if config.Arch == "gpt-oss" && len(buf.RopeGptOssInvFreq) > 0 {
		// GPT-OSS uses its own YaRN concentration rule, precomputed alongside
		// the regular table in NewDecodeBuffer.
		ropeInvFreq, ropeMscale = buf.RopeGptOssInvFreq, buf.RopeGptOssConcentration
	}
	ropeHalf, ropePairs := prepareRopeScratch(pos, headDim, config.RopeDimensionCount, ropeInvFreq, ropeMscale, &buf.RopeSin, &buf.RopeCos)
	swaRopeHalf, swaRopePairs := 0, 0
	if len(buf.RopeSWAInvFreq) > 0 {
		swaRopeHalf, swaRopePairs = prepareRopeScratch(pos, headDim, config.RopeDimensionCount, buf.RopeSWAInvFreq, buf.RopeSWAMscale, &buf.RopeSWASin, &buf.RopeSWACos)
	}
	ropeIsInterleaved := ropeInterleaved(config.Arch)
	weights.TokenEmbd.RowInto(int(token), dim, &buf.X)
	if config.EmbeddingScale != 1 {
		ScaleF32(buf.X[:dim], config.EmbeddingScale)
	}
	for l := range config.NLayers {
		layer := weights.Layers[l]
		if config.usesPostNormOnly() {
			ensureLenNoClear(&buf.XN, dim)
			copy(buf.XN, buf.X[:dim])
		} else {
			normalizeDecoderInto(config, buf.X, layer.AttnNorm, layer.AttnNormBias, &buf.XN)
		}
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
			tryMatvecAttentionInto(layer.WQ, layer.WK, layer.WV, buf.XN, &buf.Q4KXSums, &buf.Q, &buf.K, &buf.V)
			addInPlace(buf.Q, layer.BQ)
			addInPlace(buf.K, layer.BK)
			addInPlace(buf.V, layer.BV)
		}
		// Normalize projected Q/K before RoPE. Qwen3/EXAONE 4 use one RMSNorm
		// per head; OLMo 2/3 normalize each complete projection.
		normalizeProjectedQKInPlace(config, layer, buf.Q, buf.K)
		if config.layerUsesRoPE(l) {
			activeHalf, activePairs := ropeHalf, ropePairs
			activeSin, activeCos := buf.RopeSin, buf.RopeCos
			if config.layerUsesSWA(l) && swaRopePairs > 0 {
				activeHalf, activePairs = swaRopeHalf, swaRopePairs
				activeSin, activeCos = buf.RopeSWASin, buf.RopeSWACos
			}
			applyPreparedRope(buf.Q, headDim, config.NHeads, activeHalf, activePairs, activeSin, activeCos, ropeIsInterleaved)
			applyPreparedRope(buf.K, headDim, config.NKVHeads, activeHalf, activePairs, activeSin, activeCos, ropeIsInterleaved)
		}
		if temperature := attentionTemperatureAt(config, pos); temperature != 1 {
			ScaleF32(buf.Q, temperature)
		}

		cache.storeKV(l, pos, buf.K, buf.V)

		clear(buf.AttnOut)
		scale := config.AttentionScale
		if scale == 0 {
			scale = float32(1 / math.Sqrt(float64(headDim)))
		}
		if config.Arch == "phi2" {
			// Phi-2 scales Q before the dot product and uses a unit attention
			// scale to keep intermediate scores in the trained precision range.
			ScaleF32(buf.Q, scale)
			scale = 1
		}
		attnStart := 0
		if config.layerUsesSWA(l) {
			attnStart = max(0, pos-config.SlidingWindow)
		}
		// GQA/MQA query heads share K/V rows. Process each complete group together
		// so the shared cacheline is fetched once, then parallelize across KV
		// groups at contexts long enough to amortize dispatch.
		groupedGQA := useGroupedGQAAttention && !cache.F16 && kvMul == 4 && config.NKVHeads > 0 && len(layer.AttnSinks) == 0
		if attnLen := pos - attnStart + 1; groupedGQA && attnLen >= groupedGQADecodeMinContext && config.NKVHeads > 1 {
			parallelAttendHeadGroups(config, cache, buf, l, pos, attnStart, scale, kvMul)
		} else if groupedGQA && attnLen < 128 {
			attendHeadGroupsRange(&config, cache, buf, l, pos, attnStart, scale, kvMul, 0, config.NKVHeads)
		} else if attnLen >= 128 && config.NHeads > 1 {
			parallelAttendHeads(config, layer, cache, buf, l, pos, attnStart, scale, kvMul)
		} else {
			attendHeadsRange(&config, &layer, cache, buf, l, pos, attnStart, scale, kvMul, 0, config.NHeads)
		}
		layer.WO.MatvecInto(buf.AttnOut, &buf.Proj)
		addInPlace(buf.Proj, layer.BO)
		if layer.PostAttnNorm != nil {
			rmsNormInto(buf.Proj, layer.PostAttnNorm, config.RMSNormEps, &buf.Proj)
		}
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		if config.ParallelResidual {
			copy(buf.AttnProj[:dim], buf.Proj[:dim])
		} else {
			addInPlace(buf.X[:dim], buf.Proj)
		}

		if config.sharesParallelBranchNorm() {
			ensureLenNoClear(&buf.XN2, dim)
			copy(buf.XN2, buf.XN[:dim])
		} else if config.usesPostNormOnly() {
			ensureLenNoClear(&buf.XN2, dim)
			copy(buf.XN2, buf.X[:dim])
		} else {
			normalizeDecoderInto(config, buf.X, layer.FFNNorm, layer.FFNNormBias, &buf.XN2)
		}
		if layer.MoE != nil {
			sparseMoEForward(layer.MoE, buf.XN2, buf)
		} else {
			// Decode bottleneck: the selective Metal path previously synchronized and
			// copied Gate/Up to the CPU for SiLU, then copied Hidden back for Down.
			// Keep all three stages in one command buffer when the measured
			// Q4_K/Q4_K/Q6_K shape matches. Any unsupported shape or GPU failure falls
			// through to the unchanged CPU/GPU path; removing this branch is rollback.
			fusedMetalFFN := config.Arch != "phi2" && !layer.HasGateUp && !config.UseGELU &&
				matvecMetalSwiGLUInto(layer.W1.Metal, layer.W3.Metal, layer.W2.Metal, buf.XN2, &buf.Proj)
			if !fusedMetalFFN {
				if config.Arch == "phi2" {
					layer.W3.MatvecInto(buf.XN2, &buf.Up)
					addInPlace(buf.Up, layer.FFNUpBias)
					ensureLenNoClear(&buf.Hidden, config.HiddenDim)
					for i := range config.HiddenDim {
						buf.Hidden[i] = geluExact(buf.Up[i])
					}
					layer.W2.MatvecInto(buf.Hidden, &buf.Proj)
					addInPlace(buf.Proj, layer.FFNDownBias)
				} else {
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
						if config.UseGELU {
							geluMulF32(gate[:hDim], up[:hDim], hidden[:hDim])
						} else {
							siluMulF32(gate[:hDim], up[:hDim], hidden[:hDim])
						}
					}
					layer.W2.MatvecInto(buf.Hidden, &buf.Proj)
					addInPlace(buf.Proj, layer.FFNDownBias)
				}
			}
		}
		if layer.PostFFNNorm != nil {
			rmsNormInto(buf.Proj, layer.PostFFNNorm, config.RMSNormEps, &buf.Proj)
		}
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		addInPlace(buf.X[:dim], buf.Proj)
		if config.ParallelResidual {
			addInPlace(buf.X[:dim], buf.AttnProj[:dim])
		}
	}
	normalizeDecoderInto(config, buf.X, weights.OutputNorm, weights.OutputNormBias, &buf.XN)
}

// attendHeadsRange lives outside ForwardBodyInto so the common short-context
// serial path does not construct an escaping closure for every layer. The
// parallel wrapper owns the copies captured by its worker closure.
func attendHeadsRange(config *Config, layer *LayerWeights, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul, hStart, hEnd int) {
	headDim := config.HeadDim
	valueDim := config.ValueDim
	for h := hStart; h < hEnd; h++ {
		kvH := h / kvMul
		qOff := h * headDim
		outOff := h * valueDim
		sink, hasSink := float32(0), h < len(layer.AttnSinks)
		if hasSink {
			sink = layer.AttnSinks[h]
		}
		cache.attendHeadWithSink(l, kvH, buf.Q[qOff:qOff+headDim], headDim, valueDim,
			attnStart, pos, scale, config.AttnLogitSoftcap, sink, hasSink,
			buf.AttnOut[outOff:outOff+valueDim])
	}
}

func parallelAttendHeads(config Config, layer LayerWeights, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul int) {
	parallelChunks(config.NHeads, func(hStart, hEnd int) {
		attendHeadsRange(&config, &layer, cache, buf, l, pos, attnStart, scale, kvMul, hStart, hEnd)
	})
}

func attendHeadGroupsRange(config *Config, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul, kvStart, kvEnd int) {
	headDim := config.HeadDim
	valueDim := config.ValueDim
	for kvH := kvStart; kvH < kvEnd; kvH++ {
		hStart := kvH * kvMul
		hEnd := min(hStart+kvMul, config.NHeads)
		if hStart >= hEnd {
			break
		}
		cache.attendHeadGroup(l, kvH,
			buf.Q[hStart*headDim:hEnd*headDim], hEnd-hStart, headDim, valueDim,
			attnStart, pos, scale, config.AttnLogitSoftcap,
			buf.AttnOut[hStart*valueDim:hEnd*valueDim])
	}
}

func parallelAttendHeadGroups(config Config, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul int) {
	// Keep the configured worker set awake for the projection matvec that
	// immediately follows attention. GQA often exposes only eight groups on a
	// 12-core Apple SoC; dispatching exactly eight jobs made four workers sleep
	// and added a repeated wake-up penalty at every layer boundary.
	workItems := max(config.NKVHeads, min(numThreads(), config.NHeads))
	parallelChunks(workItems, func(kvStart, kvEnd int) {
		if kvStart >= config.NKVHeads {
			return
		}
		attendHeadGroupsRange(&config, cache, buf, l, pos, attnStart, scale, kvMul, kvStart, min(kvEnd, config.NKVHeads))
	})
}

func normalizeDecoderInto(config Config, x, weight, bias []float32, out *[]float32) {
	if config.UseLayerNorm {
		layerNormInto(x, weight, bias, config.RMSNormEps, out)
		return
	}
	rmsNormInto(x, weight, config.RMSNormEps, out)
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
	if weights.Native {
		forwardNativeGemma4BodyInto(config, weights, cache, buf, token, pos)
		projectNativeGemma4Logits(config, weights, buf, logits)
		return
	}
	ForwardInto(config, weights.Standard, cache, buf, token, pos, logits)
}

func ForwardHiddenGemma4(config Config, weights Gemma4Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	if weights.Native {
		forwardNativeGemma4BodyInto(config, weights, cache, buf, token, pos)
		out := make([]float32, len(buf.XN))
		copy(out, buf.XN)
		return out
	}
	ForwardBodyInto(config, weights.Standard, cache, buf, token, pos)
	out := make([]float32, len(buf.XN))
	copy(out, buf.XN)
	return out
}

// geluTanh is the tanh-approximated GELU (gelu_pytorch_tanh) used by the
// Gemma family's FFN in place of SiLU.
func geluTanh(x float32) float32 { return geluTanhScalar(x) }

// softcapF32 applies v = cap*tanh(v/cap) elementwise — Gemma's logit
// softcapping, which bounds values to (-cap, cap) while staying smooth.
// Gemma applies this to the whole logits vector, so on a 256k-entry vocabulary
// it is a quarter-million tanh calls per token — by far the heaviest consumer of
// tanh in the engine, and the reason it uses the float32 one.
func softcapF32(v []float32, cap float32) {
	inv := 1 / cap
	for i, x := range v {
		v[i] = cap * fastTanhF32(x*inv)
	}
}

// perHeadRMSNormInPlace RMS-normalizes each head's headDim-wide slice of vec
// independently against a shared headDim-length weight — Gemma 3/4-style
// QK-norm, applied to the projected Q/K before RoPE.
func perHeadRMSNormInPlace(vec []float32, headDim, nHeads int, weight []float32, eps float32) {
	if len(weight) < headDim {
		return
	}
	for h := 0; h < nHeads; h++ {
		off := h * headDim
		if off+headDim > len(vec) {
			break
		}
		sub := vec[off : off+headDim]
		ss := DotF32(sub, sub)
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+eps)))
		mulScaleF32(sub, weight[:headDim], scale, sub)
	}
}

func normalizeProjectedQKInPlace(config Config, layer LayerWeights, q, k []float32) {
	if config.usesFullProjectionQKNorm() {
		if layer.AttnQNorm != nil {
			rmsNormInto(q, layer.AttnQNorm, config.RMSNormEps, &q)
		}
		if layer.AttnKNorm != nil {
			rmsNormInto(k, layer.AttnKNorm, config.RMSNormEps, &k)
		}
		return
	}
	if layer.AttnQNorm != nil {
		perHeadRMSNormInPlace(q, config.HeadDim, config.NHeads, layer.AttnQNorm, config.RMSNormEps)
	}
	if layer.AttnKNorm != nil {
		perHeadRMSNormInPlace(k, config.HeadDim, config.NKVHeads, layer.AttnKNorm, config.RMSNormEps)
	}
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
	case "llama", "llama2", "llama3", "mistral", "mistral3", "mixtral", "ministral", "smollm3", "internlm2":
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
		if wq.Prepared != nil && wk.Prepared != nil && wv.Prepared != nil {
			if MatvecPreparedQ4K3IntoWithXSums(
				wq.Raw, wq.Prepared, wq.Rows, wq.Cols,
				wk.Raw, wk.Prepared, wk.Rows, wk.Cols,
				wv.Raw, wv.Prepared, wv.Rows, wv.Cols,
				x,
				q4kXSums,
				q,
				k,
				v,
			) {
				return true
			}
		}
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
		return matvecSameType3Into(wq, wk, wv, x, q, k, v)
	}
}

// tryMatvecAttentionInto keeps the attention projection fast for mixed-quant
// GGUFs. Bottleneck: Ministral-3-3B-Q4_K_M stores Q/K as Q4_K but V as Q6_K,
// so the previous all-or-nothing QKV fusion missed the reusable Q4_K xsums and
// worker dispatch for Q+K, and still dispatched V separately. Change: try full
// QKV fusion first, then the common mixed Q4_K/Q4_K/Q6_K one-dispatch path,
// then pairwise fusion for other same-typed projection pairs. Expected effect:
// lower decode latency on mixed Q4_K/Q6_K attention blocks. Risk: small extra
// branch cost. Rollback: replace this call with the former three independent
// MatvecInto calls.
func tryMatvecAttentionInto(wq, wk, wv Weight, x []float32, q4kXSums *[]float32, q, k, v *[]float32) {
	if matvecMetalQ4K2Q6KInto(wq.Metal, wk.Metal, wv.Metal, x, wq.Rows, wk.Rows, wv.Rows, wq.Cols, q, k, v) {
		return
	}
	if tryMatvec3Into(wq, wk, wv, x, q4kXSums, q, k, v) {
		return
	}
	if wq.F32 == nil && wk.F32 == nil && wv.F32 == nil &&
		wq.Type == GGMLTypeQ4_K && wk.Type == GGMLTypeQ4_K && wv.Type == GGMLTypeQ6_K &&
		MatvecQ4K2Q6KIntoWithXSums(wq.Raw, wq.Rows, wq.Cols, wk.Raw, wk.Rows, wk.Cols, wv.Raw, wv.Rows, wv.Cols, x, q4kXSums, q, k, v) {
		return
	}
	if tryMatvec2Into(wq, wk, x, q4kXSums, q, k) {
		wv.MatvecInto(x, v)
		return
	}
	if tryMatvec2Into(wq, wv, x, q4kXSums, q, v) {
		wk.MatvecInto(x, k)
		return
	}
	if tryMatvec2Into(wk, wv, x, q4kXSums, k, v) {
		wq.MatvecInto(x, q)
		return
	}
	wq.MatvecInto(x, q)
	wk.MatvecInto(x, k)
	wv.MatvecInto(x, v)
}

func tryMatvec2Into(a, b Weight, x []float32, q4kXSums *[]float32, aOut, bOut *[]float32) bool {
	if a.Type != b.Type || a.Cols != b.Cols || a.Cols != len(x) || a.F32 != nil || b.F32 != nil {
		return false
	}
	switch a.Type {
	case GGMLTypeQ4_K:
		if matvecMetalQ4K2Into(a.Metal, b.Metal, x, a.Rows, b.Rows, a.Cols, aOut, bOut) {
			return true
		}
		if a.Prepared != nil && b.Prepared != nil {
			if MatvecPreparedQ4K2IntoWithXSums(a.Raw, a.Prepared, a.Rows, a.Cols, b.Raw, b.Prepared, b.Rows, b.Cols, x, q4kXSums, aOut, bOut) {
				return true
			}
		}
		return MatvecQ4K2IntoWithXSums(a.Raw, a.Rows, a.Cols, b.Raw, b.Rows, b.Cols, x, q4kXSums, aOut, bOut)
	case GGMLTypeQ6_K:
		return MatvecQ6K2Into(a.Raw, a.Rows, a.Cols, b.Raw, b.Rows, b.Cols, x, aOut, bOut)
	default:
		return matvecSameType2Into(a, b, x, aOut, bOut)
	}
}

// matvecSameType{2,3}Into fuse ordinary same-format projections into one
// worker-pool dispatch. Unlike the Q4_K/Q6_K specializations above they do
// not share activation preprocessing; they avoid only the repeated channel
// dispatch. That is still valuable for Q4_0-heavy StableLM/InternLM GGUFs,
// whose Q/K/V and gate/up projections otherwise launch independently.
func matvecSameType2Into(a, b Weight, x []float32, aOut, bOut *[]float32) bool {
	dot, rowBytes, ok := sameTypeQuantDot(a, b, x)
	if !ok {
		return false
	}
	ensureLenNoClear(aOut, a.Rows)
	ensureLenNoClear(bOut, b.Rows)
	total := a.Rows + b.Rows
	parallelRows(total, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, a.Rows); as < ae {
			matvecDotRows(a.Raw, rowBytes, x, as, ae, *aOut, dot)
		}
		if bs, be := clippedRange(start, end, a.Rows, total); bs < be {
			matvecDotRows(b.Raw, rowBytes, x, bs-a.Rows, be-a.Rows, *bOut, dot)
		}
	})
	return true
}

func matvecSameType3Into(a, b, c Weight, x []float32, aOut, bOut, cOut *[]float32) bool {
	dot, rowBytes, ok := sameTypeQuantDot(a, b, x)
	if !ok || c.F32 != nil || c.Type != a.Type || c.Cols != a.Cols || c.Rows < 0 || len(c.Raw) < c.Rows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, a.Rows)
	ensureLenNoClear(bOut, b.Rows)
	ensureLenNoClear(cOut, c.Rows)
	ab := a.Rows + b.Rows
	total := ab + c.Rows
	parallelRows(total, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, a.Rows); as < ae {
			matvecDotRows(a.Raw, rowBytes, x, as, ae, *aOut, dot)
		}
		if bs, be := clippedRange(start, end, a.Rows, ab); bs < be {
			matvecDotRows(b.Raw, rowBytes, x, bs-a.Rows, be-a.Rows, *bOut, dot)
		}
		if cs, ce := clippedRange(start, end, ab, total); cs < ce {
			matvecDotRows(c.Raw, rowBytes, x, cs-ab, ce-ab, *cOut, dot)
		}
	})
	return true
}

type quantRowDot func(row []byte, x []float32, cols int) float32

func sameTypeQuantDot(a, b Weight, x []float32) (quantRowDot, int, bool) {
	if a.F32 != nil || b.F32 != nil || a.Type != b.Type || a.Cols <= 0 || a.Cols != b.Cols || a.Cols != len(x) || a.Rows < 0 || b.Rows < 0 {
		return nil, 0, false
	}
	rowBytes, ok := a.Type.DataSize(a.Cols)
	if !ok || rowBytes <= 0 || len(a.Raw) < a.Rows*rowBytes || len(b.Raw) < b.Rows*rowBytes {
		return nil, 0, false
	}
	var dot quantRowDot
	switch a.Type {
	case GGMLTypeQ8_0:
		dot = DotQ8_0F32
	case GGMLTypeQ4_0:
		dot = DotQ4_0F32
	case GGMLTypeIQ4_NL:
		dot = DotIQ4NLF32
	case GGMLTypeIQ2_S:
		dot = DotIQ2SF32
	case GGMLTypeIQ3_S:
		dot = DotIQ3SF32
	case GGMLTypeIQ4_XS:
		dot = DotIQ4XSF32
	case GGMLTypeQ4_1:
		dot = DotQ4_1F32
	case GGMLTypeQ5_0:
		dot = DotQ5_0F32
	case GGMLTypeQ5_1:
		dot = DotQ5_1F32
	case GGMLTypeQ8_1:
		dot = DotQ8_1F32
	case GGMLTypeQ8_K:
		dot = DotQ8KF32
	case GGMLTypeQ2_K:
		dot = DotQ2KF32
	case GGMLTypeQ3_K:
		dot = DotQ3KF32
	case GGMLTypeQ4_K:
		dot = DotQ4KF32
	case GGMLTypeQ5_K:
		dot = DotQ5KF32
	case GGMLTypeQ6_K:
		dot = DotQ6KF32
	case GGMLTypeMXFP4:
		dot = DotMXFP4F32
	case GGMLTypeTQ1_0:
		dot = DotTQ1_0F32
	case GGMLTypeTQ2_0:
		dot = DotTQ2_0F32
	case GGMLTypeQ1_0:
		dot = DotQ1_0F32
	case GGMLTypeQ2_0:
		dot = DotQ2_0F32
	default:
		return nil, 0, false
	}
	return dot, rowBytes, true
}

func matvecDotRows(data []byte, rowBytes int, x []float32, start, end int, out []float32, dot quantRowDot) {
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = dot(data[off:off+rowBytes], x, len(x))
	}
}

// attnScoresPool holds per-head score scratch for the two-pass attention
// below. Heads run concurrently (parallelChunks), so each in-flight head
// borrows its own buffer.
var attnScoresPool = sync.Pool{New: func() any { s := make([]float32, 0, 4096); return &s }}

// Kept as a package variable so correctness and end-to-end performance tests
// can A/B the legacy path in one process without changing model state.
var useGroupedGQAAttention = os.Getenv("GOPHERLLM_NO_GROUPED_GQA") == ""

// Below this point, 32-way head scheduling beats the lower data movement of
// eight grouped jobs on the M2 Max. At long context the KV bandwidth saved by
// the NEON x4 kernels dominates. Prefill has independent token-level
// parallelism and therefore uses grouping without this decode-only threshold.
const groupedGQADecodeMinContext = 4096

// onlineAttention computes softmax(q·K/scale)·V for one head over positions
// startT..endT, accumulating into out (which the caller has zeroed).
//
// Two-pass structure: pass 1 computes every score as an independent dot
// product — nothing but loads and FMAs in the dependency chain, so
// out-of-order execution overlaps positions freely; pass 2 takes the exact
// global max, exponentiates (iterations independent, so the exp latency
// pipelines too), and accumulates the weighted V rows. The previous
// single-pass online-softmax rescaled the accumulator inside the loop,
// chaining dot -> exp -> branch -> rescale serially per position; measured
// on the dev laptop the two-pass form is ~1.15x faster at 4k-16k context
// and numerically it uses the true maximum rather than a running one.
func onlineAttention(query, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	onlineAttentionWithSink(query, keys, values, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, 0, false, out)
}

func onlineAttentionWithSink(query, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap, sink float32, hasSink bool, out []float32) {
	span := endT - startT + 1
	if span <= 0 {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	ensureLenNoClear(scratch, span)
	scores := (*scratch)[:span]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if kOff+keyHeadDim > len(keys) {
			break
		}
		scores[n] = DotF32(query, keys[kOff:kOff+keyHeadDim]) * scale
		n++
	}
	weightedVSumWithSink(scores[:n], values, valueStride, valueHeadDim, startT, softcap, sink, hasSink, out)
	attnScoresPool.Put(scratch)
}

// onlineAttentionGroup is the IO-aware GQA/MQA path. Query heads sharing one
// KV head are kept together through both attention passes. The arithmetic for
// each head is unchanged, but K/V rows remain hot across the group instead of
// being streamed independently for every query head.
func onlineAttentionGroup(queries, keys, values []float32, queryHeads, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	if queryHeads == 4 {
		onlineAttentionGroup4(queries, keys, values, keyStride, valueStride,
			keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
		return
	}
	onlineAttentionGroupEither(queries, keys, nil, values, nil, queryHeads,
		keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
}

// onlineAttentionGroup4 is the Ministral-style GQA specialization. Its NEON
// primitives load each shared K/V row once for four query heads.
func onlineAttentionGroup4(queries, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	const queryHeads = 4
	span := endT - startT + 1
	if span <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || len(queries) < queryHeads*keyHeadDim || len(out) < queryHeads*valueHeadDim {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	scoreLen := queryHeads * span
	ensureLenNoClear(scratch, scoreLen+queryHeads)
	scores := (*scratch)[:scoreLen]
	denoms := (*scratch)[scoreLen : scoreLen+queryHeads]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if kOff+keyHeadDim > len(keys) {
			break
		}
		s0, s1, s2, s3 := dotF32x4(
			&queries[0], &queries[keyHeadDim], &queries[2*keyHeadDim], &queries[3*keyHeadDim],
			&keys[kOff], keyHeadDim)
		scores[n] = s0 * scale
		scores[span+n] = s1 * scale
		scores[2*span+n] = s2 * scale
		scores[3*span+n] = s3 * scale
		n++
	}
	if n == 0 {
		attnScoresPool.Put(scratch)
		return
	}
	for h := 0; h < queryHeads; h++ {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}

	out0 := out[:valueHeadDim]
	out1 := out[valueHeadDim : 2*valueHeadDim]
	out2 := out[2*valueHeadDim : 3*valueHeadDim]
	out3 := out[3*valueHeadDim : 4*valueHeadDim]
	for i := 0; i < n; i++ {
		vOff := (startT + i) * valueStride
		if vOff+valueHeadDim > len(values) {
			break
		}
		axpyF32x4(&out0[0], &out1[0], &out2[0], &out3[0],
			scores[i], scores[span+i], scores[2*span+i], scores[3*span+i],
			&values[vOff], valueHeadDim)
	}
	ScaleF32(out0, 1/denoms[0])
	ScaleF32(out1, 1/denoms[1])
	ScaleF32(out2, 1/denoms[2])
	ScaleF32(out3, 1/denoms[3])
	attnScoresPool.Put(scratch)
}

func onlineAttentionGroupF16(queries []float32, keys, values []uint16, queryHeads, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	onlineAttentionGroupEither(queries, nil, keys, nil, values, queryHeads,
		keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
}

func onlineAttentionGroupEither(queries []float32, keys []float32, keys16 []uint16, values []float32, values16 []uint16, queryHeads, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	span := endT - startT + 1
	if span <= 0 || queryHeads <= 0 || len(queries) < queryHeads*keyHeadDim || len(out) < queryHeads*valueHeadDim {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	scoreLen := queryHeads * span
	ensureLenNoClear(scratch, scoreLen+queryHeads)
	scores := (*scratch)[:scoreLen]
	denoms := (*scratch)[scoreLen : scoreLen+queryHeads]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if (keys != nil && kOff+keyHeadDim > len(keys)) || (keys == nil && kOff+keyHeadDim > len(keys16)) {
			break
		}
		for h := 0; h < queryHeads; h++ {
			query := queries[h*keyHeadDim : (h+1)*keyHeadDim]
			if keys != nil {
				scores[h*span+n] = DotF32(query, keys[kOff:kOff+keyHeadDim]) * scale
			} else {
				scores[h*span+n] = dotF32F16(query, keys16[kOff:kOff+keyHeadDim]) * scale
			}
		}
		n++
	}
	if n == 0 {
		attnScoresPool.Put(scratch)
		return
	}

	for h := 0; h < queryHeads; h++ {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}

	for i := 0; i < n; i++ {
		vOff := (startT + i) * valueStride
		if (values != nil && vOff+valueHeadDim > len(values)) || (values == nil && vOff+valueHeadDim > len(values16)) {
			break
		}
		for h := 0; h < queryHeads; h++ {
			weight := scores[h*span+i]
			hout := out[h*valueHeadDim : (h+1)*valueHeadDim]
			if values != nil {
				AxpyF32(hout, weight, values[vOff:vOff+valueHeadDim])
			} else {
				axpyF16(hout, weight, values16[vOff:vOff+valueHeadDim])
			}
		}
	}
	for h := 0; h < queryHeads; h++ {
		if denoms[h] > 0 {
			ScaleF32(out[h*valueHeadDim:(h+1)*valueHeadDim], 1/denoms[h])
		}
	}
	attnScoresPool.Put(scratch)
}

func attentionWeightsInPlace(scores []float32, softcap float32) float32 {
	if softcap > 0 {
		for i, s := range scores {
			scores[i] = softcap * float32(math.Tanh(float64(s/softcap)))
		}
	}
	maxScore := scores[0]
	for _, s := range scores[1:] {
		if s > maxScore {
			maxScore = s
		}
	}
	var denom float32
	for i, s := range scores {
		w := float32(math.Exp(float64(s - maxScore)))
		scores[i] = w
		denom += w
	}
	return denom
}

// weightedVSum finishes attention pass 2 shared by the f32 and f16 K-row
// variants: optional softcap, max-stabilized softmax weights in place, then
// out += sum(w_i * V_row_i) / denom. values16 is used when values is nil.
func weightedVSum(scores []float32, values []float32, valueStride, valueHeadDim, startT int, softcap float32, out []float32) {
	weightedVSumWithSink(scores, values, valueStride, valueHeadDim, startT, softcap, 0, false, out)
}

func weightedVSumWithSink(scores []float32, values []float32, valueStride, valueHeadDim, startT int, softcap, sink float32, hasSink bool, out []float32) {
	weightedVSumEitherWithSink(scores, values, nil, valueStride, valueHeadDim, startT, softcap, sink, hasSink, out)
}

func weightedVSumEither(scores []float32, values []float32, values16 []uint16, valueStride, valueHeadDim, startT int, softcap float32, out []float32) {
	weightedVSumEitherWithSink(scores, values, values16, valueStride, valueHeadDim, startT, softcap, 0, false, out)
}

func weightedVSumEitherWithSink(scores []float32, values []float32, values16 []uint16, valueStride, valueHeadDim, startT int, softcap, sink float32, hasSink bool, out []float32) {
	n := len(scores)
	if n == 0 {
		return
	}
	if softcap > 0 {
		for i, s := range scores {
			scores[i] = softcap * float32(math.Tanh(float64(s/softcap)))
		}
	}
	maxScore := scores[0]
	for _, s := range scores[1:] {
		if s > maxScore {
			maxScore = s
		}
	}
	if hasSink && sink > maxScore {
		maxScore = sink
	}
	var denom float32
	for i, s := range scores {
		w := float32(math.Exp(float64(s - maxScore)))
		scores[i] = w
		denom += w
	}
	if hasSink {
		denom += float32(math.Exp(float64(sink - maxScore)))
	}
	for i := 0; i < n; i++ {
		vOff := (startT + i) * valueStride
		if values != nil {
			if vOff+valueHeadDim > len(values) {
				break
			}
			AxpyF32(out[:valueHeadDim], scores[i], values[vOff:vOff+valueHeadDim])
		} else {
			if vOff+valueHeadDim > len(values16) {
				break
			}
			axpyF16(out[:valueHeadDim], scores[i], values16[vOff:vOff+valueHeadDim])
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
