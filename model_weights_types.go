package gopherllm

func releaseModelMetalWeights(weights *ModelWeights) {
	if weights == nil {
		return
	}
	seen := map[*MetalWeight]bool{}
	seenGPU := map[*GPUWeight]bool{}
	release := func(w *Weight) {
		if w == nil {
			return
		}
		if w.Metal != nil && !seen[w.Metal] {
			releaseMetalWeight(w.Metal)
			seen[w.Metal] = true
		}
		if w.GPU != nil && !seenGPU[w.GPU] {
			releaseWebGPUWeight(w.GPU)
			seenGPU[w.GPU] = true
		}
		w.Metal = nil
		w.GPU = nil
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
	TokenEmbd Weight
	// PositionEmbd is the learned absolute position-embedding table used by
	// GPT-2 and StarCoder (v1): its row at each sequence position is added to
	// the token embedding once, before layer 0. Zero-valued (Rows==0, F32
	// nil) for every RoPE/ALiBi-based architecture, which is the overwhelming
	// majority — see Config.usesAbsolutePositionEmbd.
	PositionEmbd   Weight
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
