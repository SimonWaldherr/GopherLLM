package gopherllm

// Architecture viewer: an hfviewer-style structural graph of a model, built
// entirely from the GGUF header, metadata, and tensor name/shape presence
// (never weight data), so it is instant even for multi-gigabyte files. See
// AnalyzeGGUF for the numeric facts this reuses; ArchGraph adds the node
// graph, per-layer schedule, and glossary needed to render it.

import (
	"fmt"
)

// ArchDType is Analysis.DTypes reshaped for JSON: a human-readable type name
// and its byte share, rather than the internal GGMLType enum.
type ArchDType struct {
	Type    string  `json:"type"`
	Tensors int     `json:"tensors"`
	Bytes   int64   `json:"bytes"`
	Percent float64 `json:"percent"`
}

// ArchGraphSummary is the "at a glance" stat panel: everything a reader
// wants before looking at the graph itself.
type ArchGraphSummary struct {
	Name         string `json:"name,omitempty"`
	Architecture string `json:"architecture"`
	LoadsAs      string `json:"loadsAs,omitempty"`
	Supported    bool   `json:"supported"`

	Params        int64       `json:"params"`
	FileBytes     int64       `json:"fileBytes"`
	BitsPerWeight float64     `json:"bitsPerWeight"`
	DTypes        []ArchDType `json:"dtypes"`

	Layers        int    `json:"layers"`
	HiddenSize    int    `json:"hiddenSize"`
	FFNSize       int    `json:"ffnSize"`
	AttentionType string `json:"attentionType"`
	Heads         int    `json:"heads"`
	KVHeads       int    `json:"kvHeads"`
	HeadDim       int    `json:"headDim"`

	ContextLength int `json:"contextLength"`
	SlidingWindow int `json:"slidingWindow,omitempty"`
	SWALayers     int `json:"swaLayers,omitempty"`
	VocabSize     int `json:"vocabSize"`

	RopeTheta       float32 `json:"ropeTheta,omitempty"`
	RopeScalingType string  `json:"ropeScalingType,omitempty"`
	PositionScheme  string  `json:"positionScheme"`

	MoEExperts       int  `json:"moeExperts,omitempty"`
	MoEUsed          int  `json:"moeUsed,omitempty"`
	MoESharedExperts int  `json:"moeSharedExperts,omitempty"`
	LeadingDense     int  `json:"leadingDenseBlocks,omitempty"`
	TiedEmbedding    bool `json:"tiedEmbedding"`

	KVCacheBytesAtFullContext int64 `json:"kvCacheBytesAtFullContext"`
	KVCacheBytesAt4K          int64 `json:"kvCacheBytesAt4k"`

	TokenizerModel string `json:"tokenizerModel,omitempty"`
	ChatTemplate   bool   `json:"chatTemplate"`
}

// ArchGraphVision summarizes a paired Pixtral-style vision encoder, when one
// is loaded alongside the text decoder. It is nil for text-only models.
type ArchGraphVision struct {
	Layers        int `json:"layers"`
	HiddenSize    int `json:"hiddenSize"`
	FFNSize       int `json:"ffnSize"`
	Heads         int `json:"heads"`
	PatchSize     int `json:"patchSize"`
	ImageSize     int `json:"imageSize"`
	SpatialMerge  int `json:"spatialMerge"`
	ProjectionDim int `json:"projectionDim"`
}

// ArchGraphNode is one box in the rendered graph. Badges are keys into
// ArchGraph.Glossary. Children, when present, are the node's internal
// structure (e.g. what is inside one transformer block).
type ArchGraphNode struct {
	ID       string          `json:"id"`
	Kind     string          `json:"kind"` // embedding | vision | block | norm | lmhead | attention | mlp | residual
	Label    string          `json:"label"`
	Detail   string          `json:"detail,omitempty"`
	Repeat   int             `json:"repeat,omitempty"`
	Badges   []string        `json:"badges,omitempty"`
	Children []ArchGraphNode `json:"children,omitempty"`
}

// ArchGraphLayer records one decoder layer's mechanism, detected from tensor
// name presence rather than assumed from the architecture label -- so hybrid
// schedules (Nemotron-H's attention/Mamba mix, Qwen3.5/3.6's attention/Gated
// DeltaNet mix, leading dense blocks before MoE starts) show up accurately.
type ArchGraphLayer struct {
	Index     int    `json:"index"`
	Attention string `json:"attention"` // attn | deltanet | mamba | none
	FFN       string `json:"ffn"`       // dense | moe | none
	RoPE      bool   `json:"rope"`
	SWA       bool   `json:"swa"`
	QKNorm    bool   `json:"qkNorm"`
}

// ArchGraph is the full response for the architecture viewer: stats, an
// optional vision tower, the node graph, the per-layer schedule, and a
// glossary of the terms referenced by node badges.
type ArchGraph struct {
	Summary  ArchGraphSummary  `json:"summary"`
	Vision   *ArchGraphVision  `json:"vision,omitempty"`
	Nodes    []ArchGraphNode   `json:"nodes"`
	Layers   []ArchGraphLayer  `json:"layers"`
	Uniform  bool              `json:"uniform"`
	Glossary map[string]string `json:"glossary"`
}

// archGlossary is static, self-contained documentation for every badge key
// used by BuildArchGraph -- no network access needed to explain a term.
var archGlossary = map[string]string{
	"mha":               "Multi-head attention: every query head has its own key/value head.",
	"gqa":               "Grouped-query attention: several query heads share one key/value head, trading a little quality for a much smaller KV cache.",
	"mqa":               "Multi-query attention: every query head shares a single key/value head (the extreme case of GQA).",
	"mla":               "Multi-head Latent Attention: keys/values are compressed into a small low-rank latent before caching, then decompressed per head at attention time.",
	"rope":              "Rotary Position Embedding: rotates query/key vectors by an angle proportional to position, so attention scores depend on relative distance.",
	"alibi":             "Attention with Linear Biases: adds a fixed, per-head penalty proportional to distance directly to attention scores instead of rotating vectors.",
	"absolute-position": "Learned absolute position embeddings, added to the token embedding once at the input (the original GPT-2 scheme).",
	"qk-norm":           "QK-Norm: query and key vectors are normalized before the dot product, stabilizing attention logits at long context or high learning rates.",
	"sliding-window":    "Sliding-window attention: this layer only attends to a fixed-size local window of recent tokens instead of the full context.",
	"swiglu":            "Gated MLP (SwiGLU): the feed-forward block gates an up-projection with a SiLU-activated second projection before the down-projection.",
	"gelu-mlp":          "Gated MLP with a GELU activation instead of SiLU.",
	"plain-mlp":         "Plain (ungated) MLP: a single up-projection, activation, and down-projection, with no gating branch.",
	"moe":               "Mixture of Experts: a router picks a small subset of expert MLPs to run per token instead of one dense MLP, growing capacity without growing compute per token.",
	"shared-expert":     "Shared expert: an always-on expert MLP added to the routed experts' output for every token.",
	"leading-dense":     "Leading dense blocks: the first few layers use an ordinary dense MLP before MoE routing begins deeper in the network.",
	"mamba":             "Mamba-2 state-space mixer: replaces attention with a linear recurrent state that updates per token in constant time and memory, instead of attending over the full context.",
	"deltanet":          "Gated DeltaNet: a linear-recurrent attention alternative that updates an associative memory with the delta rule, mixed with ordinary attention layers on a fixed schedule.",
	"rmsnorm":           "RMSNorm: normalizes activations by their root-mean-square only (no mean-centering, no learned bias), cheaper than LayerNorm.",
	"layernorm":         "LayerNorm: normalizes activations to zero mean and unit variance, then applies a learned scale and bias.",
	"weight-tying":      "Weight tying: the output (LM head) projection reuses the input token-embedding matrix instead of learning a separate one.",
	"parallel-residual": "Parallel residual: attention and MLP are computed from the same normalized input and both added back to the residual stream, instead of feeding one into the other.",
	"softcap":           "Logit softcapping: scores are squashed through cap*tanh(x/cap) before use, bounding attention or output logits to tame outliers.",
	"embedding-scale":   "Embedding scaling: the token embedding is multiplied by a fixed constant (typically sqrt(hidden size)) right after the lookup.",
	"vision-tower":      "Vision encoder: a separate patch-based transformer that turns image patches into the same embedding space as the text tokens.",
	"patch-merger":      "Patch merger: pools a small spatial block of adjacent patch embeddings into one token before handing them to the projector, shrinking the image token count.",
}

// ArchVisionInput carries the paired vision encoder's shape, when one is
// loaded, into BuildArchGraph. Runner.VisionConfig supplies this.
type ArchVisionInput struct {
	Layers        int
	HiddenSize    int
	FFNSize       int
	Heads         int
	PatchSize     int
	ImageSize     int
	SpatialMerge  int
	ProjectionDim int
}

// BuildArchGraph turns a parsed GGUF header, its derived Config, and an
// already-computed Analysis into a renderable architecture graph. vision may
// be nil for a text-only model.
func BuildArchGraph(g *GGUFFile, cfg Config, a *Analysis, vision *ArchVisionInput) *ArchGraph {
	names := make(map[string]struct{}, len(g.Tensors))
	for _, t := range g.Tensors {
		names[t.Name] = struct{}{}
	}
	has := func(name string) bool { _, ok := names[name]; return ok }

	tied := !has("output.weight")

	attnType, attnKey := attentionTypeLabel(cfg)
	ffnType, ffnKeys := ffnTypeLabel(cfg)
	normLabel, normKey := "RMSNorm", "rmsnorm"
	if cfg.UseLayerNorm {
		normLabel, normKey = "LayerNorm", "layernorm"
	}
	posLabel, posKey := "RoPE", "rope"
	switch {
	case cfg.usesALiBi():
		posLabel, posKey = "ALiBi", "alibi"
	case cfg.usesAbsolutePositionEmbd():
		posLabel, posKey = "Absolute position embedding", "absolute-position"
	}

	layers := make([]ArchGraphLayer, cfg.NLayers)
	var anyQKNorm, anySWA, anyMoE, anyDense, anyMamba, anyDeltaNet bool
	for i := 0; i < cfg.NLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		l := ArchGraphLayer{Index: i, Attention: "none", FFN: "none", RoPE: cfg.layerUsesRoPE(i), SWA: cfg.layerUsesSWA(i)}
		switch {
		case has(prefix + "ssm_alpha.weight"):
			l.Attention = "deltanet"
			anyDeltaNet = true
		case has(prefix + "ssm_in.weight"):
			l.Attention = "mamba"
			anyMamba = true
		case has(prefix+"attn_q.weight") || has(prefix+"attn_qkv.weight"):
			l.Attention = "attn"
		}
		if has(prefix+"attn_q_norm.weight") || has(prefix+"attn_k_norm.weight") {
			l.QKNorm = true
			anyQKNorm = true
		}
		switch {
		case has(prefix + "ffn_gate_inp.weight"):
			l.FFN = "moe"
			anyMoE = true
		case has(prefix+"ffn_down.weight") || has(prefix+"ffn_up.weight") || has(prefix+"ffn.weight"):
			l.FFN = "dense"
			anyDense = true
		}
		if l.SWA {
			anySWA = true
		}
		layers[i] = l
	}
	uniform := true
	for i := 1; i < len(layers); i++ {
		if layers[i].Attention != layers[0].Attention || layers[i].FFN != layers[0].FFN ||
			layers[i].RoPE != layers[0].RoPE || layers[i].SWA != layers[0].SWA || layers[i].QKNorm != layers[0].QKNorm {
			uniform = false
			break
		}
	}

	dtypes := make([]ArchDType, len(a.DTypes))
	for i, d := range a.DTypes {
		pct := 0.0
		if a.FileBytes > 0 {
			pct = 100 * float64(d.Bytes) / float64(a.FileBytes)
		}
		dtypes[i] = ArchDType{Type: d.Type.String(), Tensors: d.Tensors, Bytes: d.Bytes, Percent: pct}
	}

	summary := ArchGraphSummary{
		Name: a.Name, Architecture: a.Architecture, LoadsAs: a.LoadsAs, Supported: a.Supported,
		Params: a.Params, FileBytes: a.FileBytes, BitsPerWeight: a.BitsPerWeight, DTypes: dtypes,
		Layers: cfg.NLayers, HiddenSize: cfg.Dim, FFNSize: cfg.HiddenDim,
		AttentionType: attnType, Heads: cfg.NHeads, KVHeads: cfg.NKVHeads, HeadDim: cfg.HeadDim,
		ContextLength: a.ContextLength, SlidingWindow: a.SlidingWindow, SWALayers: a.SWALayers, VocabSize: a.VocabSize,
		RopeTheta: cfg.RopeTheta, RopeScalingType: cfg.RopeScalingType, PositionScheme: posLabel,
		MoEExperts: cfg.ExpertCount, MoEUsed: cfg.ExpertUsedCount, MoESharedExperts: cfg.ExpertSharedCount,
		LeadingDense: cfg.LeadingDenseBlockCount, TiedEmbedding: tied,
		KVCacheBytesAtFullContext: a.KVCacheBytesAtFullContext, KVCacheBytesAt4K: a.KVCacheBytesAt4K,
		TokenizerModel: a.TokenizerModel, ChatTemplate: a.ChatTemplate,
	}

	var visionOut *ArchGraphVision
	nodes := []ArchGraphNode{}
	if vision != nil {
		visionOut = &ArchGraphVision{
			Layers: vision.Layers, HiddenSize: vision.HiddenSize, FFNSize: vision.FFNSize, Heads: vision.Heads,
			PatchSize: vision.PatchSize, ImageSize: vision.ImageSize, SpatialMerge: vision.SpatialMerge, ProjectionDim: vision.ProjectionDim,
		}
		nodes = append(nodes, ArchGraphNode{
			ID: "vision", Kind: "vision", Label: "Vision encoder", Repeat: vision.Layers,
			Detail: fmt.Sprintf("%d layers, %d-dim, patch %dpx, %dx%d merge", vision.Layers, vision.HiddenSize, vision.PatchSize, vision.SpatialMerge, vision.SpatialMerge),
			Badges: []string{"vision-tower", "patch-merger"},
		})
	}

	embDetail := fmt.Sprintf("%d-token vocabulary → %d-dim embedding", cfg.VocabSize, cfg.Dim)
	var embBadges []string
	if cfg.EmbeddingScale != 0 {
		embBadges = append(embBadges, "embedding-scale")
	}
	nodes = append(nodes, ArchGraphNode{ID: "embedding", Kind: "embedding", Label: "Token embedding", Detail: embDetail, Badges: embBadges})

	attnBadges := []string{attnKey, posKey}
	if anyQKNorm {
		attnBadges = append(attnBadges, "qk-norm")
	}
	if anySWA {
		attnBadges = append(attnBadges, "sliding-window")
	}
	if anyMamba {
		attnBadges = append(attnBadges, "mamba")
	}
	if anyDeltaNet {
		attnBadges = append(attnBadges, "deltanet")
	}
	attnDetail := fmt.Sprintf("%d query / %d key-value heads, head dim %d", cfg.NHeads, cfg.NKVHeads, cfg.HeadDim)
	if anySWA {
		if a.SWALayers < 0 {
			attnDetail += fmt.Sprintf("; sliding window %d on every layer", a.SlidingWindow)
		} else {
			attnDetail += fmt.Sprintf("; sliding window %d on %d/%d layers", a.SlidingWindow, a.SWALayers, cfg.NLayers)
		}
	}

	mlpBadges := append([]string{}, ffnKeys...)
	mlpDetail := fmt.Sprintf("hidden size %d", cfg.HiddenDim)
	if anyMoE {
		mlpDetail = fmt.Sprintf("%d experts, top-%d routed", cfg.ExpertCount, cfg.ExpertUsedCount)
		if cfg.ExpertSharedCount > 0 {
			mlpDetail += fmt.Sprintf(" + %d shared", cfg.ExpertSharedCount)
			mlpBadges = append(mlpBadges, "shared-expert")
		}
		if cfg.LeadingDenseBlockCount > 0 {
			mlpDetail += fmt.Sprintf(" (first %d blocks stay dense)", cfg.LeadingDenseBlockCount)
			mlpBadges = append(mlpBadges, "leading-dense")
		}
	} else if !anyDense {
		mlpDetail = "none in this layer"
	}

	blockChildren := []ArchGraphNode{
		{ID: "attn-norm", Kind: "norm", Label: normLabel, Badges: []string{normKey}},
		{ID: "attention", Kind: "attention", Label: attnType, Detail: attnDetail, Badges: attnBadges},
		{ID: "attn-residual", Kind: "residual", Label: "+ residual"},
		{ID: "mlp-norm", Kind: "norm", Label: normLabel, Badges: []string{normKey}},
		{ID: "mlp", Kind: "mlp", Label: ffnType, Detail: mlpDetail, Badges: mlpBadges},
		{ID: "mlp-residual", Kind: "residual", Label: "+ residual"},
	}
	if cfg.ParallelResidual {
		blockChildren = []ArchGraphNode{
			{ID: "block-norm", Kind: "norm", Label: normLabel, Badges: []string{normKey}},
			{ID: "attention", Kind: "attention", Label: attnType, Detail: attnDetail, Badges: attnBadges},
			{ID: "mlp", Kind: "mlp", Label: ffnType, Detail: mlpDetail, Badges: mlpBadges},
			{ID: "residual", Kind: "residual", Label: "+ residual (parallel)"},
		}
	}

	blockLabel := fmt.Sprintf("Transformer block x%d", cfg.NLayers)
	if !uniform {
		blockLabel = fmt.Sprintf("Transformer blocks x%d (heterogeneous, see layer schedule)", cfg.NLayers)
	}
	blockBadges := []string{}
	if cfg.ParallelResidual {
		blockBadges = append(blockBadges, "parallel-residual")
	}
	if cfg.AttnLogitSoftcap > 0 || cfg.FinalLogitSoftcap > 0 {
		blockBadges = append(blockBadges, "softcap")
	}
	nodes = append(nodes, ArchGraphNode{
		ID: "block", Kind: "block", Label: blockLabel, Repeat: cfg.NLayers,
		Badges: blockBadges, Children: blockChildren,
	})

	nodes = append(nodes, ArchGraphNode{ID: "final-norm", Kind: "norm", Label: "Final " + normLabel, Badges: []string{normKey}})

	lmHeadLabel := "LM head"
	var lmHeadBadges []string
	if tied {
		lmHeadLabel = "LM head (tied embedding)"
		lmHeadBadges = append(lmHeadBadges, "weight-tying")
	}
	nodes = append(nodes, ArchGraphNode{
		ID: "lm-head", Kind: "lmhead", Label: lmHeadLabel,
		Detail: fmt.Sprintf("%d-dim → %d-token logits", cfg.Dim, cfg.VocabSize), Badges: lmHeadBadges,
	})

	usedGlossary := map[string]string{}
	var collectBadges func(n ArchGraphNode)
	collectBadges = func(n ArchGraphNode) {
		for _, b := range n.Badges {
			if def, ok := archGlossary[b]; ok {
				usedGlossary[b] = def
			}
		}
		for _, c := range n.Children {
			collectBadges(c)
		}
	}
	for _, n := range nodes {
		collectBadges(n)
	}

	return &ArchGraph{
		Summary: summary, Vision: visionOut, Nodes: nodes, Layers: layers, Uniform: uniform,
		Glossary: usedGlossary,
	}
}

// attentionTypeLabel classifies the attention mechanism from Config alone.
func attentionTypeLabel(cfg Config) (label, key string) {
	switch {
	case cfg.UsesMLA:
		return "Multi-head Latent Attention (MLA)", "mla"
	case cfg.NKVHeads <= 0 || cfg.NHeads <= 0 || cfg.NKVHeads == cfg.NHeads:
		return "Multi-head attention (MHA)", "mha"
	case cfg.NKVHeads == 1:
		return "Multi-query attention (MQA)", "mqa"
	default:
		return "Grouped-query attention (GQA)", "gqa"
	}
}

// ffnTypeLabel classifies the feed-forward mechanism from Config alone.
func ffnTypeLabel(cfg Config) (label string, keys []string) {
	if cfg.ExpertCount > 0 {
		return "Mixture of Experts (MoE)", []string{"moe"}
	}
	if cfg.usesPlainMLP() {
		if cfg.UseExactGELU || cfg.UseGELU {
			return "Plain MLP (GELU)", []string{"plain-mlp"}
		}
		return "Plain MLP", []string{"plain-mlp"}
	}
	if cfg.UseGELU {
		return "Gated MLP (GEGLU)", []string{"gelu-mlp"}
	}
	return "Gated MLP (SwiGLU)", []string{"swiglu"}
}

// runnerVisionInput reads r's paired vision encoder, if any, into the shape
// BuildArchGraph expects.
func runnerVisionInput(r *Runner) *ArchVisionInput {
	vc, ok := r.VisionConfig()
	if !ok {
		return nil
	}
	return &ArchVisionInput{
		Layers: vc.BlockCount, HiddenSize: vc.EmbeddingLength, FFNSize: vc.FeedForwardLength, Heads: vc.HeadCount,
		PatchSize: vc.PatchSize, ImageSize: vc.ImageSize, SpatialMerge: vc.SpatialMergeSize, ProjectionDim: vc.ProjectionDim,
	}
}

// RunnerArchGraph builds the architecture graph for an already-loaded
// Runner, including its paired vision encoder if one was loaded via
// LoadOptions.VisionProjector*. Exported so server-side handlers holding a
// *Runner (rather than the higher-level Model) can reuse it directly.
func RunnerArchGraph(r *Runner) *ArchGraph {
	a := AnalyzeGGUF(r.GGUF(), r.Tokenizer())
	return BuildArchGraph(r.GGUF(), r.Config(), a, runnerVisionInput(r))
}

// ArchGraph builds the architecture graph for m's loaded model, including
// its paired vision encoder if one was loaded via LoadOptions.VisionProjector*.
func (m *Model) ArchGraph() *ArchGraph {
	return RunnerArchGraph(m.r)
}
