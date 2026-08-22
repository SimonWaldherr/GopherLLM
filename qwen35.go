package gopherllm

// Native Qwen3.5 / Qwen3.6 / Qwen3.8 ("qwen35") inference.
//
// EXPERIMENTAL: this is a native, text-only single-token decode path for the
// Qwen3.5/3.6/3.8 hybrid architecture. Its DeltaNet and gated-attention math
// follows the Qwen reference graph and the llama.cpp GGUF conversion layout:
// QKV is [Q|K|V], Q/K use L2 normalization, and full attention uses a sigmoid
// gate. The implementation has focused unit coverage, including the optional
// one-layer MTP draft head used for exact greedy speculation, but does not yet
// offer cross-runtime logit-parity validation, chunked DeltaNet prefill, or
// multimodal vision/3D-MRoPE.
//
// Reconstructed architecture: most layers replace ordinary self-attention
// with a Gated DeltaNet linear-recurrent mixer, and every
// FullAttentionInterval-th layer (here: every 4th) keeps a standard
// Qwen3-style GQA attention block with QK-RMSNorm and partial RoPE
// (RopeDimensionCount is a fraction of HeadDim — 64 of 256 here — the
// existing partial-rotary mechanism already used by the standard decoder).
// Those full-attention layers project attn_q to twice HeadDim per head;
// dimensional analysis rules out "32 plain heads" (attn_output's input width
// only equals NHeads*ValueDim for 16 heads), so the extra half is treated as
// a per-head sigmoid gate applied to that head's attention output — Qwen3's
// tech report describes gating even the full-attention layers to match the
// DeltaNet layers' own output gate, and there is no separate attn_gate tensor
// on these layers, unlike the DeltaNet ones.
//
// The GGUF converter names the DeltaNet's projections with the same
// "attn_"/"ssm_" prefixes llama.cpp uses for Mamba-2 (see nemotron_h.go's
// package comment); the tensor shapes against
// ssm.inner_size/state_size/group_count/time_step_rank match Mamba-2's exact
// conv+SSM recurrence (nemotronMambaForward) PLUS one addition: before
// writing a token's value into the per-head state, the state is first asked
// what it currently recalls for that token's key, and only the residual (the
// "delta") is written in — the delta rule (Yang et al., "Parallelizing Linear
// Transformers with the Delta Rule over Sequence Length"), gating the update
// by a per-head, input-dependent β_t in addition to Mamba-2's own decay gate.
//
// qwen35moe (Ornith and similar) replaces the dense SwiGLU FFN with a sparse
// MoE using Mixtral-style gated experts plus Qwen's gated shared expert. The
// common sparse-MoE loader/forward path is shared with the other decoder
// families, while this file owns only the hybrid DeltaNet/attention schedule.

import (
	"fmt"
	"io"
	"math"
)

type qwen35LayerKind uint8

const (
	qwen35DeltaNet qwen35LayerKind = iota
	qwen35Attention
)

func qwen35Family(arch string) bool {
	return arch == "qwen35" || arch == "qwen35moe"
}

type Qwen35AttentionWeights struct {
	Q, K, V, O   Weight
	QNorm, KNorm []float32
}

// Qwen35DeltaNetWeights holds one Gated DeltaNet layer's projections. QKVConv
// projects the normalized hidden state to the concatenated [Q | K | V] input
// of the causal short convolution. Q/K are per-key-group streams, while V has
// one stream per value head. Gate/AlphaProj/BetaProj read directly from the
// un-convolved hidden state.
type Qwen35DeltaNetWeights struct {
	QKVConv    Weight
	ConvKernel Weight
	Gate       Weight
	AlphaProj  Weight
	BetaProj   Weight
	A          []float32
	DTBias     []float32
	Norm       []float32
	Out        Weight
}

type Qwen35FFNWeights struct {
	Gate, Up, Down Weight
	MoE            *SparseMoEWeights
}

type Qwen35LayerWeights struct {
	Norm         []float32
	PostAttnNorm []float32
	Kind         qwen35LayerKind
	// KVCacheSlot and RecurrentCacheSlot compact the two disjoint cache
	// classes. They are assigned at load time; only a periodic subset of Qwen
	// layers uses full attention, while every other layer uses DeltaNet state.
	KVCacheSlot        int
	RecurrentCacheSlot int
	Attention          Qwen35AttentionWeights
	DeltaNet           Qwen35DeltaNetWeights
	FFN                Qwen35FFNWeights
}

type Qwen35Weights struct {
	TokenEmbd  Weight
	OutputNorm []float32
	Output     Weight
	Layers     []Qwen35LayerWeights
	// MTP is Qwen's optional NextN draft block. It deliberately lives outside
	// Layers: the block is stored after the autoregressive trunk in GGUF, owns
	// a separate attention KV cache, and consumes a pair of the previous target
	// hidden state plus the current token embedding.
	MTP *Qwen35MTPWeights
}

// Qwen35MTPWeights is Qwen3.5/3.6/3.8's one-layer NextN/MTP draft head. The
// transformer sub-block has the same gated full-attention and dense SwiGLU
// shape as a regular Qwen full-attention layer, while the input pair and final
// shared head are specific to MTP.
type Qwen35MTPWeights struct {
	TokenEmbd      Weight
	EmbeddingNorm  []float32
	HiddenNorm     []float32
	EHProj         Weight
	Norm           []float32
	PostAttnNorm   []float32
	Attention      Qwen35AttentionWeights
	FFN            Qwen35FFNWeights
	SharedHeadNorm []float32
	Output         Weight
}

// qwen35MTPExecutionWeights selects CPU-SIMD views of the MTP body matrices
// even when the large target graph uses selective Metal offload. One draft
// block performs several small, dependency-chained matvecs; individually
// submitting them to Metal costs more synchronization than it saves. The
// vocab-wide shared head intentionally remains on Metal, where its projection
// plus argmax reduction does amortize one command buffer. This only changes a
// value copy, so the original Metal ownership remains available to the target
// decoder and is released exactly once with the loaded model.
func qwen35MTPExecutionWeights(weights Qwen35MTPWeights) Qwen35MTPWeights {
	disableMetal := func(w *Weight) { w.Metal = nil }
	disableMetal(&weights.TokenEmbd)
	disableMetal(&weights.EHProj)
	disableMetal(&weights.Attention.Q)
	disableMetal(&weights.Attention.K)
	disableMetal(&weights.Attention.V)
	disableMetal(&weights.Attention.O)
	disableMetal(&weights.FFN.Gate)
	disableMetal(&weights.FFN.Up)
	disableMetal(&weights.FFN.Down)
	return weights
}

// Qwen35Cache holds the Gated DeltaNet recurrent state (one HeadDim x HeadDim
// matrix per head per recurrent layer — the delta rule's associative memory)
// plus the short causal-conv history. Attention-kind layers use the separate
// compact K/V cache, exactly like Nemotron-H's dual-cache pattern.
type Qwen35Cache struct {
	Conv     []float32
	State    []float32
	Layers   int
	Channels int
	ConvLen  int
	Heads    int
	HeadDim  int
}

// Qwen35MTPState is the independent state of the MTP draft head. Unlike the
// target hybrid graph it has exactly one full-attention layer, so it needs only
// one KV-cache slot. PendingHidden carries the target's final hidden state from
// the preceding position; it is intentionally separate from draft-only hidden
// state, because speculative candidate rows must never overwrite the accepted
// target boundary.
type Qwen35MTPState struct {
	KV            *KVCache
	PendingHidden []float32
	DraftHidden   []float32
	DraftTokens   []uint32
	DraftRecent   []uint32
	Scratch       *DecodeBuffer
}

func newQwen35MTPState(cfg Config, maxLen int, format kvFormat) *Qwen35MTPState {
	kDim := cfg.NKVHeads * cfg.HeadDim
	vDim := cfg.KVDim
	if vDim <= 0 {
		vDim = cfg.NKVHeads * cfg.ValueDim
	}
	var kv *KVCache
	switch format {
	case kvI8:
		kv = NewKVCacheI8(1, kDim, vDim, maxLen)
	case kvF16:
		kv = NewKVCacheF16(1, kDim, vDim, maxLen)
	default:
		kv = NewKVCache(1, kDim, vDim, maxLen)
	}
	return &Qwen35MTPState{
		KV:            kv,
		PendingHidden: make([]float32, cfg.Dim),
		DraftHidden:   make([]float32, cfg.Dim),
		DraftTokens:   make([]uint32, 0, 4),
		DraftRecent:   make([]uint32, 0, repeatPenaltyWindow),
		Scratch:       NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim),
	}
}

func (s *Qwen35MTPState) compatible(cfg Config, maxLen int, format kvFormat) bool {
	if s == nil || s.KV == nil || s.Scratch == nil ||
		len(s.PendingHidden) != cfg.Dim || len(s.DraftHidden) != cfg.Dim {
		return false
	}
	kDim := cfg.NKVHeads * cfg.HeadDim
	vDim := cfg.KVDim
	if vDim <= 0 {
		vDim = cfg.NKVHeads * cfg.ValueDim
	}
	return s.KV.layerCount() == 1 && s.KV.kvFormat() == format &&
		s.KV.PerPosKDim == kDim && s.KV.PerPosVDim == vDim && s.KV.MaxLen >= maxLen
}

// reset starts a fresh target sequence while retaining KV backing storage.
// Causal attention only ever reads through the current position, so stale rows
// are harmless and retaining them is what makes prefix-cache reuse possible.
func (s *Qwen35MTPState) reset() {
	if s == nil {
		return
	}
	clear(s.PendingHidden)
	clear(s.DraftHidden)
	s.DraftTokens = s.DraftTokens[:0]
	s.DraftRecent = s.DraftRecent[:0]
}

func (s *Qwen35MTPState) pendingSnapshot() []float32 {
	if s == nil || len(s.PendingHidden) == 0 {
		return nil
	}
	return append([]float32(nil), s.PendingHidden...)
}

func (s *Qwen35MTPState) restorePending(snapshot []float32) bool {
	if s == nil || len(snapshot) != len(s.PendingHidden) {
		return false
	}
	copy(s.PendingHidden, snapshot)
	return true
}

func copyQwen35MTPPrefix(dst, src *Qwen35MTPState, positions int) int {
	if dst == nil || src == nil {
		return 0
	}
	return copyKVPrefix(dst.KV, src.KV, positions)
}

func newQwen35Cache(c Config, recurrentLayers int) *Qwen35Cache {
	channels := c.SSMInner + 2*c.SSMGroups*c.SSMState
	headDim := 0
	if c.SSMHeads > 0 {
		headDim = c.SSMInner / c.SSMHeads
	}
	cache := &Qwen35Cache{
		Layers: recurrentLayers, Channels: channels, ConvLen: max(0, c.SSMConv-1),
		Heads: c.SSMHeads, HeadDim: headDim,
	}
	if cache.ConvLen > 0 && channels > 0 {
		cache.Conv = make([]float32, cache.Layers*channels*cache.ConvLen)
	}
	if headDim > 0 && c.SSMHeads > 0 {
		cache.State = make([]float32, cache.Layers*c.SSMHeads*headDim*headDim)
	}
	return cache
}

func (c *Qwen35Cache) compatible(cfg Config, recurrentLayers int) bool {
	if c == nil {
		return false
	}
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	headDim := 0
	if cfg.SSMHeads > 0 {
		headDim = cfg.SSMInner / cfg.SSMHeads
	}
	return c.Layers == recurrentLayers && c.Channels == channels && c.ConvLen == max(0, cfg.SSMConv-1) &&
		c.Heads == cfg.SSMHeads && c.HeadDim == headDim
}

func (c *Qwen35Cache) reset() {
	clear(c.Conv)
	clear(c.State)
}

// bytes reports the resident recurrent-state footprint. It deliberately
// excludes the small struct header so callers can budget snapshots alongside
// the ordinary K/V cache without depending on Go's object layout.
func (c *Qwen35Cache) bytes() int64 {
	if c == nil {
		return 0
	}
	return int64(len(c.Conv)+len(c.State)) * 4
}

// snapshot returns an independent copy of a fully initialized recurrent
// cache. Qwen's DeltaNet state is updated in place for every token, so K/V
// rows alone are insufficient to resume a cached chat prefix safely.
func (c *Qwen35Cache) snapshot() *Qwen35Cache {
	if c == nil {
		return nil
	}
	snap := &Qwen35Cache{
		Layers: c.Layers, Channels: c.Channels, ConvLen: c.ConvLen,
		Heads: c.Heads, HeadDim: c.HeadDim,
	}
	snap.Conv = append(snap.Conv, c.Conv...)
	snap.State = append(snap.State, c.State...)
	return snap
}

// restore replaces this cache's mutable recurrent state from a snapshot.
// Shapes and backing lengths must agree exactly: accepting a partial state
// would silently corrupt a later DeltaNet update.
func (c *Qwen35Cache) restore(snapshot *Qwen35Cache) bool {
	if c == nil || snapshot == nil ||
		c.Layers != snapshot.Layers || c.Channels != snapshot.Channels || c.ConvLen != snapshot.ConvLen ||
		c.Heads != snapshot.Heads || c.HeadDim != snapshot.HeadDim ||
		len(c.Conv) != len(snapshot.Conv) || len(c.State) != len(snapshot.State) {
		return false
	}
	copy(c.Conv, snapshot.Conv)
	copy(c.State, snapshot.State)
	return true
}

func (c *Qwen35Cache) convOffset(layer, channel int) int {
	return (layer*c.Channels + channel) * c.ConvLen
}

// stateOffset returns the start of head h's HeadDim x HeadDim state matrix,
// stored row-major with rows indexed by the value (X) dimension and columns
// by the key (B/C) dimension — recall(row) = dot(row, k), update adds
// delta*k to each row, and output reads dot(row, q) per row.
func (c *Qwen35Cache) stateOffset(layer, head int) int {
	return (layer*c.Heads + head) * c.HeadDim * c.HeadDim
}

func LoadQwen35Model(data []byte, gguf *GGUFFile, borrow, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, Qwen35Weights, error) {
	if logw == nil {
		logw = io.Discard
	}
	cfg := ConfigFromGGUF(gguf)
	if cfg.Arch != "qwen35" && cfg.Arch != "qwen35moe" {
		return cfg, Qwen35Weights{}, fmt.Errorf("not a Qwen hybrid GGUF: %s", cfg.Arch)
	}
	tensors := indexTensors(gguf)
	// Some Qwen3.6/3.8 exports append MTP (multi-token prediction) draft blocks
	// to the decoder stack. Prefer the explicit metadata count, then retain a
	// tensor-marker fallback for older converters. The draft blocks are useful
	// only to speculative decoding; normal autoregressive inference stops at
	// the preceding trunk layer.
	skippedMTP := cfg.NextNPredictLayers
	if skippedMTP > 0 {
		if skippedMTP >= cfg.NLayers {
			return cfg, Qwen35Weights{}, fmt.Errorf("%s: nextn_predict_layers=%d leaves no decoder layers", cfg.Arch, skippedMTP)
		}
		for il := cfg.NLayers - skippedMTP; il < cfg.NLayers; il++ {
			if _, ok := tensors[fmt.Sprintf("blk.%d.nextn.eh_proj.weight", il)]; !ok {
				return cfg, Qwen35Weights{}, fmt.Errorf("%s: MTP metadata declares draft layer %d but its nextn.eh_proj.weight is missing", cfg.Arch, il)
			}
		}
		cfg.NLayers -= skippedMTP
	} else {
		for cfg.NLayers > 1 {
			last := cfg.NLayers - 1
			if _, ok := tensors[fmt.Sprintf("blk.%d.nextn.eh_proj.weight", last)]; !ok {
				break
			}
			cfg.NLayers--
			skippedMTP++
		}
	}
	if skippedMTP > 1 {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: %d MTP draft layers are unsupported (only the one-layer Qwen NextN head is implemented)", cfg.Arch, skippedMTP)
	}
	if cfg.Dim <= 0 || cfg.NLayers <= 0 || cfg.SSMConv <= 0 || cfg.SSMInner <= 0 || cfg.SSMState <= 0 ||
		cfg.SSMHeads <= 0 || cfg.SSMGroups <= 0 ||
		(cfg.FullAttentionInterval <= 0 && len(cfg.QwenRecurrentLayers) == 0) {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: incomplete Gated DeltaNet metadata", cfg.Arch)
	}
	if cfg.SSMInner%cfg.SSMHeads != 0 || cfg.SSMHeads%cfg.SSMGroups != 0 {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: invalid DeltaNet dimensions inner=%d heads=%d groups=%d", cfg.Arch, cfg.SSMInner, cfg.SSMHeads, cfg.SSMGroups)
	}
	if headV := cfg.SSMInner / cfg.SSMHeads; headV != cfg.SSMState {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: unsupported DeltaNet head geometry value=%d key=%d (Qwen requires equal head widths)", cfg.Arch, headV, cfg.SSMState)
	}
	if cfg.HeadDim <= 0 || cfg.ValueDim != cfg.HeadDim {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: unsupported full-attention head geometry key=%d value=%d", cfg.Arch, cfg.HeadDim, cfg.ValueDim)
	}
	if len(cfg.QwenRecurrentLayers) > 0 && len(cfg.QwenRecurrentLayers) < cfg.NLayers {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: recurrent-layer schedule has %d entries, need at least %d", cfg.Arch, len(cfg.QwenRecurrentLayers), cfg.NLayers)
	}
	if cfg.Arch == "qwen35moe" {
		if cfg.ExpertCount <= 0 || cfg.ExpertUsedCount <= 0 || cfg.ExpertUsedCount > cfg.ExpertCount {
			return cfg, Qwen35Weights{}, fmt.Errorf("%s: invalid MoE metadata experts=%d used=%d", cfg.Arch, cfg.ExpertCount, cfg.ExpertUsedCount)
		}
	} else if cfg.ExpertCount != 0 {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: dense graph declares %d experts; use qwen35moe for sparse-MoE checkpoints", cfg.Arch, cfg.ExpertCount)
	}
	inferred := inferTensorSizes(data, gguf)
	lazyScalarWeights := len(outOfCore) > 0 && outOfCore[0]
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
	}
	// The depthwise DeltaNet convolution is tiny relative to the checkpoint but
	// runs once per channel for every recurrent layer and generated token. Keep
	// scalar kernels hot even for an out-of-core model: a Qwen3.8-27B kernel
	// bank is only a few MiB, while repeatedly decoding it from the mmap would
	// otherwise add hundreds of thousands of row conversions per token.
	loadHotScalar := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal, false)
	}
	loadVec := func(name string) ([]float32, error) {
		return loadF32Vec(data, gguf.DataOffset, name, tensors, inferred)
	}
	token, err := load("token_embd.weight")
	if err != nil {
		return cfg, Qwen35Weights{}, err
	}
	outNorm, err := loadVec("output_norm.weight")
	if err != nil {
		return cfg, Qwen35Weights{}, err
	}
	out := token
	if _, ok := tensors["output.weight"]; ok {
		out, err = load("output.weight")
		if err != nil {
			return cfg, Qwen35Weights{}, err
		}
	}
	weights := Qwen35Weights{TokenEmbd: token, OutputNorm: outNorm, Output: out, Layers: make([]Qwen35LayerWeights, cfg.NLayers)}
	attentionLayers, recurrentLayers := 0, 0
	for i := range cfg.NLayers {
		prefix := fmt.Sprintf("blk.%d.", i)
		norm, err := loadVec(prefix + "attn_norm.weight")
		if err != nil {
			return cfg, Qwen35Weights{}, err
		}
		postNorm, err := loadVec(prefix + "post_attention_norm.weight")
		if err != nil {
			return cfg, Qwen35Weights{}, err
		}
		layer := Qwen35LayerWeights{
			Norm: norm, PostAttnNorm: postNorm,
			KVCacheSlot: -1, RecurrentCacheSlot: -1,
		}
		isRecurrent := true
		if len(cfg.QwenRecurrentLayers) > 0 {
			isRecurrent = cfg.QwenRecurrentLayers[i]
		} else {
			isRecurrent = (i+1)%cfg.FullAttentionInterval != 0
		}
		if !isRecurrent {
			layer.Kind = qwen35Attention
			layer.KVCacheSlot = attentionLayers
			attentionLayers++
			layer.Attention.Q, err = load(prefix + "attn_q.weight")
			if err == nil {
				layer.Attention.K, err = load(prefix + "attn_k.weight")
			}
			if err == nil {
				layer.Attention.V, err = load(prefix + "attn_v.weight")
			}
			if err == nil {
				layer.Attention.O, err = load(prefix + "attn_output.weight")
			}
			if err == nil {
				layer.Attention.QNorm, err = loadVec(prefix + "attn_q_norm.weight")
			}
			if err == nil {
				layer.Attention.KNorm, err = loadVec(prefix + "attn_k_norm.weight")
			}
			if err != nil {
				return cfg, Qwen35Weights{}, fmt.Errorf("layer %d attention: %w", i, err)
			}
			if len(layer.Attention.QNorm) != cfg.HeadDim || len(layer.Attention.KNorm) != cfg.HeadDim {
				return cfg, Qwen35Weights{}, fmt.Errorf("layer %d: qwen35 attention requires %d-element attn_q_norm/attn_k_norm", i, cfg.HeadDim)
			}
		} else {
			layer.Kind = qwen35DeltaNet
			layer.RecurrentCacheSlot = recurrentLayers
			recurrentLayers++
			layer.DeltaNet.QKVConv, err = load(prefix + "attn_qkv.weight")
			if err == nil {
				layer.DeltaNet.ConvKernel, err = loadHotScalar(prefix + "ssm_conv1d.weight")
			}
			if err == nil {
				layer.DeltaNet.Gate, err = load(prefix + "attn_gate.weight")
			}
			if err == nil {
				layer.DeltaNet.AlphaProj, err = load(prefix + "ssm_alpha.weight")
			}
			if err == nil {
				layer.DeltaNet.BetaProj, err = load(prefix + "ssm_beta.weight")
			}
			if err == nil {
				layer.DeltaNet.A, err = loadVec(prefix + "ssm_a")
			}
			if err == nil {
				layer.DeltaNet.DTBias, err = loadVec(prefix + "ssm_dt.bias")
			}
			if err == nil {
				layer.DeltaNet.Norm, err = loadVec(prefix + "ssm_norm.weight")
			}
			if err == nil {
				layer.DeltaNet.Out, err = load(prefix + "ssm_out.weight")
			}
			if err != nil {
				return cfg, Qwen35Weights{}, fmt.Errorf("layer %d Gated DeltaNet: %w", i, err)
			}
		}
		if _, isMoE := tensors[prefix+"ffn_gate_inp.weight"]; isMoE {
			layer.FFN.MoE, err = loadSparseMoEWeights(data, gguf.DataOffset, prefix, cfg, tensors, inferred, borrow, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return cfg, Qwen35Weights{}, fmt.Errorf("layer %d MoE: %w", i, err)
			}
		} else {
			if cfg.Arch == "qwen35moe" {
				return cfg, Qwen35Weights{}, fmt.Errorf("layer %d MoE: missing tensor: %s", i, prefix+"ffn_gate_inp.weight")
			}
			layer.FFN.Gate, err = load(prefix + "ffn_gate.weight")
			if err == nil {
				layer.FFN.Up, err = load(prefix + "ffn_up.weight")
			}
			if err == nil {
				layer.FFN.Down, err = load(prefix + "ffn_down.weight")
			}
			if err != nil {
				return cfg, Qwen35Weights{}, fmt.Errorf("layer %d FFN: %w", i, err)
			}
		}
		weights.Layers[i] = layer
		if i == 0 || (i+1)%8 == 0 || i+1 == cfg.NLayers {
			fmt.Fprintf(logw, "  Loaded Qwen hybrid layer %d/%d\n", i+1, cfg.NLayers)
		}
	}
	if attentionLayers == 0 || recurrentLayers == 0 {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: hybrid schedule must contain both attention and recurrent layers", cfg.Arch)
	}
	if skippedMTP > 0 {
		// Qwen stores its single MTP block directly after the autoregressive
		// trunk. The block's attention/FFN tensors deliberately retain their
		// ordinary names; only its input pair and shared output head are under
		// nextn.*. Keep it separate from Layers so ordinary target decode never
		// appends these K/V rows to the hybrid cache.
		prefix := fmt.Sprintf("blk.%d.", cfg.NLayers)
		mtp := &Qwen35MTPWeights{
			TokenEmbd:      token,
			Output:         out,
			SharedHeadNorm: outNorm,
		}
		if _, ok := tensors[prefix+"nextn.embed_tokens.weight"]; ok {
			mtp.TokenEmbd, err = load(prefix + "nextn.embed_tokens.weight")
		}
		if err == nil {
			mtp.EmbeddingNorm, err = loadVec(prefix + "nextn.enorm.weight")
		}
		if err == nil {
			mtp.HiddenNorm, err = loadVec(prefix + "nextn.hnorm.weight")
		}
		if err == nil {
			mtp.EHProj, err = load(prefix + "nextn.eh_proj.weight")
		}
		if err == nil {
			mtp.Norm, err = loadVec(prefix + "attn_norm.weight")
		}
		if err == nil {
			mtp.PostAttnNorm, err = loadVec(prefix + "post_attention_norm.weight")
		}
		if err == nil {
			mtp.Attention.Q, err = load(prefix + "attn_q.weight")
		}
		if err == nil {
			mtp.Attention.K, err = load(prefix + "attn_k.weight")
		}
		if err == nil {
			mtp.Attention.V, err = load(prefix + "attn_v.weight")
		}
		if err == nil {
			mtp.Attention.O, err = load(prefix + "attn_output.weight")
		}
		if err == nil {
			mtp.Attention.QNorm, err = loadVec(prefix + "attn_q_norm.weight")
		}
		if err == nil {
			mtp.Attention.KNorm, err = loadVec(prefix + "attn_k_norm.weight")
		}
		if err == nil {
			mtp.FFN.Gate, err = load(prefix + "ffn_gate.weight")
		}
		if err == nil {
			mtp.FFN.Up, err = load(prefix + "ffn_up.weight")
		}
		if err == nil {
			mtp.FFN.Down, err = load(prefix + "ffn_down.weight")
		}
		if err == nil {
			if _, ok := tensors[prefix+"nextn.shared_head_norm.weight"]; ok {
				mtp.SharedHeadNorm, err = loadVec(prefix + "nextn.shared_head_norm.weight")
			}
		}
		if err == nil {
			// Most Qwen exports share the target LM head, but preserve a
			// checkpoint-provided dedicated head when it is present.
			if _, ok := tensors[prefix+"nextn.shared_head_head.weight"]; ok {
				mtp.Output, err = load(prefix + "nextn.shared_head_head.weight")
			}
		}
		if err != nil {
			return cfg, Qwen35Weights{}, fmt.Errorf("Qwen MTP draft layer %d: %w", cfg.NLayers, err)
		}
		if len(mtp.EmbeddingNorm) != cfg.Dim || len(mtp.HiddenNorm) != cfg.Dim ||
			len(mtp.Norm) != cfg.Dim || len(mtp.PostAttnNorm) != cfg.Dim || len(mtp.SharedHeadNorm) != cfg.Dim {
			return cfg, Qwen35Weights{}, fmt.Errorf("Qwen MTP draft layer %d: expected %d-element RMSNorm weights", cfg.NLayers, cfg.Dim)
		}
		if len(mtp.Attention.QNorm) != cfg.HeadDim || len(mtp.Attention.KNorm) != cfg.HeadDim {
			return cfg, Qwen35Weights{}, fmt.Errorf("Qwen MTP draft layer %d: expected %d-element attn_q_norm/attn_k_norm", cfg.NLayers, cfg.HeadDim)
		}
		weights.MTP = mtp
		fmt.Fprintf(logw, "  Loaded Qwen MTP draft layer %d (separate NextN cache)\n", cfg.NLayers+1)
	}
	return cfg, weights, nil
}

func qwen35AttentionLayerCount(weights Qwen35Weights) int {
	n := 0
	for _, layer := range weights.Layers {
		if layer.Kind == qwen35Attention {
			n++
		}
	}
	return n
}

func qwen35RecurrentLayerCount(weights Qwen35Weights) int {
	n := 0
	for _, layer := range weights.Layers {
		if layer.Kind == qwen35DeltaNet {
			n++
		}
	}
	return n
}

func ForwardQwen35Into(cfg Config, weights Qwen35Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardQwen35BodyInto(cfg, weights, cache, buf, token, pos)
	weights.Output.MatvecInto(buf.XN, logits)
}

func ForwardQwen35BodyInto(cfg Config, weights Qwen35Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	if cache == nil || cache.Qwen35 == nil {
		panic("Qwen hybrid forward requires a recurrent cache")
	}
	weights.TokenEmbd.RowInto(int(token), cfg.Dim, &buf.X)
	// RoPE only depends on position, not layer, for this architecture: safe
	// to prepare once and reuse across every attention-kind layer below,
	// exactly as the standard decoder does in ForwardBodyInto.
	ropeHalf, ropePairs := prepareRopeScratch(pos, cfg.HeadDim, cfg.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &buf.RopeSin, &buf.RopeCos)
	for _, layer := range weights.Layers {
		rmsNormInto(buf.X, layer.Norm, cfg.RMSNormEps, &buf.XN)
		switch layer.Kind {
		case qwen35Attention:
			qwen35AttentionForward(cfg, layer.Attention, cache, buf.XN, layer.KVCacheSlot, pos, ropeHalf, ropePairs, buf)
		default:
			qwen35DeltaNetForward(cfg, layer.DeltaNet, cache.Qwen35, buf.XN, layer.RecurrentCacheSlot, buf)
		}
		addInPlace(buf.X[:cfg.Dim], buf.Proj[:cfg.Dim])
		rmsNormInto(buf.X, layer.PostAttnNorm, cfg.RMSNormEps, &buf.XN)
		if layer.FFN.MoE != nil {
			sparseMoEForward(layer.FFN.MoE, buf.XN, buf)
		} else {
			qwen35FFNForward(layer.FFN, buf.XN, buf)
		}
		addInPlace(buf.X[:cfg.Dim], buf.Proj[:cfg.Dim])
	}
	rmsNormInto(buf.X, weights.OutputNorm, cfg.RMSNormEps, &buf.XN)
}

// ForwardQwen35MTPBodyInto runs the one-layer Qwen NextN draft head and
// leaves its shared-head-normalized hidden state in state.Scratch.XN. The
// caller chooses whether to project logits: prompt/cache catch-up only needs
// K/V and skips the expensive vocabulary projection, while drafting needs it.
//
// hPrev is the target's final hidden state at pos-1. This one-position shift
// is part of Qwen's MTP training graph: at position p the draft layer combines
// token[p] with target_hidden[p-1] and predicts token[p+1].
func ForwardQwen35MTPBodyInto(cfg Config, weights Qwen35MTPWeights, state *Qwen35MTPState, token uint32, hPrev []float32, pos int) {
	if state == nil || state.KV == nil || state.Scratch == nil {
		panic("Qwen MTP forward requires an MTP cache")
	}
	if len(hPrev) < cfg.Dim {
		panic("Qwen MTP forward received a short target hidden state")
	}
	weights = qwen35MTPExecutionWeights(weights)
	buf := state.Scratch
	weights.TokenEmbd.RowInto(int(token), cfg.Dim, &buf.XN2)
	rmsNormInto(buf.XN2, weights.EmbeddingNorm, cfg.RMSNormEps, &buf.X)
	rmsNormInto(hPrev[:cfg.Dim], weights.HiddenNorm, cfg.RMSNormEps, &buf.XN)
	ensureLenNoClear(&buf.MTPInput, 2*cfg.Dim)
	copy(buf.MTPInput[:cfg.Dim], buf.X[:cfg.Dim])
	copy(buf.MTPInput[cfg.Dim:], buf.XN[:cfg.Dim])
	weights.EHProj.MatvecInto(buf.MTPInput, &buf.X)

	ropeHalf, ropePairs := prepareRopeScratch(pos, cfg.HeadDim, cfg.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &buf.RopeSin, &buf.RopeCos)
	rmsNormInto(buf.X, weights.Norm, cfg.RMSNormEps, &buf.XN)
	qwen35AttentionForward(cfg, weights.Attention, state.KV, buf.XN, 0, pos, ropeHalf, ropePairs, buf)
	addInPlace(buf.X[:cfg.Dim], buf.Proj[:cfg.Dim])
	rmsNormInto(buf.X, weights.PostAttnNorm, cfg.RMSNormEps, &buf.XN)
	qwen35FFNForward(weights.FFN, buf.XN, buf)
	addInPlace(buf.X[:cfg.Dim], buf.Proj[:cfg.Dim])
	rmsNormInto(buf.X, weights.SharedHeadNorm, cfg.RMSNormEps, &buf.XN)
}

func ForwardQwen35MTPInto(cfg Config, weights Qwen35MTPWeights, state *Qwen35MTPState, token uint32, hPrev []float32, pos int, logits *[]float32) {
	ForwardQwen35MTPBodyInto(cfg, weights, state, token, hPrev, pos)
	qwen35MTPExecutionWeights(weights).Output.MatvecInto(state.Scratch.XN, logits)
}

// qwen35MTPProcess catches the draft head up with one accepted target token.
// It intentionally overwrites any speculative row at pos with the true target
// hidden-state pair; later stale draft rows are causally invisible until they
// too are overwritten, so no costly KV copy or rollback is necessary.
func qwen35MTPProcess(cfg Config, weights *Qwen35MTPWeights, state *Qwen35MTPState, token uint32, targetHidden []float32, pos int) {
	if weights == nil || state == nil {
		return
	}
	ForwardQwen35MTPBodyInto(cfg, *weights, state, token, state.PendingHidden, pos)
	copy(state.PendingHidden, targetHidden[:cfg.Dim])
}

// qwen35MTPDraftGreedy produces up to maxTokens candidate continuations from
// the synchronized MTP state. It is used only with deterministic target
// sampling, so equality verification preserves the target distribution
// exactly. Draft rows beyond the accepted boundary are harmless: process()
// replaces them before any later attention reads them.
func qwen35MTPDraftGreedy(cfg Config, weights *Qwen35MTPWeights, state *Qwen35MTPState, token uint32, pos, maxTokens int, recent []uint32, repeatPenalty float32) []uint32 {
	if weights == nil || state == nil || maxTokens <= 0 {
		return nil
	}
	state.DraftTokens = state.DraftTokens[:0]
	state.DraftRecent = append(state.DraftRecent[:0], recent...)
	copy(state.DraftHidden, state.PendingHidden)
	input := token
	executionWeights := qwen35MTPExecutionWeights(*weights)
	for step := 0; step < maxTokens && pos+step < state.KV.MaxLen; step++ {
		ForwardQwen35MTPBodyInto(cfg, executionWeights, state, input, state.DraftHidden, pos+step)
		output := ModelWeights{Output: executionWeights.Output}
		next, ok := argmaxOutputTokenPenalizedInto(cfg, output, state.Scratch, state.DraftRecent, repeatPenalty, &state.Scratch.Logits)
		if !ok {
			ProjectLogitsInto(cfg, output, state.Scratch, &state.Scratch.Logits)
			applyRepeatPenalty(state.Scratch.Logits, state.DraftRecent, repeatPenalty)
			next = argmaxFiniteToken(state.Scratch.Logits)
		}
		state.DraftTokens = append(state.DraftTokens, next)
		copy(state.DraftHidden, state.Scratch.XN[:cfg.Dim])
		state.DraftRecent = append(state.DraftRecent, next)
		if len(state.DraftRecent) > repeatPenaltyWindow {
			copy(state.DraftRecent, state.DraftRecent[len(state.DraftRecent)-repeatPenaltyWindow:])
			state.DraftRecent = state.DraftRecent[:repeatPenaltyWindow]
		}
		input = next
	}
	return state.DraftTokens
}

// qwen35AttentionForward runs one of the periodic full-attention layers. Its
// doubled Q projection is split into a QK-normed/RoPE'd query and a per-head
// sigmoid gate applied before the O projection.
func qwen35AttentionForward(cfg Config, w Qwen35AttentionWeights, cache *KVCache, x []float32, layer, pos, ropeHalf, ropePairs int, buf *DecodeBuffer) {
	headDim := cfg.HeadDim
	nHeads := cfg.NHeads
	kvMul := max(1, cfg.KVMul)
	// Qwen3.8's Q4_K_M export uses Q4_K/Q4_K/Q6_K for Q/K/V.  The mixed
	// kernel shares the Q4 activation sums and one worker-pool dispatch across
	// all three projections (and uses the matching fused Metal path for
	// compatible shapes). It also retains the ordinary per-weight fallback for other
	// Qwen3.5/3.6/3.8 quantizations.
	tryMatvecAttentionInto(w.Q, w.K, w.V, x, &buf.Q4KXSums, &buf.QGate, &buf.K, &buf.V)
	if len(buf.QGate) < nHeads*2*headDim {
		panic("Qwen hybrid attention: attn_q projection has an invalid shape")
	}
	ensureLenNoClear(&buf.Q, nHeads*headDim)
	ensureLenNoClear(&buf.AttnGate, nHeads*headDim)
	for h := range nHeads {
		src := buf.QGate[h*2*headDim : h*2*headDim+2*headDim]
		copy(buf.Q[h*headDim:(h+1)*headDim], src[:headDim])
		copy(buf.AttnGate[h*headDim:(h+1)*headDim], src[headDim:2*headDim])
	}
	perHeadRMSNormInPlace(buf.Q, headDim, nHeads, w.QNorm, cfg.RMSNormEps)
	perHeadRMSNormInPlace(buf.K, headDim, cfg.NKVHeads, w.KNorm, cfg.RMSNormEps)
	applyPreparedRope(buf.Q, headDim, nHeads, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, false)
	applyPreparedRope(buf.K, headDim, cfg.NKVHeads, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, false)
	cache.storeKV(layer, pos, buf.K, buf.V)
	clear(buf.AttnOut)
	scale := cfg.AttentionScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(headDim)))
	}
	// Qwen3.8-27B has 24 query heads but only four KV heads: each six-head
	// group reads the same K/V history. At a long context, stream that history
	// once per group and give the independent groups to the worker pool. The
	// short-context scalar path avoids scheduling overhead, while the grouped
	// kernel retains the exact Qwen attention parameters (notably softcap=0).
	if useGroupedGQAAttention && kvMul > 1 && cfg.NKVHeads > 1 && pos+1 >= groupedGQADecodeMinContext {
		qwen35ParallelAttendHeadGroups(cfg, cache, buf, layer, pos, scale, kvMul)
	} else {
		for h := range nHeads {
			qOff, outOff := h*headDim, h*cfg.ValueDim
			cache.attendHead(layer, h/kvMul, buf.Q[qOff:qOff+headDim], headDim, cfg.ValueDim, 0, pos, scale, 0, buf.AttnOut[outOff:outOff+cfg.ValueDim])
		}
	}
	for h := range nHeads {
		outOff := h * cfg.ValueDim
		gate := buf.AttnGate[h*headDim : h*headDim+cfg.ValueDim]
		out := buf.AttnOut[outOff : outOff+cfg.ValueDim]
		for d, g := range gate {
			out[d] *= fastSigmoidF32(g)
		}
	}
	w.O.MatvecInto(buf.AttnOut[:nHeads*cfg.ValueDim], &buf.Proj)
}

// qwen35ParallelAttendHeadGroups is the Qwen-specific long-context GQA path.
// Unlike the standard decoder it deliberately passes a zero softcap: Qwen's
// hybrid full-attention graph has no attention-logit softcap tensor, and its
// scalar fallback above has always used zero. Keep that semantic detail local
// instead of relying on Config.AttnLogitSoftcap remaining unset.
func qwen35ParallelAttendHeadGroups(cfg Config, cache *KVCache, buf *DecodeBuffer, layer, pos int, scale float32, kvMul int) {
	// Use the shared worker-pool scheduling policy from ordinary GQA. Keeping
	// the configured workers active also avoids a wake-up gap before Qwen's
	// comparatively large output projection.
	workItems := max(cfg.NKVHeads, min(numThreads(), cfg.NHeads))
	parallelChunks(workItems, func(kvStart, kvEnd int) {
		if kvStart >= cfg.NKVHeads {
			return
		}
		qwen35AttendHeadGroupsRange(&cfg, cache, buf, layer, pos, scale, kvMul, kvStart, min(kvEnd, cfg.NKVHeads))
	})
}

func qwen35AttendHeadGroupsRange(cfg *Config, cache *KVCache, buf *DecodeBuffer, layer, pos int, scale float32, kvMul, kvStart, kvEnd int) {
	headDim := cfg.HeadDim
	valueDim := cfg.ValueDim
	for kvH := kvStart; kvH < kvEnd; kvH++ {
		hStart := kvH * kvMul
		hEnd := min(hStart+kvMul, cfg.NHeads)
		if hStart >= hEnd {
			break
		}
		cache.attendHeadGroup(layer, kvH,
			buf.Q[hStart*headDim:hEnd*headDim], hEnd-hStart, headDim, valueDim,
			0, pos, scale, 0,
			buf.AttnOut[hStart*valueDim:hEnd*valueDim])
	}
}

// qwen35DeltaNetForward runs one Gated DeltaNet layer. It performs the
// reference recurrent Gated Delta Rule: convolve [Q|K|V], L2-normalize Q/K,
// decay the state, write beta*(V-recall) through K, then read through the
// scaled Q before gated RMSNorm and the output projection.
func qwen35DeltaNetForward(cfg Config, w Qwen35DeltaNetWeights, state *Qwen35Cache, x []float32, layer int, buf *DecodeBuffer) {
	dInner, dState, nHeads, nGroups := cfg.SSMInner, cfg.SSMState, cfg.SSMHeads, cfg.SSMGroups
	headDim := dInner / nHeads
	channels := dInner + 2*nGroups*dState

	// QKV and the output gate share the same input. For CPU-resident Q4_K
	// layers, compute them in one quantized worker-pool dispatch (and reuse
	// x-sums) instead of launching the gate projection after the recurrent
	// update. The gate is not data-dependent on that update, so moving its
	// projection here is exactly equivalent. Keep a large QKV projection on its
	// direct-Metal path: pairing it with the smaller CPU-favorable gate would
	// otherwise demote both projections to the CPU.
	qkvGateFused := !metalWeightUsesDirect(w.QKVConv.Metal) &&
		tryMatvec2Into(w.QKVConv, w.Gate, x, &buf.Q4KXSums, &buf.MambaIn, &buf.MambaZ)
	if !qkvGateFused {
		w.QKVConv.MatvecInto(x, &buf.MambaIn)
	}
	if len(buf.MambaIn) < channels {
		panic("Qwen hybrid Gated DeltaNet qkv projection has an invalid shape")
	}
	ensureLenNoClear(&buf.MambaConv, channels)
	copy(buf.MambaConv, buf.MambaIn[:channels])

	// Causal depthwise convolution, then SiLU over the Q|K|V channels.
	ensureLenNoClear(&buf.MambaKernel, cfg.SSMConv)
	// F32/F16/BF16 convolution weights are intentionally materialized by the
	// loader, including in out-of-core mode. Direct row views avoid a copy and
	// method dispatch for every channel; quantized/future kernels retain the
	// general RowInto fallback.
	f32Kernel := w.ConvKernel.F32
	for ch := range channels {
		kernel := buf.MambaKernel[:cfg.SSMConv]
		if len(f32Kernel) >= (ch+1)*cfg.SSMConv {
			kernel = f32Kernel[ch*cfg.SSMConv : (ch+1)*cfg.SSMConv]
		} else {
			w.ConvKernel.RowInto(ch, cfg.SSMConv, &buf.MambaKernel)
		}
		off := state.convOffset(layer, ch)
		v := float32(0)
		for k := 0; k < state.ConvLen; k++ {
			v += kernel[k] * state.Conv[off+k]
		}
		v += kernel[state.ConvLen] * buf.MambaConv[ch]
		if state.ConvLen > 0 {
			copy(state.Conv[off:off+state.ConvLen-1], state.Conv[off+1:off+state.ConvLen])
			state.Conv[off+state.ConvLen-1] = buf.MambaConv[ch]
		}
		buf.MambaConv[ch] = v * fastSigmoidF32(v)
	}

	ensureLenNoClear(&buf.MambaX, dInner)         // V, one vector per value head.
	ensureLenNoClear(&buf.MambaB, nGroups*dState) // Q, one vector per key group.
	ensureLenNoClear(&buf.MambaC, nGroups*dState) // K, one vector per key group.
	ensureLenNoClear(&buf.MambaY, dInner)
	keyDim := nGroups * dState
	copy(buf.MambaB, buf.MambaConv[:keyDim])
	copy(buf.MambaC, buf.MambaConv[keyDim:2*keyDim])
	copy(buf.MambaX, buf.MambaConv[2*keyDim:channels])

	// GGUF conversion keeps Q/K in key-head order and tiles the V-head stream
	// to match its broadcast layout. Normalize each shared Q/K once; Q carries
	// the 1/sqrt(d_k) DeltaNet attention scale after L2 normalization.
	qScale := float32(1 / math.Sqrt(float64(dState)))
	for group := range nGroups {
		qwen35L2NormalizeInPlace(buf.MambaB[group*dState:(group+1)*dState], cfg.RMSNormEps, qScale)
		qwen35L2NormalizeInPlace(buf.MambaC[group*dState:(group+1)*dState], cfg.RMSNormEps, 1)
	}

	ensureLenNoClear(&buf.MambaDT, nHeads)
	ensureLenNoClear(&buf.MambaBeta, nHeads)
	if !tryMatvec2Into(w.AlphaProj, w.BetaProj, x, &buf.Q4KXSums, &buf.MambaDT, &buf.MambaBeta) {
		w.AlphaProj.MatvecInto(x, &buf.MambaDT)
		w.BetaProj.MatvecInto(x, &buf.MambaBeta)
	}

	// Each DeltaNet head owns a disjoint state matrix and output range. On the
	// production 27B geometry a head updates a 128x128 matrix, so splitting
	// heads across the persistent worker pool is substantially coarser than an
	// ordinary small-row matvec and exposes useful CPU parallelism. Give every
	// head its own recall span; the former one-head scratch was necessarily
	// serial.
	ensureLenNoClear(&buf.MambaRecall, nHeads*headDim)
	updateHeads := func(start, end int) {
		for h := start; h < end; h++ {
			alpha := nemotronSoftplus(buf.MambaDT[h] + w.DTBias[h])
			decay := float32(math.Exp(float64(alpha * w.A[h])))
			beta := fastSigmoidF32(buf.MambaBeta[h])
			group := h % nGroups
			q := buf.MambaB[group*dState : group*dState+dState]
			k := buf.MambaC[group*dState : group*dState+dState]
			xv := buf.MambaX[h*headDim : h*headDim+headDim]
			recall := buf.MambaRecall[h*headDim : (h+1)*headDim]

			off := state.stateOffset(layer, h)
			for d := range headDim {
				row := state.State[off+d*headDim : off+d*headDim+headDim]
				ScaleF32(row, decay)
				recall[d] = DotF32(row, k)
			}
			for d := range headDim {
				row := state.State[off+d*headDim : off+d*headDim+headDim]
				delta := beta * (xv[d] - recall[d])
				AxpyF32(row, delta, k)
				buf.MambaY[h*headDim+d] = DotF32(row, q)
			}
		}
	}
	if nHeads >= 8 && headDim >= 64 {
		parallelChunks(nHeads, updateHeads)
	} else {
		updateHeads(0, nHeads)
	}

	// Qwen's Gated RMSNorm normalizes the recurrent output first, then applies
	// SiLU(z). Applying the gate before RMSNorm would mostly cancel its scale
	// and materially changes the reference graph.
	for h := range nHeads {
		part := buf.MambaY[h*headDim : h*headDim+headDim]
		ss := DotF32(part, part)
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+cfg.RMSNormEps)))
		mulScaleF32(part, w.Norm[:headDim], scale, part)
	}
	if !qkvGateFused {
		ensureLenNoClear(&buf.MambaZ, dInner)
		w.Gate.MatvecInto(x, &buf.MambaZ)
	}
	for i, g := range buf.MambaZ[:dInner] {
		buf.MambaY[i] *= g * fastSigmoidF32(g)
	}
	w.Out.MatvecInto(buf.MambaY, &buf.Proj)
}

// qwen35L2NormalizeInPlace applies the Qwen/FLA L2 normalization used by the
// DeltaNet kernel. postScale is 1 for K and 1/sqrt(d_k) for Q.
func qwen35L2NormalizeInPlace(v []float32, eps, postScale float32) {
	if len(v) == 0 {
		return
	}
	ss := DotF32(v, v)
	scale := postScale * float32(1/math.Sqrt(float64(ss+eps)))
	ScaleF32(v, scale)
}

func qwen35FFNForward(w Qwen35FFNWeights, x []float32, buf *DecodeBuffer) {
	// Qwen3.8-27B-Q4_K_M stores every dense FFN as Q4_K gate/up and Q6_K
	// down. Keep those three stages inside the existing Metal command buffer
	// when it is available; otherwise the fused CPU gate/up kernel below is
	// still used. This is intentionally checked before allocating Hidden.
	if matvecMetalSwiGLUInto(w.Gate.Metal, w.Up.Metal, w.Down.Metal, x, &buf.Proj) {
		return
	}
	if !tryMatvec2Into(w.Gate, w.Up, x, &buf.Q4KXSums, &buf.Gate, &buf.Up) {
		w.Gate.MatvecInto(x, &buf.Gate)
		w.Up.MatvecInto(x, &buf.Up)
	}
	hDim := len(buf.Gate)
	ensureLenNoClear(&buf.Hidden, hDim)
	siluMulF32(buf.Gate[:hDim], buf.Up[:hDim], buf.Hidden[:hDim])
	w.Down.MatvecInto(buf.Hidden, &buf.Proj)
}

// releaseQwen35MetalWeights releases every optional Metal backing buffer owned
// by the hybrid graph, mirroring releaseNemotronHMetalWeights.
func releaseQwen35MetalWeights(weights *Qwen35Weights) {
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
	if weights.MTP != nil {
		mtp := weights.MTP
		release(&mtp.TokenEmbd)
		release(&mtp.EHProj)
		release(&mtp.Attention.Q)
		release(&mtp.Attention.K)
		release(&mtp.Attention.V)
		release(&mtp.Attention.O)
		release(&mtp.FFN.Gate)
		release(&mtp.FFN.Up)
		release(&mtp.FFN.Down)
		release(&mtp.Output)
	}
	for i := range weights.Layers {
		layer := &weights.Layers[i]
		release(&layer.Attention.Q)
		release(&layer.Attention.K)
		release(&layer.Attention.V)
		release(&layer.Attention.O)
		release(&layer.DeltaNet.QKVConv)
		release(&layer.DeltaNet.ConvKernel)
		release(&layer.DeltaNet.Gate)
		release(&layer.DeltaNet.AlphaProj)
		release(&layer.DeltaNet.BetaProj)
		release(&layer.DeltaNet.Out)
		release(&layer.FFN.Gate)
		release(&layer.FFN.Up)
		release(&layer.FFN.Down)
		if layer.FFN.MoE != nil {
			moe := layer.FFN.MoE
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
