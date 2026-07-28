package gopherllm

// Native Qwen3.5 / Qwen3.6 ("qwen35") inference.
//
// EXPERIMENTAL: this is a native, text-only single-token decode path for the
// Qwen3.5/3.6 hybrid architecture. Its DeltaNet and gated-attention math
// follows the Qwen reference graph and the llama.cpp GGUF conversion layout:
// QKV is [Q|K|V], Q/K use L2 normalization, and full attention uses a sigmoid
// gate. The implementation has focused unit coverage, but does not yet offer
// cross-runtime logit-parity validation, chunked DeltaNet prefill, multimodal
// vision/3D-MRoPE, or MTP speculative decoding.
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
		return cfg, Qwen35Weights{}, fmt.Errorf("not a Qwen3.5 GGUF: %s", cfg.Arch)
	}
	tensors := indexTensors(gguf)
	// Some Qwen3.6 exports append MTP (multi-token prediction) draft blocks
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
	if skippedMTP > 0 {
		fmt.Fprintf(logw, "  Skipping %d Qwen MTP draft layer(s); speculative decoding is not enabled\n", skippedMTP)
	}
	inferred := inferTensorSizes(data, gguf)
	lazyScalarWeights := len(outOfCore) > 0 && outOfCore[0]
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
	}
	// The depthwise DeltaNet convolution is tiny relative to the checkpoint but
	// runs once per channel for every recurrent layer and generated token. Keep
	// scalar kernels hot even for an out-of-core model: a Qwen3.6-27B kernel
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
		panic("Qwen3.5 forward requires a recurrent cache")
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

// qwen35AttentionForward runs one of the periodic full-attention layers. Its
// doubled Q projection is split into a QK-normed/RoPE'd query and a per-head
// sigmoid gate applied before the O projection.
func qwen35AttentionForward(cfg Config, w Qwen35AttentionWeights, cache *KVCache, x []float32, layer, pos, ropeHalf, ropePairs int, buf *DecodeBuffer) {
	headDim := cfg.HeadDim
	nHeads := cfg.NHeads
	kvMul := max(1, cfg.KVMul)
	// Q and K share their input and are commonly the same quantization in
	// Qwen3.6 GGUFs (including the local IQ variants). Fuse their row work
	// when possible; V remains separate because its format often differs.
	if !tryMatvec2Into(w.Q, w.K, x, &buf.Q4KXSums, &buf.QGate, &buf.K) {
		w.Q.MatvecInto(x, &buf.QGate)
		w.K.MatvecInto(x, &buf.K)
	}
	if len(buf.QGate) < nHeads*2*headDim {
		panic("Qwen3.5 attention: attn_q projection has an invalid shape")
	}
	ensureLenNoClear(&buf.Q, nHeads*headDim)
	ensureLenNoClear(&buf.AttnGate, nHeads*headDim)
	for h := range nHeads {
		src := buf.QGate[h*2*headDim : h*2*headDim+2*headDim]
		copy(buf.Q[h*headDim:(h+1)*headDim], src[:headDim])
		copy(buf.AttnGate[h*headDim:(h+1)*headDim], src[headDim:2*headDim])
	}
	w.V.MatvecInto(x, &buf.V)
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
	for h := range nHeads {
		qOff, outOff := h*headDim, h*cfg.ValueDim
		cache.attendHead(layer, h/kvMul, buf.Q[qOff:qOff+headDim], headDim, cfg.ValueDim, 0, pos, scale, 0, buf.AttnOut[outOff:outOff+cfg.ValueDim])
		gate := buf.AttnGate[h*headDim : h*headDim+cfg.ValueDim]
		out := buf.AttnOut[outOff : outOff+cfg.ValueDim]
		for d, g := range gate {
			out[d] *= 1 / (1 + float32(math.Exp(float64(-g))))
		}
	}
	w.O.MatvecInto(buf.AttnOut[:nHeads*cfg.ValueDim], &buf.Proj)
}

// qwen35DeltaNetForward runs one Gated DeltaNet layer. It performs the
// reference recurrent Gated Delta Rule: convolve [Q|K|V], L2-normalize Q/K,
// decay the state, write beta*(V-recall) through K, then read through the
// scaled Q before gated RMSNorm and the output projection.
func qwen35DeltaNetForward(cfg Config, w Qwen35DeltaNetWeights, state *Qwen35Cache, x []float32, layer int, buf *DecodeBuffer) {
	dInner, dState, nHeads, nGroups := cfg.SSMInner, cfg.SSMState, cfg.SSMHeads, cfg.SSMGroups
	headDim := dInner / nHeads
	channels := dInner + 2*nGroups*dState

	w.QKVConv.MatvecInto(x, &buf.MambaIn)
	if len(buf.MambaIn) < channels {
		panic("Qwen3.5 Gated DeltaNet qkv projection has an invalid shape")
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
		buf.MambaConv[ch] = v / (1 + float32(math.Exp(float64(-v))))
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
	w.AlphaProj.MatvecInto(x, &buf.MambaDT)
	w.BetaProj.MatvecInto(x, &buf.MambaBeta)

	ensureLenNoClear(&buf.MambaRecall, headDim)
	recall := buf.MambaRecall[:headDim]
	for h := range nHeads {
		alpha := nemotronSoftplus(buf.MambaDT[h] + w.DTBias[h])
		decay := float32(math.Exp(float64(alpha * w.A[h])))
		beta := nemotronSigmoid(buf.MambaBeta[h])
		group := h % nGroups
		q := buf.MambaB[group*dState : group*dState+dState]
		k := buf.MambaC[group*dState : group*dState+dState]
		xv := buf.MambaX[h*headDim : h*headDim+headDim]

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

	// Qwen's Gated RMSNorm normalizes the recurrent output first, then applies
	// SiLU(z). Applying the gate before RMSNorm would mostly cancel its scale
	// and materially changes the reference graph.
	for h := range nHeads {
		part := buf.MambaY[h*headDim : h*headDim+headDim]
		ss := DotF32(part, part)
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+cfg.RMSNormEps)))
		mulScaleF32(part, w.Norm[:headDim], scale, part)
	}
	ensureLenNoClear(&buf.MambaZ, dInner)
	w.Gate.MatvecInto(x, &buf.MambaZ)
	for i, g := range buf.MambaZ[:dInner] {
		buf.MambaY[i] *= g / (1 + float32(math.Exp(float64(-g))))
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
