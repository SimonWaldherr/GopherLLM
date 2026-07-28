package gopherllm

// Native Qwen3.5 / Qwen3.6 ("qwen35") inference.
//
// EXPERIMENTAL AND KNOWN-UNVALIDATED: this is a from-scratch reconstruction
// written without a reference implementation (no internet access was
// available while writing it, and none was supplied). It loads real
// Qwen3.5-9B / Qwen3.6-27B GGUFs, every tensor shape it asserts against
// matches the file, and generation runs to completion with finite,
// reasonably-scaled activations at every layer (no NaN/Inf, no runaway
// magnitudes) — but the generated text is not yet coherent. Several
// combinations of plausible alternatives (Gated-DeltaNet channel order,
// disabling the delta-rule correction entirely, swapping which half of the
// attention layers' doubled attn_q width is the query vs. the gate) were
// tried empirically against a real Qwen3.5-9B GGUF and none produced
// materially different (let alone better) output, which narrows the bug
// somewhat but does not localize it. Treat this file as a documented,
// falsifiable hypothesis to correct against a real reference (the Qwen3-Next
// tech report, or an llama.cpp/vLLM implementation of qwen3_next/qwen3.5),
// not as validated inference.
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

type Qwen35AttentionWeights struct {
	Q, K, V, O   Weight
	QNorm, KNorm []float32
}

// Qwen35DeltaNetWeights holds one Gated DeltaNet layer's projections. QKVConv
// projects the normalized hidden state to the concatenated [X | B | C] input
// of the causal short convolution (X is the per-head "value" stream, B/C are
// the per-group "key"/"query" streams that write into and read out of the
// recurrent state — Mamba-2 naming, despite the GGUF's "attn_" tensor
// prefixes). Gate/AlphaProj/BetaProj read directly from the un-convolved
// hidden state, matching how nemotron_h's dt/z are separate from its own
// conv'd stream.
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
	Attention    Qwen35AttentionWeights
	DeltaNet     Qwen35DeltaNetWeights
	FFN          Qwen35FFNWeights
}

type Qwen35Weights struct {
	TokenEmbd  Weight
	OutputNorm []float32
	Output     Weight
	Layers     []Qwen35LayerWeights
}

// Qwen35Cache holds the Gated DeltaNet recurrent state (one HeadDim x HeadDim
// matrix per head per layer — the delta rule's associative memory) plus the
// short causal-conv history. Attention-kind layers use the same *KVCache's
// ordinary K/V rows, exactly like Nemotron-H's dual-cache pattern.
type Qwen35Cache struct {
	Conv     []float32
	State    []float32
	Layers   int
	Channels int
	ConvLen  int
	Heads    int
	HeadDim  int
}

func newQwen35Cache(c Config) *Qwen35Cache {
	channels := c.SSMInner + 2*c.SSMGroups*c.SSMState
	headDim := 0
	if c.SSMHeads > 0 {
		headDim = c.SSMInner / c.SSMHeads
	}
	cache := &Qwen35Cache{
		Layers: c.NLayers, Channels: channels, ConvLen: max(0, c.SSMConv-1),
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

func (c *Qwen35Cache) compatible(cfg Config) bool {
	if c == nil {
		return false
	}
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	headDim := 0
	if cfg.SSMHeads > 0 {
		headDim = cfg.SSMInner / cfg.SSMHeads
	}
	return c.Layers == cfg.NLayers && c.Channels == channels && c.ConvLen == max(0, cfg.SSMConv-1) &&
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
	// Some Qwen3.6 exports append one or more MTP (multi-token prediction)
	// draft layers to the ordinary decoder stack. They are marked by nextn.*
	// tensors and are useful only to a speculative-decoding implementation;
	// normal autoregressive inference must stop at the preceding base layer.
	// Treating one as a DeltaNet layer otherwise produces an unhelpful missing
	// attn_qkv error on an otherwise usable text model.
	skippedMTP := 0
	for cfg.NLayers > 1 {
		last := cfg.NLayers - 1
		if _, ok := tensors[fmt.Sprintf("blk.%d.nextn.eh_proj.weight", last)]; !ok {
			break
		}
		cfg.NLayers--
		skippedMTP++
	}
	if cfg.Dim <= 0 || cfg.NLayers <= 0 || cfg.SSMConv <= 0 || cfg.SSMInner <= 0 || cfg.SSMState <= 0 ||
		cfg.SSMHeads <= 0 || cfg.SSMGroups <= 0 || cfg.FullAttentionInterval <= 0 {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: incomplete Gated DeltaNet metadata", cfg.Arch)
	}
	if cfg.SSMInner%cfg.SSMHeads != 0 || cfg.SSMHeads%cfg.SSMGroups != 0 {
		return cfg, Qwen35Weights{}, fmt.Errorf("%s: invalid DeltaNet dimensions inner=%d heads=%d groups=%d", cfg.Arch, cfg.SSMInner, cfg.SSMHeads, cfg.SSMGroups)
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
		layer := Qwen35LayerWeights{Norm: norm, PostAttnNorm: postNorm}
		if (i+1)%cfg.FullAttentionInterval == 0 {
			layer.Kind = qwen35Attention
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
			layer.DeltaNet.QKVConv, err = load(prefix + "attn_qkv.weight")
			if err == nil {
				layer.DeltaNet.ConvKernel, err = load(prefix + "ssm_conv1d.weight")
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
			fmt.Fprintf(logw, "  Loaded Qwen3.5 layer %d/%d\n", i+1, cfg.NLayers)
		}
	}
	return cfg, weights, nil
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
	for i, layer := range weights.Layers {
		rmsNormInto(buf.X, layer.Norm, cfg.RMSNormEps, &buf.XN)
		switch layer.Kind {
		case qwen35Attention:
			qwen35AttentionForward(cfg, layer.Attention, cache, buf.XN, i, pos, ropeHalf, ropePairs, buf)
		default:
			qwen35DeltaNetForward(cfg, layer.DeltaNet, cache.Qwen35, buf.XN, i, buf)
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

// qwen35AttentionForward runs one of the periodic full-attention layers. See
// the file header for why attn_q's doubled width is split into a query half
// (QK-normed and RoPE'd) and a gate half (a plain per-head sigmoid on that
// head's attention output before the O projection).
func qwen35AttentionForward(cfg Config, w Qwen35AttentionWeights, cache *KVCache, x []float32, layer, pos, ropeHalf, ropePairs int, buf *DecodeBuffer) {
	headDim := cfg.HeadDim
	nHeads := cfg.NHeads
	kvMul := max(1, cfg.KVMul)
	w.Q.MatvecInto(x, &buf.QGate)
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
	w.K.MatvecInto(x, &buf.K)
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
			out[d] *= g / (1 + float32(math.Exp(float64(-g))))
		}
	}
	w.O.MatvecInto(buf.AttnOut[:nHeads*cfg.ValueDim], &buf.Proj)
}

// qwen35DeltaNetForward runs one Gated DeltaNet layer. See the file header for
// the reconstructed recurrence; in one sentence: decay the existing per-head
// state, subtract what it already recalls for this token's key from the true
// value (the "delta"), and write beta times that residual back in — then
// read the state out via this token's query, gate with SiLU, and RMS-norm
// per head before the output projection.
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

	// Causal depthwise convolution, then SiLU — same short-conv preprocessing
	// Mamba-2 applies to its combined x/B/C stream before the recurrence.
	ensureLenNoClear(&buf.MambaKernel, cfg.SSMConv)
	for ch := range channels {
		w.ConvKernel.RowInto(ch, cfg.SSMConv, &buf.MambaKernel)
		off := state.convOffset(layer, ch)
		v := float32(0)
		for k := 0; k < state.ConvLen; k++ {
			v += buf.MambaKernel[k] * state.Conv[off+k]
		}
		v += buf.MambaKernel[state.ConvLen] * buf.MambaConv[ch]
		if state.ConvLen > 0 {
			copy(state.Conv[off:off+state.ConvLen-1], state.Conv[off+1:off+state.ConvLen])
			state.Conv[off+state.ConvLen-1] = buf.MambaConv[ch]
		}
		buf.MambaConv[ch] = v / (1 + float32(math.Exp(float64(-v))))
	}

	ensureLenNoClear(&buf.MambaX, dInner)
	ensureLenNoClear(&buf.MambaB, nGroups*dState)
	ensureLenNoClear(&buf.MambaC, nGroups*dState)
	ensureLenNoClear(&buf.MambaY, dInner)
	bcDim := nGroups * dState
	copy(buf.MambaX, buf.MambaConv[:dInner])
	copy(buf.MambaB, buf.MambaConv[dInner:dInner+bcDim])
	copy(buf.MambaC, buf.MambaConv[dInner+bcDim:channels])

	ensureLenNoClear(&buf.MambaDT, nHeads)
	ensureLenNoClear(&buf.MambaBeta, nHeads)
	w.AlphaProj.MatvecInto(x, &buf.MambaDT)
	w.BetaProj.MatvecInto(x, &buf.MambaBeta)

	ensureLenNoClear(&buf.MambaRecall, headDim)
	recall := buf.MambaRecall[:headDim]
	groupSize := nHeads / nGroups
	for h := range nHeads {
		dt := nemotronSoftplus(buf.MambaDT[h] + w.DTBias[h])
		decay := float32(math.Exp(float64(dt * w.A[h])))
		beta := nemotronSigmoid(buf.MambaBeta[h])
		group := h / groupSize
		k := buf.MambaB[group*dState : group*dState+dState]
		q := buf.MambaC[group*dState : group*dState+dState]
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

	ensureLenNoClear(&buf.MambaZ, dInner)
	w.Gate.MatvecInto(x, &buf.MambaZ)
	for i, g := range buf.MambaZ[:dInner] {
		buf.MambaY[i] *= g / (1 + float32(math.Exp(float64(-g))))
	}
	for h := range nHeads {
		part := buf.MambaY[h*headDim : h*headDim+headDim]
		ss := DotF32(part, part)
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+cfg.RMSNormEps)))
		mulScaleF32(part, w.Norm[:headDim], scale, part)
	}
	w.Out.MatvecInto(buf.MambaY, &buf.Proj)
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
