package gopherllm

// Native Nemotron-H-MoE / Soofi S inference.
//
// Soofi Isar is a hybrid sequence model: individual residual blocks are
// either Mamba-2 recurrence, conventional (position-free) attention, or a
// sparse ReLU² MoE. Keeping this graph separate from the llama-style block is
// deliberate: its layer schedule and recurrent cache are fundamentally
// different from a transformer FFN-after-attention block.

import (
	"fmt"
	"io"
	"math"
)

type nemotronLayerKind uint8

const (
	nemotronMamba nemotronLayerKind = iota
	nemotronAttention
	nemotronMoE
)

type NemotronAttentionWeights struct {
	Q Weight
	K Weight
	V Weight
	O Weight
}

type NemotronMambaWeights struct {
	In       Weight
	Conv     Weight
	ConvBias []float32
	DtBias   []float32
	A        []float32
	D        []float32
	Norm     []float32
	Out      Weight
}

// ExpertWeight retains the third GGUF tensor dimension. Weight itself stores
// two-dimensional matrices, while the expert tensors are laid out as
// [input, output, expert].
type ExpertWeight struct {
	Weight  Weight
	Input   int
	Output  int
	Experts int
}

type NemotronMoEWeights struct {
	Router     Weight
	RouterBias []float32
	Up         ExpertWeight
	Down       ExpertWeight
	LatentIn   *Weight
	LatentOut  *Weight
	SharedUp   *Weight
	SharedDown *Weight
}

type NemotronHLayerWeights struct {
	Norm      []float32
	Kind      nemotronLayerKind
	Attention NemotronAttentionWeights
	Mamba     NemotronMambaWeights
	MoE       NemotronMoEWeights
}

type NemotronHWeights struct {
	TokenEmbd  Weight
	OutputNorm []float32
	Output     Weight
	Layers     []NemotronHLayerWeights
}

// NemotronHCache stores the recurrent Mamba-2 state. Attention K/V continues
// to use KVCache so the existing f16 cache and online-attention kernels apply
// unchanged. Conv is [layer, channel, d_conv-1]; State is [layer, head,
// head_dim, d_state].
type NemotronHCache struct {
	Conv      []float32
	State     []float32
	Layers    int
	Channels  int
	ConvLen   int
	Heads     int
	HeadDim   int
	StateSize int
}

func newNemotronHCache(c Config) *NemotronHCache {
	channels := c.SSMInner + 2*c.SSMGroups*c.SSMState
	headDim := 0
	if c.SSMHeads > 0 {
		headDim = c.SSMInner / c.SSMHeads
	}
	cache := &NemotronHCache{
		Layers: c.NLayers, Channels: channels, ConvLen: max(0, c.SSMConv-1),
		Heads: c.SSMHeads, HeadDim: headDim, StateSize: c.SSMState,
	}
	if cache.ConvLen > 0 && channels > 0 {
		cache.Conv = make([]float32, cache.Layers*channels*cache.ConvLen)
	}
	if c.SSMState > 0 && c.SSMHeads > 0 && headDim > 0 {
		cache.State = make([]float32, cache.Layers*c.SSMHeads*headDim*c.SSMState)
	}
	return cache
}

func (c *NemotronHCache) compatible(cfg Config) bool {
	if c == nil {
		return false
	}
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	headDim := 0
	if cfg.SSMHeads > 0 {
		headDim = cfg.SSMInner / cfg.SSMHeads
	}
	return c.Layers == cfg.NLayers && c.Channels == channels && c.ConvLen == max(0, cfg.SSMConv-1) &&
		c.Heads == cfg.SSMHeads && c.HeadDim == headDim && c.StateSize == cfg.SSMState
}

func (c *NemotronHCache) reset() {
	clear(c.Conv)
	clear(c.State)
}

func (c *NemotronHCache) convOffset(layer, channel int) int {
	return (layer*c.Channels + channel) * c.ConvLen
}

func (c *NemotronHCache) stateOffset(layer, head, dim int) int {
	return (((layer*c.Heads)+head)*c.HeadDim + dim) * c.StateSize
}

func LoadNemotronHModel(data []byte, gguf *GGUFFile, borrow, prepareQuantized, useMetal bool, logw io.Writer) (Config, NemotronHWeights, error) {
	if logw == nil {
		logw = io.Discard
	}
	cfg := ConfigFromGGUF(gguf)
	if cfg.Arch != "nemotron_h" && cfg.Arch != "nemotron_h_moe" {
		return cfg, NemotronHWeights{}, fmt.Errorf("not a Nemotron-H GGUF: %s", cfg.Arch)
	}
	if cfg.Dim <= 0 || cfg.NLayers <= 0 || cfg.SSMConv <= 0 || cfg.SSMInner <= 0 || cfg.SSMState <= 0 || cfg.SSMHeads <= 0 || cfg.SSMGroups <= 0 {
		return cfg, NemotronHWeights{}, fmt.Errorf("%s: incomplete Mamba-2 metadata", cfg.Arch)
	}
	if cfg.SSMInner%cfg.SSMHeads != 0 || cfg.SSMHeads%cfg.SSMGroups != 0 {
		return cfg, NemotronHWeights{}, fmt.Errorf("%s: invalid Mamba dimensions inner=%d heads=%d groups=%d", cfg.Arch, cfg.SSMInner, cfg.SSMHeads, cfg.SSMGroups)
	}
	if cfg.ExpertCount <= 0 || cfg.ExpertUsedCount <= 0 || cfg.ExpertUsedCount > cfg.ExpertCount {
		return cfg, NemotronHWeights{}, fmt.Errorf("%s: invalid MoE metadata experts=%d used=%d", cfg.Arch, cfg.ExpertCount, cfg.ExpertUsedCount)
	}
	if len(cfg.LayerKVHeads) != cfg.NLayers || len(cfg.LayerFFNDim) != cfg.NLayers {
		return cfg, NemotronHWeights{}, fmt.Errorf("%s: per-layer attention.head_count_kv and feed_forward_length metadata are required", cfg.Arch)
	}
	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal)
	}
	loadVec := func(name string) ([]float32, error) {
		return loadF32Vec(data, gguf.DataOffset, name, tensors, inferred)
	}
	optionalWeight := func(name string) (*Weight, error) {
		if _, ok := tensors[name]; !ok {
			return nil, nil
		}
		w, err := load(name)
		if err != nil {
			return nil, err
		}
		return &w, nil
	}
	token, err := load("token_embd.weight")
	if err != nil {
		return cfg, NemotronHWeights{}, err
	}
	outNorm, err := loadVec("output_norm.weight")
	if err != nil {
		return cfg, NemotronHWeights{}, err
	}
	out := token
	if _, ok := tensors["output.weight"]; ok {
		out, err = load("output.weight")
		if err != nil {
			return cfg, NemotronHWeights{}, err
		}
	}
	weights := NemotronHWeights{TokenEmbd: token, OutputNorm: outNorm, Output: out, Layers: make([]NemotronHLayerWeights, cfg.NLayers)}
	for i := range cfg.NLayers {
		prefix := fmt.Sprintf("blk.%d.", i)
		norm, err := loadVec(prefix + "attn_norm.weight")
		if err != nil {
			return cfg, NemotronHWeights{}, err
		}
		layer := NemotronHLayerWeights{Norm: norm}
		kvHeads, ffnDim := cfg.LayerKVHeads[i], cfg.LayerFFNDim[i]
		switch {
		case kvHeads == 0 && ffnDim == 0:
			layer.Kind = nemotronMamba
			layer.Mamba.In, err = load(prefix + "ssm_in.weight")
			if err == nil {
				layer.Mamba.Conv, err = load(prefix + "ssm_conv1d.weight")
			}
			if err == nil {
				layer.Mamba.DtBias, err = loadVec(prefix + "ssm_dt.bias")
			}
			if err == nil {
				layer.Mamba.A, err = loadVec(prefix + "ssm_a")
			}
			if err == nil {
				layer.Mamba.D, err = loadVec(prefix + "ssm_d")
			}
			if err == nil {
				layer.Mamba.Norm, err = loadVec(prefix + "ssm_norm.weight")
			}
			if err == nil {
				layer.Mamba.Out, err = load(prefix + "ssm_out.weight")
			}
			if err != nil {
				return cfg, NemotronHWeights{}, fmt.Errorf("layer %d Mamba-2: %w", i, err)
			}
			layer.Mamba.ConvBias = loadOptionalF32Vec(data, gguf.DataOffset, prefix+"ssm_conv1d.bias", tensors, inferred, cfg.SSMInner+2*cfg.SSMGroups*cfg.SSMState)
		case kvHeads > 0 && ffnDim == 0:
			layer.Kind = nemotronAttention
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
			if err != nil {
				return cfg, NemotronHWeights{}, fmt.Errorf("layer %d attention: %w", i, err)
			}
		case ffnDim > 0:
			layer.Kind = nemotronMoE
			layer.MoE.Router, err = load(prefix + "ffn_gate_inp.weight")
			if err == nil {
				layer.MoE.Up, err = loadExpertWeight(data, gguf.DataOffset, prefix+"ffn_up_exps.weight", tensors, inferred, borrow, prepareQuantized, useMetal)
			}
			if err == nil {
				layer.MoE.Down, err = loadExpertWeight(data, gguf.DataOffset, prefix+"ffn_down_exps.weight", tensors, inferred, borrow, prepareQuantized, useMetal)
			}
			if err != nil {
				return cfg, NemotronHWeights{}, fmt.Errorf("layer %d MoE: %w", i, err)
			}
			layer.MoE.RouterBias = loadOptionalF32Vec(data, gguf.DataOffset, prefix+"exp_probs_b.bias", tensors, inferred, cfg.ExpertCount)
			if len(layer.MoE.RouterBias) == 0 {
				layer.MoE.RouterBias = loadOptionalF32Vec(data, gguf.DataOffset, prefix+"exp_probs_b", tensors, inferred, cfg.ExpertCount)
			}
			if layer.MoE.LatentIn, err = optionalWeight(prefix + "ffn_latent_in.weight"); err != nil {
				return cfg, NemotronHWeights{}, err
			}
			if layer.MoE.LatentOut, err = optionalWeight(prefix + "ffn_latent_out.weight"); err != nil {
				return cfg, NemotronHWeights{}, err
			}
			if layer.MoE.SharedUp, err = optionalWeight(prefix + "ffn_up_shexp.weight"); err != nil {
				return cfg, NemotronHWeights{}, err
			}
			if layer.MoE.SharedDown, err = optionalWeight(prefix + "ffn_down_shexp.weight"); err != nil {
				return cfg, NemotronHWeights{}, err
			}
		default:
			return cfg, NemotronHWeights{}, fmt.Errorf("layer %d: invalid Nemotron-H schedule kv_heads=%d ffn=%d", i, kvHeads, ffnDim)
		}
		weights.Layers[i] = layer
		if i == 0 || (i+1)%8 == 0 || i+1 == cfg.NLayers {
			fmt.Fprintf(logw, "  Loaded Nemotron-H layer %d/%d\n", i+1, cfg.NLayers)
		}
	}
	// Attention metadata is per-layer in Nemotron-H. Derive the dimensions
	// from an actual attention projection; Mamba-only layers deliberately have
	// zero KV heads and must not influence the cache shape.
	for i, layer := range weights.Layers {
		if layer.Kind != nemotronAttention {
			continue
		}
		qInfo := tensors[fmt.Sprintf("blk.%d.attn_q.weight", i)]
		kInfo := tensors[fmt.Sprintf("blk.%d.attn_k.weight", i)]
		vInfo := tensors[fmt.Sprintf("blk.%d.attn_v.weight", i)]
		if len(qInfo.Dims) < 2 || len(kInfo.Dims) < 2 || len(vInfo.Dims) < 2 {
			return cfg, NemotronHWeights{}, fmt.Errorf("layer %d: malformed attention dimensions", i)
		}
		heads := 0
		if i < len(cfg.LayerHeads) {
			heads = cfg.LayerHeads[i]
		}
		if heads <= 0 {
			return cfg, NemotronHWeights{}, fmt.Errorf("layer %d: missing attention head count", i)
		}
		headDim := int(qInfo.Dims[1]) / heads
		kvHeads := cfg.LayerKVHeads[i]
		if headDim <= 0 || kvHeads <= 0 || int(kInfo.Dims[1]) != kvHeads*headDim || int(vInfo.Dims[1])%kvHeads != 0 {
			return cfg, NemotronHWeights{}, fmt.Errorf("layer %d: inconsistent attention dimensions", i)
		}
		cfg.NHeads, cfg.NKVHeads, cfg.HeadDim = heads, kvHeads, headDim
		cfg.ValueDim = int(vInfo.Dims[1]) / kvHeads
		cfg.KVDim, cfg.KVMul = cfg.NKVHeads*cfg.ValueDim, max(1, cfg.NHeads/cfg.NKVHeads)
		break
	}
	return cfg, weights, nil
}

func loadExpertWeight(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, borrow, prepareQuantized, useMetal bool) (ExpertWeight, error) {
	info, ok := tensors[name]
	if !ok {
		return ExpertWeight{}, fmt.Errorf("missing tensor: %s", name)
	}
	if len(info.Dims) != 3 || info.Dims[0] == 0 || info.Dims[1] == 0 || info.Dims[2] == 0 {
		return ExpertWeight{}, fmt.Errorf("tensor %s must have [input, output, expert] dimensions, got %v", name, info.Dims)
	}
	w, err := loadWeight(data, dataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal)
	if err != nil {
		return ExpertWeight{}, err
	}
	return ExpertWeight{Weight: w, Input: int(info.Dims[0]), Output: int(info.Dims[1]), Experts: int(info.Dims[2])}, nil
}

func nemotronSoftplus(x float32) float32 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return float32(math.Exp(float64(x)))
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func nemotronSigmoid(x float32) float32 {
	if x >= 0 {
		return 1 / (1 + float32(math.Exp(float64(-x))))
	}
	e := float32(math.Exp(float64(x)))
	return e / (1 + e)
}

func ForwardNemotronHInto(cfg Config, weights NemotronHWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardNemotronHBodyInto(cfg, weights, cache, buf, token, pos)
	weights.Output.MatvecInto(buf.XN, logits)
}

func ForwardNemotronHBodyInto(cfg Config, weights NemotronHWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	if cache == nil || cache.Nemotron == nil {
		panic("Nemotron-H forward requires a recurrent cache")
	}
	weights.TokenEmbd.RowInto(int(token), cfg.Dim, &buf.X)
	for i, layer := range weights.Layers {
		rmsNormInto(buf.X, layer.Norm, cfg.RMSNormEps, &buf.XN)
		switch layer.Kind {
		case nemotronMamba:
			nemotronMambaForward(cfg, layer.Mamba, cache.Nemotron, buf.XN, i, buf)
		case nemotronAttention:
			nemotronAttentionForward(cfg, layer.Attention, cache, buf.XN, i, pos, buf)
		case nemotronMoE:
			nemotronMoEForward(cfg, layer.MoE, buf.XN, buf)
		default:
			panic("unknown Nemotron-H layer kind")
		}
		addInPlace(buf.X[:cfg.Dim], buf.Proj[:cfg.Dim])
	}
	rmsNormInto(buf.X, weights.OutputNorm, cfg.RMSNormEps, &buf.XN)
}

func nemotronAttentionForward(cfg Config, w NemotronAttentionWeights, cache *KVCache, x []float32, layer, pos int, buf *DecodeBuffer) {
	w.Q.MatvecInto(x, &buf.Q)
	w.K.MatvecInto(x, &buf.K)
	w.V.MatvecInto(x, &buf.V)
	headDim := cfg.HeadDim
	cache.storeKV(layer, pos, buf.K, buf.V)
	clear(buf.AttnOut)
	scale := cfg.AttentionScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(headDim)))
	}
	kvMul := max(1, cfg.KVMul)
	for h := range cfg.NHeads {
		qOff, outOff := h*headDim, h*cfg.ValueDim
		cache.attendHead(layer, h/kvMul, buf.Q[qOff:qOff+headDim], headDim, cfg.ValueDim, 0, pos, scale, 0, buf.AttnOut[outOff:outOff+cfg.ValueDim])
	}
	w.O.MatvecInto(buf.AttnOut[:cfg.NHeads*cfg.ValueDim], &buf.Proj)
}

func nemotronMambaForward(cfg Config, w NemotronMambaWeights, state *NemotronHCache, x []float32, layer int, buf *DecodeBuffer) {
	dInner, dState, nHeads, nGroups := cfg.SSMInner, cfg.SSMState, cfg.SSMHeads, cfg.SSMGroups
	headDim := dInner / nHeads
	channels := dInner + 2*nGroups*dState
	dIn := 2*dInner + 2*nGroups*dState + nHeads
	w.In.MatvecInto(x, &buf.MambaIn)
	if len(buf.MambaIn) < dIn {
		panic("Nemotron-H Mamba input projection has an invalid shape")
	}
	ensureLenNoClear(&buf.MambaZ, dInner)
	ensureLenNoClear(&buf.MambaConv, channels)
	ensureLenNoClear(&buf.MambaDT, nHeads)
	copy(buf.MambaZ, buf.MambaIn[:dInner])
	copy(buf.MambaConv, buf.MambaIn[dInner:dInner+channels])
	copy(buf.MambaDT, buf.MambaIn[dInner+channels:dIn])

	// Causal depthwise convolution over [cached d_conv-1 inputs, current].
	// The GGUF stores one kernel row per channel, ordered oldest to newest.
	ensureLenNoClear(&buf.MambaKernel, cfg.SSMConv)
	for ch := range channels {
		w.Conv.RowInto(ch, cfg.SSMConv, &buf.MambaKernel)
		off := state.convOffset(layer, ch)
		v := float32(0)
		for k := 0; k < state.ConvLen; k++ {
			v += buf.MambaKernel[k] * state.Conv[off+k]
		}
		v += buf.MambaKernel[state.ConvLen] * buf.MambaConv[ch]
		if ch < len(w.ConvBias) {
			v += w.ConvBias[ch]
		}
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
	copy(buf.MambaX, buf.MambaConv[:dInner])
	copy(buf.MambaB, buf.MambaConv[dInner:dInner+nGroups*dState])
	copy(buf.MambaC, buf.MambaConv[dInner+nGroups*dState:channels])
	for h := range nHeads {
		dt := nemotronSoftplus(buf.MambaDT[h] + w.DtBias[h])
		decay := float32(math.Exp(float64(dt * w.A[h])))
		group := h / (nHeads / nGroups)
		for d := range headDim {
			off := state.stateOffset(layer, h, d)
			y := float32(0)
			xv := buf.MambaX[h*headDim+d]
			for s := range dState {
				idx := group*dState + s
				state.State[off+s] = state.State[off+s]*decay + dt*xv*buf.MambaB[idx]
				y += state.State[off+s] * buf.MambaC[idx]
			}
			if h < len(w.D) {
				y += xv * w.D[h]
			}
			buf.MambaY[h*headDim+d] = buf.MambaZ[h*headDim+d] * (y / (1 + float32(math.Exp(float64(-y)))))
		}
	}
	// Mamba-2 applies RMS normalization independently to each B/C group.
	perGroup := dInner / nGroups
	for group := range nGroups {
		start := group * perGroup
		part := buf.MambaY[start : start+perGroup]
		ss := DotF32(part, part)
		scale := float32(1 / math.Sqrt(float64(ss/float32(perGroup)+cfg.RMSNormEps)))
		for i := range part {
			weight := float32(1)
			if start+i < len(w.Norm) {
				weight = w.Norm[start+i]
			}
			part[i] *= weight * scale
		}
	}
	w.Out.MatvecInto(buf.MambaY, &buf.Proj)
}

func nemotronMoEForward(cfg Config, w NemotronMoEWeights, x []float32, buf *DecodeBuffer) {
	w.Router.MatvecInto(x, &buf.RouterLogits)
	if len(buf.RouterLogits) < cfg.ExpertCount {
		panic("Nemotron-H router has an invalid shape")
	}
	ensureLenNoClear(&buf.ExpertProbs, cfg.ExpertCount)
	ensureLenNoClear(&buf.TopExperts, cfg.ExpertUsedCount)
	for i := range cfg.ExpertCount {
		p := nemotronSigmoid(buf.RouterLogits[i])
		buf.ExpertProbs[i] = p
		score := p
		if i < len(w.RouterBias) {
			score += w.RouterBias[i]
		}
		if i < cfg.ExpertUsedCount {
			buf.TopExperts[i] = ExpertScore{Index: i, Score: score}
			continue
		}
		minAt := 0
		for j := 1; j < cfg.ExpertUsedCount; j++ {
			if buf.TopExperts[j].Score < buf.TopExperts[minAt].Score {
				minAt = j
			}
		}
		if score > buf.TopExperts[minAt].Score {
			buf.TopExperts[minAt] = ExpertScore{Index: i, Score: score}
		}
	}
	weightSum := float32(0)
	for _, selected := range buf.TopExperts {
		weightSum += buf.ExpertProbs[selected.Index]
	}
	clear(buf.Proj[:cfg.Dim])
	routed := x
	if w.LatentIn != nil {
		w.LatentIn.MatvecInto(x, &buf.MOE)
		routed = buf.MOE
	}
	for _, selected := range buf.TopExperts {
		expertMatvecInto(w.Up, selected.Index, routed, &buf.ExpertHidden, &buf.ExpertRow)
		for i, v := range buf.ExpertHidden {
			if v < 0 {
				buf.ExpertHidden[i] = 0
			} else {
				buf.ExpertHidden[i] = v * v
			}
		}
		expertMatvecInto(w.Down, selected.Index, buf.ExpertHidden, &buf.MOE, &buf.ExpertRow)
		routing := buf.ExpertProbs[selected.Index]
		if cfg.ExpertWeightsNorm && weightSum > 6.103515625e-5 {
			routing /= weightSum
		}
		routing *= cfg.ExpertWeightsScale
		AxpyF32(buf.Proj[:cfg.Dim], routing, buf.MOE)
	}
	if w.LatentOut != nil {
		w.LatentOut.MatvecInto(buf.Proj[:cfg.Dim], &buf.MOE)
		copy(buf.Proj[:cfg.Dim], buf.MOE)
	}
	if w.SharedUp != nil && w.SharedDown != nil {
		w.SharedUp.MatvecInto(x, &buf.ExpertHidden)
		for i, v := range buf.ExpertHidden {
			if v < 0 {
				buf.ExpertHidden[i] = 0
			} else {
				buf.ExpertHidden[i] = v * v
			}
		}
		w.SharedDown.MatvecInto(buf.ExpertHidden, &buf.MOE)
		addInPlace(buf.Proj[:cfg.Dim], buf.MOE)
	}
}

func expertMatvecInto(w ExpertWeight, expert int, x []float32, out *[]float32, row *[]float32) {
	if expert < 0 || expert >= w.Experts || len(x) != w.Input {
		panic("invalid Nemotron-H expert matvec")
	}
	ensureLenNoClear(out, w.Output)
	if w.Weight.F32 != nil {
		base := expert * w.Output * w.Input
		MatvecF32Into(w.Weight.F32[base:base+w.Output*w.Input], x, w.Output, w.Input, out)
		return
	}
	ensureLenNoClear(row, w.Input)
	for r := range w.Output {
		w.Weight.RowInto(expert*w.Output+r, w.Input, row)
		(*out)[r] = DotF32(*row, x)
	}
}
