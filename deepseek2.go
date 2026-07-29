package gopherllm

import (
	"fmt"
	"io"
	"math"
)

// MLAAttentionWeights is the DeepSeek-V2/V3/Kimi-K2 Multi-head Latent
// Attention layout.  Q/K/V cache widths in Config are compressed widths; KB
// absorbs the no-position Q projection into the cached KV latent and VB
// expands the attended latent back into normal per-head values.
type MLAAttentionWeights struct {
	// Q is used by small/legacy DeepSeek2 variants without Q-LoRA. Kimi K2
	// uses QA -> RMSNorm -> QB instead.
	Q      Weight
	QA     Weight
	QANorm []float32
	QB     Weight

	KVA     Weight
	KVANorm []float32
	KB      ExpertWeight // [q_nope, kv_rank, head]
	VB      ExpertWeight // [kv_rank, value_mla, head]
}

// GGUF's expert_gating_func enum follows llama.cpp. Kimi K2 writes 2 for the
// sigmoid/noaux gate; unknown and older DeepSeek2 files use softmax.
const deepSeekExpertGatingSigmoid = 2

// LoadDeepSeek2Model loads DeepSeek-V2/V3 MLA models, including all current
// Kimi-K2 GGUFs. It deliberately rejects Metal: the existing Metal kernels
// assume ordinary Q/K/V matrices and would otherwise silently bypass MLA's
// latent compression/expansion math.
func LoadDeepSeek2Model(data []byte, gguf *GGUFFile, borrowQuantized, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, ModelWeights, error) {
	if logw == nil {
		logw = io.Discard
	}
	config := ConfigFromGGUF(gguf)
	if !deepSeek2Family(config.Arch) {
		return config, ModelWeights{}, fmt.Errorf("%s is not a DeepSeek2/Kimi architecture", config.Arch)
	}
	if useMetal {
		return config, ModelWeights{}, fmt.Errorf("%s MLA attention does not support Metal yet; retry without --metal", config.Arch)
	}
	if err := validateDeepSeek2Config(&config); err != nil {
		return config, ModelWeights{}, err
	}
	lazyScalarWeights := len(outOfCore) > 0 && outOfCore[0]
	fmt.Fprintf(logw, "DeepSeek2/MLA config: dim=%d, layers=%d, heads=%d, kv-rank=%d, key/value=%d/%d, experts=%d/%d\n",
		config.Dim, config.NLayers, config.NHeads, config.MLAKVLoRARank, config.MLAKeyDim, config.MLAValueDim, config.ExpertCount, config.ExpertUsedCount)
	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)

	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrowQuantized, prepareQuantized, false, lazyScalarWeights)
	}
	tokenEmbd, err := load("token_embd.weight")
	if err != nil {
		return config, ModelWeights{}, err
	}
	outputNorm, err := loadF32Vec(data, gguf.DataOffset, "output_norm.weight", tensors, inferred)
	if err != nil {
		return config, ModelWeights{}, err
	}
	output := tokenEmbd
	if _, ok := tensors["output.weight"]; ok {
		output, err = load("output.weight")
		if err != nil {
			return config, ModelWeights{}, err
		}
	} else {
		fmt.Fprintln(logw, "Note: output tied to embeddings")
	}

	layers := make([]LayerWeights, 0, config.NLayers)
	for l := range config.NLayers {
		layer, err := loadDeepSeek2Layer(data, gguf.DataOffset, l, config, tensors, inferred, borrowQuantized, prepareQuantized, lazyScalarWeights)
		if err != nil {
			return config, ModelWeights{}, fmt.Errorf("layer %d: %w", l, err)
		}
		layers = append(layers, layer)
		if l == 0 || (l+1)%8 == 0 || l+1 == config.NLayers {
			fmt.Fprintf(logw, "  Loaded MLA layer %d/%d\n", l+1, config.NLayers)
		}
	}
	return config, ModelWeights{
		TokenEmbd:      tokenEmbd,
		OutputNorm:     outputNorm,
		OutputNormBias: loadOptionalF32Vec(data, gguf.DataOffset, "output_norm.bias", tensors, inferred, config.Dim),
		Output:         output,
		Layers:         layers,
	}, nil
}

func validateDeepSeek2Config(config *Config) error {
	if !config.UsesMLA || config.Dim <= 0 || config.NLayers <= 0 || config.NHeads <= 0 ||
		config.MLAQueryLoRARank < 0 || config.MLAKVLoRARank <= 0 || config.MLAKeyDim <= 0 || config.MLAValueDim <= 0 {
		return fmt.Errorf("%s requires MLA metadata: q_lora_rank, kv_lora_rank, key_length_mla, and value_length_mla", config.Arch)
	}
	if config.NKVHeads != 1 {
		return fmt.Errorf("%s MLA requires attention.head_count_kv=1, got %d", config.Arch, config.NKVHeads)
	}
	if config.RopeDimensionCount <= 0 || config.MLAKeyDim <= config.RopeDimensionCount || config.RopeDimensionCount%2 != 0 {
		return fmt.Errorf("%s MLA has invalid key_length_mla=%d / rope.dimension_count=%d", config.Arch, config.MLAKeyDim, config.RopeDimensionCount)
	}
	cacheKeyDim := config.MLAKVLoRARank + config.RopeDimensionCount
	if config.HeadDim != cacheKeyDim || config.ValueDim != config.MLAKVLoRARank {
		return fmt.Errorf("%s MLA cache metadata mismatch: key/value lengths are %d/%d, want %d/%d from kv_lora_rank + rope and kv_lora_rank", config.Arch, config.HeadDim, config.ValueDim, cacheKeyDim, config.MLAKVLoRARank)
	}
	if config.LeadingDenseBlockCount < 0 || config.LeadingDenseBlockCount > config.NLayers {
		return fmt.Errorf("%s has invalid leading_dense_block_count=%d", config.Arch, config.LeadingDenseBlockCount)
	}
	if config.ExpertGroupCount == 0 {
		config.ExpertGroupCount = 1
	}
	if config.ExpertGroupUsedCount == 0 {
		config.ExpertGroupUsedCount = 1
	}
	if config.ExpertCount <= 0 || config.ExpertGroupCount > config.ExpertCount ||
		config.ExpertCount%config.ExpertGroupCount != 0 || config.ExpertGroupUsedCount > config.ExpertGroupCount ||
		config.ExpertUsedCount > (config.ExpertCount/config.ExpertGroupCount)*config.ExpertGroupUsedCount {
		return fmt.Errorf("%s has invalid grouped MoE metadata: experts=%d, expert_group_count=%d, expert_group_used_count=%d", config.Arch, config.ExpertCount, config.ExpertGroupCount, config.ExpertGroupUsedCount)
	}
	return nil
}

func requireDeepSeek2Matrix(tensors map[string]TensorInfo, name string, input, output int) error {
	info, ok := tensors[name]
	if !ok {
		return fmt.Errorf("missing tensor: %s", name)
	}
	if len(info.Dims) != 2 || int(info.Dims[0]) != input || int(info.Dims[1]) != output {
		return fmt.Errorf("tensor %s has dimensions %v, want [%d %d]", name, info.Dims, input, output)
	}
	return nil
}

func loadDeepSeek2Layer(data []byte, dataOffset, l int, config Config, tensors map[string]TensorInfo, inferred map[string]int, borrow, prepareQuantized, lazyScalarWeights bool) (LayerWeights, error) {
	prefix := fmt.Sprintf("blk.%d.", l)
	load := func(name string) (Weight, error) {
		return loadWeight(data, dataOffset, name, tensors, inferred, false, borrow, prepareQuantized, false, lazyScalarWeights)
	}
	attnNorm, err := loadF32Vec(data, dataOffset, prefix+"attn_norm.weight", tensors, inferred)
	if err != nil {
		return LayerWeights{}, err
	}
	keyNope := config.MLAKeyDim - config.RopeDimensionCount
	if keyNope <= 0 {
		return LayerWeights{}, fmt.Errorf("invalid MLA no-position key width")
	}
	mla := &MLAAttentionWeights{}
	if config.MLAQueryLoRARank > 0 {
		for _, shape := range []struct {
			name          string
			input, output int
		}{
			{prefix + "attn_q_a.weight", config.Dim, config.MLAQueryLoRARank},
			{prefix + "attn_q_b.weight", config.MLAQueryLoRARank, config.NHeads * config.MLAKeyDim},
		} {
			if err := requireDeepSeek2Matrix(tensors, shape.name, shape.input, shape.output); err != nil {
				return LayerWeights{}, err
			}
		}
		mla.QA, err = load(prefix + "attn_q_a.weight")
		if err != nil {
			return LayerWeights{}, err
		}
		mla.QANorm, err = loadF32Vec(data, dataOffset, prefix+"attn_q_a_norm.weight", tensors, inferred)
		if err != nil {
			return LayerWeights{}, err
		}
		if len(mla.QANorm) != config.MLAQueryLoRARank {
			return LayerWeights{}, fmt.Errorf("tensor %sattn_q_a_norm.weight has length %d, want %d", prefix, len(mla.QANorm), config.MLAQueryLoRARank)
		}
		mla.QB, err = load(prefix + "attn_q_b.weight")
		if err != nil {
			return LayerWeights{}, err
		}
	} else {
		if err := requireDeepSeek2Matrix(tensors, prefix+"attn_q.weight", config.Dim, config.NHeads*config.MLAKeyDim); err != nil {
			return LayerWeights{}, err
		}
		mla.Q, err = load(prefix + "attn_q.weight")
		if err != nil {
			return LayerWeights{}, err
		}
	}
	if err := requireDeepSeek2Matrix(tensors, prefix+"attn_kv_a_mqa.weight", config.Dim, config.MLAKVLoRARank+config.RopeDimensionCount); err != nil {
		return LayerWeights{}, err
	}
	mla.KVA, err = load(prefix + "attn_kv_a_mqa.weight")
	if err != nil {
		return LayerWeights{}, err
	}
	mla.KVANorm, err = loadF32Vec(data, dataOffset, prefix+"attn_kv_a_norm.weight", tensors, inferred)
	if err != nil {
		return LayerWeights{}, err
	}
	if len(mla.KVANorm) != config.MLAKVLoRARank {
		return LayerWeights{}, fmt.Errorf("tensor %sattn_kv_a_norm.weight has length %d, want %d", prefix, len(mla.KVANorm), config.MLAKVLoRARank)
	}
	mla.KB, err = loadExpertWeight(data, dataOffset, prefix+"attn_k_b.weight", tensors, inferred, borrow, lazyScalarWeights)
	if err != nil {
		return LayerWeights{}, err
	}
	if err := validateExpertWeight(prefix+"attn_k_b.weight", mla.KB, keyNope, config.MLAKVLoRARank, config.NHeads); err != nil {
		return LayerWeights{}, err
	}
	mla.VB, err = loadExpertWeight(data, dataOffset, prefix+"attn_v_b.weight", tensors, inferred, borrow, lazyScalarWeights)
	if err != nil {
		return LayerWeights{}, err
	}
	if err := validateExpertWeight(prefix+"attn_v_b.weight", mla.VB, config.MLAKVLoRARank, config.MLAValueDim, config.NHeads); err != nil {
		return LayerWeights{}, err
	}
	if err := requireDeepSeek2Matrix(tensors, prefix+"attn_output.weight", config.NHeads*config.MLAValueDim, config.Dim); err != nil {
		return LayerWeights{}, err
	}
	wo, err := load(prefix + "attn_output.weight")
	if err != nil {
		return LayerWeights{}, err
	}
	ffnNorm, err := loadF32Vec(data, dataOffset, prefix+"ffn_norm.weight", tensors, inferred)
	if err != nil {
		return LayerWeights{}, err
	}

	layer := LayerWeights{
		AttnNorm:     attnNorm,
		AttnNormBias: loadOptionalF32Vec(data, dataOffset, prefix+"attn_norm.bias", tensors, inferred, len(attnNorm)),
		MLA:          mla,
		WO:           wo,
		BO:           make([]float32, config.Dim),
		FFNNorm:      ffnNorm,
		FFNNormBias:  loadOptionalF32Vec(data, dataOffset, prefix+"ffn_norm.bias", tensors, inferred, len(ffnNorm)),
	}
	if l < config.LeadingDenseBlockCount {
		for _, shape := range []struct {
			name          string
			input, output int
		}{
			{prefix + "ffn_gate.weight", config.Dim, config.HiddenDim},
			{prefix + "ffn_up.weight", config.Dim, config.HiddenDim},
			{prefix + "ffn_down.weight", config.HiddenDim, config.Dim},
		} {
			if err := requireDeepSeek2Matrix(tensors, shape.name, shape.input, shape.output); err != nil {
				return LayerWeights{}, err
			}
		}
		layer.W1, err = load(prefix + "ffn_gate.weight")
		if err != nil {
			return LayerWeights{}, err
		}
		layer.W3, err = load(prefix + "ffn_up.weight")
		if err != nil {
			return LayerWeights{}, err
		}
		layer.W2, err = load(prefix + "ffn_down.weight")
		return layer, err
	}
	layer.MoE, err = loadSparseMoEWeights(data, dataOffset, prefix, config, tensors, inferred, borrow, prepareQuantized, false, lazyScalarWeights)
	if err != nil {
		return LayerWeights{}, err
	}
	return layer, nil
}

// ForwardDeepSeek2BodyInto executes MLA attention followed by the normal
// pre-norm SwiGLU/MoE residual graph. It is intentionally separate from the
// ordinary GQA fast path: its cache holds compressed KV latents, and applying
// the normal Q/K/V code would produce plausible-looking but invalid output.
func ForwardDeepSeek2BodyInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	if !config.UsesMLA {
		panic("DeepSeek2 forward called without MLA configuration")
	}
	if cache == nil || cache.PerPosKDim != config.HeadDim || cache.PerPosVDim != config.MLAKVLoRARank {
		panic("DeepSeek2 MLA cache has incompatible dimensions")
	}
	dim := config.Dim
	weights.TokenEmbd.RowInto(int(token), dim, &buf.X)
	if config.EmbeddingScale != 1 {
		ScaleF32(buf.X[:dim], config.EmbeddingScale)
	}
	ropeHalf, ropePairs := prepareRopeScratch(pos, config.RopeDimensionCount, config.RopeDimensionCount, buf.RopeInvFreq, 1, &buf.RopeSin, &buf.RopeCos)
	for l := range config.NLayers {
		layer := weights.Layers[l]
		if layer.MLA == nil {
			panic("DeepSeek2 layer is missing MLA weights")
		}
		normalizeDecoderInto(config, buf.X, layer.AttnNorm, layer.AttnNormBias, &buf.XN)
		forwardMLAAttentionInto(config, layer, l, cache, buf, pos, ropeHalf, ropePairs)
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		addInPlace(buf.X[:dim], buf.Proj)

		normalizeDecoderInto(config, buf.X, layer.FFNNorm, layer.FFNNormBias, &buf.XN2)
		if layer.MoE != nil {
			sparseMoEForward(layer.MoE, buf.XN2, buf)
		} else {
			if !tryMatvec2Into(layer.W1, layer.W3, buf.XN2, &buf.Q4KXSums, &buf.Gate, &buf.Up) {
				layer.W1.MatvecInto(buf.XN2, &buf.Gate)
				layer.W3.MatvecInto(buf.XN2, &buf.Up)
			}
			ensureLenNoClear(&buf.Hidden, config.HiddenDim)
			siluMulF32(buf.Gate[:config.HiddenDim], buf.Up[:config.HiddenDim], buf.Hidden[:config.HiddenDim])
			layer.W2.MatvecInto(buf.Hidden[:config.HiddenDim], &buf.Proj)
		}
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		addInPlace(buf.X[:dim], buf.Proj)
	}
	normalizeDecoderInto(config, buf.X, weights.OutputNorm, weights.OutputNormBias, &buf.XN)
}

func forwardMLAAttentionInto(config Config, layer LayerWeights, layerIndex int, cache *KVCache, buf *DecodeBuffer, pos, ropeHalf, ropePairs int) {
	mla := layer.MLA
	keyNope := config.MLAKeyDim - config.RopeDimensionCount
	cacheKeyDim := config.MLAKVLoRARank + config.RopeDimensionCount
	if keyNope <= 0 || cacheKeyDim != config.HeadDim {
		panic("invalid MLA dimensions")
	}
	if config.MLAQueryLoRARank > 0 {
		mla.QA.MatvecInto(buf.XN, &buf.MLAQ)
		rmsNormInto(buf.MLAQ[:config.MLAQueryLoRARank], mla.QANorm, config.RMSNormEps, &buf.MLAQ)
		mla.QB.MatvecInto(buf.MLAQ[:config.MLAQueryLoRARank], &buf.Q)
	} else {
		mla.Q.MatvecInto(buf.XN, &buf.Q)
	}
	qOriginalLen := config.NHeads * config.MLAKeyDim
	if len(buf.Q) != qOriginalLen {
		panic("MLA Q projection has invalid shape")
	}
	mla.KVA.MatvecInto(buf.XN, &buf.MLAKV)
	if len(buf.MLAKV) != cacheKeyDim {
		panic("MLA KV projection has invalid shape")
	}
	ensureLenNoClear(&buf.K, cacheKeyDim)
	ensureLenNoClear(&buf.V, config.MLAKVLoRARank)
	rmsNormInto(buf.MLAKV[:config.MLAKVLoRARank], mla.KVANorm, config.RMSNormEps, &buf.V)
	copy(buf.K[:config.MLAKVLoRARank], buf.V[:config.MLAKVLoRARank])
	copy(buf.K[config.MLAKVLoRARank:cacheKeyDim], buf.MLAKV[config.MLAKVLoRARank:cacheKeyDim])

	// RoPE applies only to the positional suffix of Q and K, never to the
	// compressed KV latent/no-position Q subspaces.
	for h := range config.NHeads {
		qOffset := h*config.MLAKeyDim + keyNope
		applyPreparedRope(buf.Q[qOffset:qOffset+config.RopeDimensionCount], config.RopeDimensionCount, 1, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, false)
	}
	applyPreparedRope(buf.K[config.MLAKVLoRARank:cacheKeyDim], config.RopeDimensionCount, 1, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, false)

	// Absorb Wk_b into every no-position query. Expand from the end so writing
	// the larger [kv_rank + rope] head layout cannot overwrite a not-yet-read
	// compact [nope + rope] query for a later head.
	ensureLenNoClear(&buf.Q, config.NHeads*cacheKeyDim)
	for h := config.NHeads - 1; h >= 0; h-- {
		src := h * config.MLAKeyDim
		dst := h * cacheKeyDim
		ensureLenNoClear(&buf.MLATmp, config.RopeDimensionCount)
		copy(buf.MLATmp[:config.RopeDimensionCount], buf.Q[src+keyNope:src+config.MLAKeyDim])
		expertMatvecInto(mla.KB, h, buf.Q[src:src+keyNope], &buf.MLAQ, &buf.ExpertRow)
		copy(buf.Q[dst:dst+config.MLAKVLoRARank], buf.MLAQ[:config.MLAKVLoRARank])
		copy(buf.Q[dst+config.MLAKVLoRARank:dst+cacheKeyDim], buf.MLATmp[:config.RopeDimensionCount])
	}

	cache.storeKV(layerIndex, pos, buf.K[:cacheKeyDim], buf.V[:config.MLAKVLoRARank])

	// Each head attends a distinct absorbed Q projection against the one
	// compact KV-head. The attended value is still a latent of kv_lora_rank;
	// V_b below expands it to the normal per-head value width before WO.
	ensureLenNoClear(&buf.AttnOut, config.NHeads*config.MLAKVLoRARank)
	clear(buf.AttnOut[:config.NHeads*config.MLAKVLoRARank])
	scale := config.AttentionScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(config.MLAKeyDim)))
	}
	// llama.cpp's MLA graph folds YaRN's mscale into the attention scale rather
	// than multiplying the compact Q/K suffixes. buildRopeInvFreq already uses
	// the GGUF YaRN frequency interpolation; preserve the matching score scale.
	if config.RopeScalingType == "yarn" && config.RopeScalingFactor > 1 && config.RopeYarnLogMultiplier != 0 {
		logMultiplier := config.RopeYarnLogMultiplier / 0.1
		attnFactor := config.RopeAttentionFactor
		if attnFactor == 0 {
			attnFactor = 1
		}
		mscale := attnFactor * (1 + 0.1*logMultiplier*float32(math.Log(float64(config.RopeScalingFactor))))
		scale *= mscale * mscale
	}
	attendHeads := func(start, end int) {
		for h := start; h < end; h++ {
			qOffset := h * cacheKeyDim
			outOffset := h * config.MLAKVLoRARank
			cache.attendHead(layerIndex, 0, buf.Q[qOffset:qOffset+cacheKeyDim], cacheKeyDim, config.MLAKVLoRARank,
				0, pos, scale, 0, buf.AttnOut[outOffset:outOffset+config.MLAKVLoRARank])
		}
	}
	if pos >= 127 && config.NHeads > 1 {
		parallelChunks(config.NHeads, attendHeads)
	} else {
		attendHeads(0, config.NHeads)
	}

	ensureLenNoClear(&buf.MLAValues, config.NHeads*config.MLAValueDim)
	for h := range config.NHeads {
		latentOffset := h * config.MLAKVLoRARank
		valueOffset := h * config.MLAValueDim
		expertMatvecInto(mla.VB, h, buf.AttnOut[latentOffset:latentOffset+config.MLAKVLoRARank], &buf.MLATmp, &buf.ExpertRow)
		copy(buf.MLAValues[valueOffset:valueOffset+config.MLAValueDim], buf.MLATmp[:config.MLAValueDim])
	}
	layer.WO.MatvecInto(buf.MLAValues[:config.NHeads*config.MLAValueDim], &buf.Proj)
	addInPlace(buf.Proj, layer.BO)
}
