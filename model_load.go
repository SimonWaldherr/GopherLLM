package gopherllm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sort"
)

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
	var positionEmbd Weight
	if config.usesAbsolutePositionEmbd() {
		positionEmbd, err = loadWeight(data, gguf.DataOffset, "position_embd.weight", tensorIdx, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, ModelWeights{}, err
		}
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
		TokenEmbd: tokenEmbd, PositionEmbd: positionEmbd, OutputNorm: outputNorm,
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

// loadFalconNorms resolves Falcon's attn_norm/attn_norm_2 aliasing. 7B
// checkpoints carry one LayerNorm (attn_norm) that feeds both the attention
// and FFN branches; 40B/180B ("new decoder architecture") checkpoints carry
// a second attn_norm_2 that — despite the naming — feeds the ATTENTION
// branch, leaving the base attn_norm feeding FFN. See
// Config.sharesParallelBranchNorm's doc comment for why this is handled here
// as tensor aliasing rather than as a forward-pass special case.
func loadFalconNorms(data []byte, dataOffset int, prefix string, tensors map[string]TensorInfo, inferred map[string]int) (attnNorm, ffnNorm []float32, err error) {
	base, err := loadF32Vec(data, dataOffset, prefix+"attn_norm.weight", tensors, inferred)
	if err != nil {
		return nil, nil, err
	}
	if _, hasNorm2 := tensors[prefix+"attn_norm_2.weight"]; hasNorm2 {
		attnNorm, err = loadF32Vec(data, dataOffset, prefix+"attn_norm_2.weight", tensors, inferred)
		if err != nil {
			return nil, nil, err
		}
		return attnNorm, base, nil
	}
	return base, base, nil
}

func loadLayer(data []byte, dataOffset, l int, config Config, tensors map[string]TensorInfo, inferred map[string]int, borrow, prepareQuantized, useMetal, lazyScalarWeights bool, qRows, kRows, vRows int) (LayerWeights, error) {
	prefix := fmt.Sprintf("blk.%d.", l)
	var attnNorm, falconFFNNorm []float32
	var err error
	if config.Arch == "falcon" {
		attnNorm, falconFFNNorm, err = loadFalconNorms(data, dataOffset, prefix, tensors, inferred)
		if err != nil {
			return LayerWeights{}, err
		}
	} else if config.usesPostNormOnly() {
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
	if config.Arch == "falcon" {
		ffnNorm = falconFFNNorm
	} else if config.usesPostNormOnly() || config.sharesParallelBranchNorm() {
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
		if config.usesPlainMLP() {
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
	} else if config.usesPlainMLP() {
		// GPT-2/GPT-NeoX/GPT-J/BLOOM/MPT/StarCoder/StarCoder2's plain MLP
		// biases. Unlike phi2 these are not strictly required (GPT-J and a
		// few community exports omit them), so fall back to zero-filled.
		ffnUpBias = loadOptionalF32Vec(data, dataOffset, prefix+"ffn_up.bias", tensors, inferred, config.HiddenDim)
		ffnDownBias = loadOptionalF32Vec(data, dataOffset, prefix+"ffn_down.bias", tensors, inferred, config.Dim)
	}
	if config.Arch == "gpt-oss" {
		if len(attnSinks) != config.NHeads {
			return LayerWeights{}, fmt.Errorf("gpt-oss requires %d-element attn_sinks.weight", config.NHeads)
		}
		if _, ok := tensors[prefix+"attn_output.bias"]; !ok || len(bo) != config.Dim {
			return LayerWeights{}, fmt.Errorf("gpt-oss requires %d-element attn_output.bias", config.Dim)
		}
	}
	bq, bk, bv := loadQKVBias(data, dataOffset, prefix, tensors, inferred, qRows, kRows, vRows)
	return LayerWeights{
		AttnNorm:     attnNorm,
		AttnNormBias: loadOptionalF32Vec(data, dataOffset, prefix+"attn_norm.bias", tensors, inferred, len(attnNorm)),
		WQ:           wq,
		BQ:           bq,
		WK:           wk,
		BK:           bk,
		WV:           wv,
		BV:           bv,
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

// loadQKVBias loads Q/K/V projection biases, preferring a single fused
// attn_qkv.bias tensor (GPT-2/GPT-NeoX/BLOOM/MPT/StarCoder/ChatGLM's
// convention: one bias tensor covering Q rows, then K rows, then V rows,
// matching their fused attn_qkv.weight row layout) and falling back to
// separate attn_q.bias/attn_k.bias/attn_v.bias tensors otherwise. Returns
// zero-filled slices of the expected length when no bias tensor exists at
// all, matching loadOptionalF32Vec's existing no-bias contract.
func loadQKVBias(data []byte, dataOffset int, prefix string, tensors map[string]TensorInfo, inferred map[string]int, qRows, kRows, vRows int) (bq, bk, bv []float32) {
	if _, ok := tensors[prefix+"attn_qkv.bias"]; ok {
		fused := loadOptionalF32Vec(data, dataOffset, prefix+"attn_qkv.bias", tensors, inferred, qRows+kRows+vRows)
		bq = append([]float32(nil), fused[:qRows]...)
		bk = append([]float32(nil), fused[qRows:qRows+kRows]...)
		bv = append([]float32(nil), fused[qRows+kRows:qRows+kRows+vRows]...)
		return bq, bk, bv
	}
	bq = loadOptionalF32Vec(data, dataOffset, prefix+"attn_q.bias", tensors, inferred, qRows)
	bk = loadOptionalF32Vec(data, dataOffset, prefix+"attn_k.bias", tensors, inferred, kRows)
	bv = loadOptionalF32Vec(data, dataOffset, prefix+"attn_v.bias", tensors, inferred, vRows)
	return bq, bk, bv
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

// tensorContainer resolves where a tensor's bytes live and at what offset
// within that slice. It is the single point that understands TensorInfo.Shard,
// which is what keeps out-of-core split-GGUF support from having to thread a
// multi-file tensor source through all 130-odd loadWeight call sites.
func tensorContainer(data []byte, dataOffset int, info TensorInfo) ([]byte, int) {
	if info.Shard != nil {
		// Offset is already relative to the shard's tensor region.
		return info.Shard, int(info.Offset)
	}
	return data, dataOffset + int(info.Offset)
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
	// Tensors carrying their own Shard live in separate offset spaces, so a
	// single global sort would compute gaps between tensors in different
	// files and size them nonsensically. Group by shard identity first; the
	// nil-Shard (single-file) case collapses to exactly one group, which is
	// the original behaviour byte for byte.
	groups := make(map[*byte][]offIdx)
	limits := make(map[*byte]uint64)
	for i, t := range gguf.Tensors {
		var key *byte
		limit := uint64(len(data) - gguf.DataOffset)
		if t.Shard != nil {
			key = &t.Shard[0]
			limit = uint64(len(t.Shard))
		}
		groups[key] = append(groups[key], offIdx{t.Offset, i})
		limits[key] = limit
	}
	out := make(map[string]int, len(gguf.Tensors))
	for key, offs := range groups {
		sort.Slice(offs, func(i, j int) bool { return offs[i].off < offs[j].off })
		for i, oi := range offs {
			next := limits[key]
			if i+1 < len(offs) {
				next = offs[i+1].off
			}
			size := 0
			if next > oi.off {
				size = int(next - oi.off)
			}
			out[gguf.Tensors[oi.idx].Name] = size
		}
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
	// container is the byte range this tensor actually lives in: the whole
	// model buffer for a single-file GGUF, or just this tensor's own shard
	// mapping for an out-of-core split model. Everything below is expressed
	// against it, so the two layouts share one code path.
	container, offset := tensorContainer(data, dataOffset, info)
	if inferredSize := inferred[name]; inferredSize > 0 {
		end := offset + byteSize
		if end > len(container) || byteSize == 0 {
			byteSize = inferredSize
		}
	}
	if offset < 0 || offset > len(container) {
		return Weight{}, fmt.Errorf("tensor %s offset out of range", name)
	}
	rawEnd := min(offset+byteSize, len(container))
	raw := container[offset:rawEnd]
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
	case GGMLTypeQ8_0, GGMLTypeQ4_0, GGMLTypeQ4_1, GGMLTypeQ5_0, GGMLTypeQ5_1, GGMLTypeQ8_1, GGMLTypeQ8_K,
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
		// prepareWebGPUWeight is a no-op (returns nil) on every non-js build
		// and whenever no WebGPU device is available, so this is safe to call
		// unconditionally rather than threading a new LoadOptions flag through
		// every architecture's Load*Model function the way useMetal is:
		// WebGPU only ever exists under GOOS=js, where useMetal is already
		// always false, so the two GPU backends can never conflict.
		w.GPU = prepareWebGPUWeight(raw, info.DType, rows, cols)
		// Direct Metal weights skip redundant prepared data. CPU-side matrices
		// retain their prepared representation; in particular, narrow GQA
		// Q/K/V projections are intentionally never made into Metal handles.
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
