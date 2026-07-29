package gopherllm

// Native Gemma 4 decoder support.
//
// Gemma 4 is not a parameter-only variation of the older Gemma graph.  Its
// local (SWA) and global blocks have different Q/K/V widths, global blocks
// can use K as V, global RoPE uses proportional frequency factors, and every
// block ends with a learned scalar.  Keeping this graph separate prevents a
// malformed native checkpoint from silently falling through to the ordinary
// GQA path.

import (
	"fmt"
	"io"
	"math"
)

func isNativeGemma4Layout(tensors map[string]TensorInfo) bool {
	_, ok := tensors["blk.0.layer_output_scale.weight"]
	return ok
}

// Gemma4MoEWeights is deliberately separate from SparseMoEWeights. Gemma 4
// does not replace its dense MLP with a generic sparse branch: each MoE block
// runs a shared dense GEGLU MLP and a scaled-router expert GEGLU MLP in
// parallel, normalizes each independently, then normalizes their sum.
type Gemma4MoEWeights struct {
	Router      Weight
	RouterScale []float32 // ffn_gate_inp.scale, length Dim
	PreNorm2    []float32 // pre_ffw_norm_2
	PostNorm1   []float32 // post_ffw_norm_1 (shared dense branch)
	PostNorm2   []float32 // post_ffw_norm_2 (expert branch)
	Gate        ExpertWeight
	Up          ExpertWeight
	Down        ExpertWeight
	DownScale   []float32 // ffn_down_exps.scale, one scale per expert
	ExpertUsed  int
}

func loadGemma4MoEWeights(data []byte, dataOffset int, prefix string, config Config, tensors map[string]TensorInfo, inferred map[string]int, borrow, lazyScalarWeights bool) (*Gemma4MoEWeights, error) {
	if config.ExpertCount <= 0 || config.ExpertUsedCount <= 0 || config.ExpertUsedCount > config.ExpertCount || config.ExpertFeedForwardDim <= 0 {
		return nil, fmt.Errorf("%sgemma4 MoE metadata is invalid: experts=%d used=%d expert_ffn=%d", prefix, config.ExpertCount, config.ExpertUsedCount, config.ExpertFeedForwardDim)
	}
	routerRows, routerCols, err := gemma4MatrixShape(tensors, prefix+"ffn_gate_inp.weight")
	if err != nil {
		return nil, err
	}
	if routerRows != config.ExpertCount || routerCols != config.Dim {
		return nil, fmt.Errorf("%sffn_gate_inp.weight has shape [%d,%d], want [%d,%d]", prefix, routerRows, routerCols, config.ExpertCount, config.Dim)
	}
	router, err := loadWeight(data, dataOffset, prefix+"ffn_gate_inp.weight", tensors, inferred, false, borrow, false, false, lazyScalarWeights)
	if err != nil {
		return nil, err
	}
	routerScale, err := loadF32Vec(data, dataOffset, prefix+"ffn_gate_inp.scale", tensors, inferred)
	if err != nil {
		return nil, err
	}
	preNorm2, err := loadF32Vec(data, dataOffset, prefix+"pre_ffw_norm_2.weight", tensors, inferred)
	if err != nil {
		return nil, err
	}
	postNorm1, err := loadF32Vec(data, dataOffset, prefix+"post_ffw_norm_1.weight", tensors, inferred)
	if err != nil {
		return nil, err
	}
	postNorm2, err := loadF32Vec(data, dataOffset, prefix+"post_ffw_norm_2.weight", tensors, inferred)
	if err != nil {
		return nil, err
	}
	for name, values := range map[string][]float32{
		"ffn_gate_inp.scale":     routerScale,
		"pre_ffw_norm_2.weight":  preNorm2,
		"post_ffw_norm_1.weight": postNorm1,
		"post_ffw_norm_2.weight": postNorm2,
	} {
		if len(values) != config.Dim {
			return nil, fmt.Errorf("%s%s has length %d, want %d", prefix, name, len(values), config.Dim)
		}
		for _, value := range values {
			if !finite32(value) {
				return nil, fmt.Errorf("%s%s contains a non-finite value", prefix, name)
			}
		}
	}
	gate, up, err := loadFusedExpertGateUpWeight(data, dataOffset, prefix+"ffn_gate_up_exps.weight", tensors, inferred, borrow, lazyScalarWeights)
	if err != nil {
		return nil, fmt.Errorf("%sgemma4 fused expert gate/up: %w", prefix, err)
	}
	down, err := loadExpertWeight(data, dataOffset, prefix+"ffn_down_exps.weight", tensors, inferred, borrow, lazyScalarWeights)
	if err != nil {
		return nil, err
	}
	if err := validateExpertWeight(prefix+"ffn_gate_up_exps.weight", gate, config.Dim, config.ExpertFeedForwardDim, config.ExpertCount); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(prefix+"ffn_gate_up_exps.weight", up, config.Dim, config.ExpertFeedForwardDim, config.ExpertCount); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(prefix+"ffn_down_exps.weight", down, config.ExpertFeedForwardDim, config.Dim, config.ExpertCount); err != nil {
		return nil, err
	}
	downScale, err := loadF32Vec(data, dataOffset, prefix+"ffn_down_exps.scale", tensors, inferred)
	if err != nil {
		return nil, err
	}
	if len(downScale) != config.ExpertCount {
		return nil, fmt.Errorf("%sffn_down_exps.scale has length %d, want %d", prefix, len(downScale), config.ExpertCount)
	}
	for _, value := range downScale {
		if !finite32(value) {
			return nil, fmt.Errorf("%sffn_down_exps.scale contains a non-finite value", prefix)
		}
	}
	return &Gemma4MoEWeights{
		Router:      router,
		RouterScale: routerScale,
		PreNorm2:    preNorm2,
		PostNorm1:   postNorm1,
		PostNorm2:   postNorm2,
		Gate:        gate,
		Up:          up,
		Down:        down,
		DownScale:   downScale,
		ExpertUsed:  config.ExpertUsedCount,
	}, nil
}

func loadNativeGemma4Model(data []byte, gguf *GGUFFile, borrowQuantized, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, Gemma4Weights, error) {
	if logw == nil {
		logw = io.Discard
	}
	config := ConfigFromGGUF(gguf)
	if config.Arch != "gemma4" {
		return config, Gemma4Weights{}, fmt.Errorf("native Gemma 4 loader received architecture %q", config.Arch)
	}
	if config.Dim <= 0 || config.NLayers <= 0 || config.NHeads <= 0 {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 invalid decoder geometry: dim=%d layers=%d heads=%d", config.Dim, config.NLayers, config.NHeads)
	}

	// E2B injects per-layer embeddings and reuses the final self-K/V rows in a
	// tail of query-only blocks. Both facts are structural graph changes, not
	// metadata hints, so validate and model them explicitly below.
	perLayerDim := int(gguf.GetU32("gemma4.embedding_length_per_layer_input", 0))
	sharedKVLayers := int(gguf.GetU32("gemma4.attention.shared_kv_layers", 0))
	if perLayerDim < 0 || sharedKVLayers < 0 || sharedKVLayers >= config.NLayers {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 invalid PLE/shared-KV metadata per_layer=%d shared_kv_layers=%d layers=%d", perLayerDim, sharedKVLayers, config.NLayers)
	}
	if config.ExpertCount > 0 && (config.ExpertUsedCount <= 0 || config.ExpertUsedCount > config.ExpertCount || config.ExpertFeedForwardDim <= 0) {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 invalid sparse-MoE metadata experts=%d used=%d expert_ffn=%d", config.ExpertCount, config.ExpertUsedCount, config.ExpertFeedForwardDim)
	}

	tensors := indexTensors(gguf)
	kvHeadsMeta, ok := gguf.GetU32Array("gemma4.attention.head_count_kv")
	if !ok {
		// E2B declares one KV head as a scalar. Expand it only here, after
		// confirming it is positive; all layer geometry remains tensor-driven.
		if scalar := gguf.GetU32("gemma4.attention.head_count_kv", 0); scalar > 0 {
			kvHeadsMeta = make([]uint32, config.NLayers)
			for i := range kvHeadsMeta {
				kvHeadsMeta[i] = scalar
			}
			ok = true
		}
	}
	if !ok || len(kvHeadsMeta) != config.NLayers {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 requires either scalar or %d-entry attention.head_count_kv metadata; got %d entries", config.NLayers, len(kvHeadsMeta))
	}
	if len(config.SWAPattern) != config.NLayers {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 requires a %d-entry attention.sliding_window_pattern", config.NLayers)
	}
	kvSlots, kvSlotCount, err := gemma4KVCacheSlots(config.SWAPattern, sharedKVLayers)
	if err != nil {
		return config, Gemma4Weights{}, err
	}

	inferred := inferTensorSizes(data, gguf)
	lazyScalarWeights := len(outOfCore) > 0 && outOfCore[0]
	tokenEmbd, err := loadWeight(data, gguf.DataOffset, "token_embd.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
	if err != nil {
		return config, Gemma4Weights{}, err
	}
	outputNorm, err := loadF32Vec(data, gguf.DataOffset, "output_norm.weight", tensors, inferred)
	if err != nil {
		return config, Gemma4Weights{}, err
	}
	output := tokenEmbd
	if _, hasOutput := tensors["output.weight"]; hasOutput {
		output, err = loadWeight(data, gguf.DataOffset, "output.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
	} else {
		fmt.Fprintln(logw, "Note: output tied to embeddings")
	}

	var perLayer *Gemma4PerLayerWeights
	if perLayerDim > 0 {
		totalPerLayerDim := perLayerDim * config.NLayers
		if totalPerLayerDim <= 0 || totalPerLayerDim/perLayerDim != config.NLayers {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 invalid total per-layer embedding width %d x %d", perLayerDim, config.NLayers)
		}
		tokRows, tokCols, err := gemma4MatrixShape(tensors, "per_layer_token_embd.weight")
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		projRows, projCols, err := gemma4MatrixShape(tensors, "per_layer_model_proj.weight")
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		if tokRows < config.VocabSize || tokCols != totalPerLayerDim || projRows != totalPerLayerDim || projCols != config.Dim {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 PLE global shapes token=[%d,%d] model_proj=[%d,%d], want token=[vocab>=%d,%d] model_proj=[%d,%d]", tokRows, tokCols, projRows, projCols, config.VocabSize, totalPerLayerDim, totalPerLayerDim, config.Dim)
		}
		projNorm, err := loadF32Vec(data, gguf.DataOffset, "per_layer_proj_norm.weight", tensors, inferred)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		if len(projNorm) != perLayerDim {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 per_layer_proj_norm.weight has length %d, want %d", len(projNorm), perLayerDim)
		}
		perLayerToken, err := loadWeight(data, gguf.DataOffset, "per_layer_token_embd.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		perLayerModelProj, err := loadWeight(data, gguf.DataOffset, "per_layer_model_proj.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		perLayer = &Gemma4PerLayerWeights{
			TokenEmbd: perLayerToken,
			ModelProj: perLayerModelProj,
			ProjNorm:  projNorm,
			Dim:       perLayerDim,
		}
	}

	// Global attention uses proportional RoPE.  These are factors (not a
	// second learned embedding): huge factors intentionally make the matching
	// dimensions effectively unrotated.  Do not replace them with metadata
	// defaults; that produces superficially valid but incorrect logits.
	ropeFactors, err := loadF32Vec(data, gguf.DataOffset, "rope_freqs.weight", tensors, inferred)
	if err != nil {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 global proportional RoPE: %w", err)
	}
	if len(ropeFactors) == 0 {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 global proportional RoPE tensor is empty")
	}

	localHeadHint := int(gguf.GetU32("gemma4.attention.key_length_swa", 0))
	globalHeadHint := int(gguf.GetU32("gemma4.attention.key_length", 0))
	localRopeDim := int(gguf.GetU32("gemma4.rope.dimension_count_swa", uint32(localHeadHint)))
	globalRopeDim := int(gguf.GetU32("gemma4.rope.dimension_count", uint32(globalHeadHint)))
	localRopeTheta := gguf.GetF32("gemma4.rope.freq_base_swa", 0)
	globalRopeTheta := gguf.GetF32("gemma4.rope.freq_base", 0)
	if localRopeTheta <= 0 || globalRopeTheta <= 0 {
		return config, Gemma4Weights{}, fmt.Errorf("gemma4 requires positive local/global RoPE bases (got %g/%g)", localRopeTheta, globalRopeTheta)
	}

	layers := make([]Gemma4LayerWeights, 0, config.NLayers)
	moes := make([]*Gemma4MoEWeights, config.NLayers)
	maxHeadDim, maxKVHeads, maxValueDim, maxFFNDim := 0, 0, 0, 0
	var localInv, globalInv []float32
	for l := range config.NLayers {
		prefix := fmt.Sprintf("blk.%d.", l)
		isSWA := config.SWAPattern[l]
		hasKV := l < kvSlotCount
		kvHeads := int(kvHeadsMeta[l])
		if kvHeads <= 0 || kvHeads > config.NHeads || config.NHeads%kvHeads != 0 {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d invalid KV-head count %d for %d query heads", l, kvHeads, config.NHeads)
		}

		qRows, qCols, err := gemma4MatrixShape(tensors, prefix+"attn_q.weight")
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		if qCols != config.Dim || qRows%config.NHeads != 0 {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d invalid Q shape [%d,%d] for dim=%d heads=%d", l, qRows, qCols, config.Dim, config.NHeads)
		}
		headDim := qRows / config.NHeads
		if isSWA && localHeadHint > 0 && headDim != localHeadHint {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d local Q head dimension=%d, metadata says %d", l, headDim, localHeadHint)
		}
		if !isSWA && globalHeadHint > 0 && headDim != globalHeadHint {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d global Q head dimension=%d, metadata says %d", l, headDim, globalHeadHint)
		}

		// E2B's final shared-KV tail owns Q and O but intentionally does not
		// calculate or write K/V. Those tensors may still be serialized in the
		// GGUF, so only the first kvSlotCount layers are validated/loaded as
		// physical cache producers.
		hasV := false
		valueDim := headDim
		if hasKV {
			kRows, kCols, err := gemma4MatrixShape(tensors, prefix+"attn_k.weight")
			if err != nil {
				return config, Gemma4Weights{}, err
			}
			if kCols != config.Dim || kRows != kvHeads*headDim {
				return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d invalid K shape [%d,%d], want [%d,%d]", l, kRows, kCols, kvHeads*headDim, config.Dim)
			}
			_, hasV = tensors[prefix+"attn_v.weight"]
			if isSWA && !hasV {
				return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d local attention is missing attn_v.weight", l)
			}
			if hasV {
				vRows, vCols, err := gemma4MatrixShape(tensors, prefix+"attn_v.weight")
				if err != nil {
					return config, Gemma4Weights{}, err
				}
				if vCols != config.Dim || vRows != kvHeads*headDim {
					return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d invalid V shape [%d,%d], want [%d,%d]", l, vRows, vCols, kvHeads*headDim, config.Dim)
				}
			}
		}
		woRows, woCols, err := gemma4MatrixShape(tensors, prefix+"attn_output.weight")
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		if woRows != config.Dim || woCols != qRows {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d invalid output shape [%d,%d], want [%d,%d]", l, woRows, woCols, config.Dim, qRows)
		}

		gateRows, gateCols, err := gemma4MatrixShape(tensors, prefix+"ffn_gate.weight")
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		upRows, upCols, err := gemma4MatrixShape(tensors, prefix+"ffn_up.weight")
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		downRows, downCols, err := gemma4MatrixShape(tensors, prefix+"ffn_down.weight")
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		if gateCols != config.Dim || upCols != config.Dim || gateRows <= 0 || upRows != gateRows || downCols != gateRows || downRows != config.Dim {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d invalid dense FFN shapes gate=[%d,%d] up=[%d,%d] down=[%d,%d]", l, gateRows, gateCols, upRows, upCols, downRows, downCols)
		}

		attnNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"attn_norm.weight", tensors, inferred)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		qNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"attn_q_norm.weight", tensors, inferred)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		var kNorm []float32
		if hasKV {
			kNorm, err = loadF32Vec(data, gguf.DataOffset, prefix+"attn_k_norm.weight", tensors, inferred)
			if err != nil {
				return config, Gemma4Weights{}, err
			}
		}
		postAttnNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"post_attention_norm.weight", tensors, inferred)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		ffnNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"ffn_norm.weight", tensors, inferred)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		postFFNNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"post_ffw_norm.weight", tensors, inferred)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		outScale, err := loadF32Vec(data, gguf.DataOffset, prefix+"layer_output_scale.weight", tensors, inferred)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		if len(attnNorm) != config.Dim || len(postAttnNorm) != config.Dim || len(ffnNorm) != config.Dim || len(postFFNNorm) != config.Dim || len(qNorm) != headDim || (hasKV && len(kNorm) != headDim) || len(outScale) != 1 || !finite32(outScale[0]) {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d has incompatible norm or layer_output_scale dimensions", l)
		}

		wq, err := loadWeight(data, gguf.DataOffset, prefix+"attn_q.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		var wk Weight
		if hasKV {
			wk, err = loadWeight(data, gguf.DataOffset, prefix+"attn_k.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return config, Gemma4Weights{}, err
			}
		}
		var wv Weight
		if hasKV && hasV {
			wv, err = loadWeight(data, gguf.DataOffset, prefix+"attn_v.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return config, Gemma4Weights{}, err
			}
		}
		wo, err := loadWeight(data, gguf.DataOffset, prefix+"attn_output.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		wGate, err := loadWeight(data, gguf.DataOffset, prefix+"ffn_gate.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		wUp, err := loadWeight(data, gguf.DataOffset, prefix+"ffn_up.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		wDown, err := loadWeight(data, gguf.DataOffset, prefix+"ffn_down.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
		if err != nil {
			return config, Gemma4Weights{}, err
		}
		var perLayerGate, perLayerProj Weight
		var perLayerPostNorm []float32
		if perLayer != nil {
			inpRows, inpCols, err := gemma4MatrixShape(tensors, prefix+"inp_gate.weight")
			if err != nil {
				return config, Gemma4Weights{}, err
			}
			projRows, projCols, err := gemma4MatrixShape(tensors, prefix+"proj.weight")
			if err != nil {
				return config, Gemma4Weights{}, err
			}
			if inpRows != perLayer.Dim || inpCols != config.Dim || projRows != config.Dim || projCols != perLayer.Dim {
				return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d PLE shapes inp_gate=[%d,%d] proj=[%d,%d], want [%d,%d]/[%d,%d]", l, inpRows, inpCols, projRows, projCols, perLayer.Dim, config.Dim, config.Dim, perLayer.Dim)
			}
			perLayerPostNorm, err = loadF32Vec(data, gguf.DataOffset, prefix+"post_norm.weight", tensors, inferred)
			if err != nil {
				return config, Gemma4Weights{}, err
			}
			if len(perLayerPostNorm) != config.Dim {
				return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d post_norm.weight has length %d, want %d", l, len(perLayerPostNorm), config.Dim)
			}
			perLayerGate, err = loadWeight(data, gguf.DataOffset, prefix+"inp_gate.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return config, Gemma4Weights{}, err
			}
			perLayerProj, err = loadWeight(data, gguf.DataOffset, prefix+"proj.weight", tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
			if err != nil {
				return config, Gemma4Weights{}, err
			}
		} else if _, hasPLE := tensors[prefix+"inp_gate.weight"]; hasPLE {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d has PLE tensors but embedding_length_per_layer_input is zero", l)
		}
		if _, hasMoE := tensors[prefix+"ffn_gate_inp.weight"]; hasMoE {
			moes[l], err = loadGemma4MoEWeights(data, gguf.DataOffset, prefix, config, tensors, inferred, borrowQuantized, lazyScalarWeights)
			if err != nil {
				return config, Gemma4Weights{}, err
			}
		} else if config.ExpertCount > 0 {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d is missing ffn_gate_inp.weight for its declared MoE graph", l)
		}

		layer := Gemma4LayerWeights{
			Native:            true,
			AttnNorm:          attnNorm,
			AttnQ:             wq,
			AttnK:             wk,
			AttnV:             wv,
			AttnOutput:        wo,
			FFNNorm:           ffnNorm,
			FFNDown:           wDown,
			FFNUp:             wUp,
			FFNGate:           wGate,
			HeadDim:           headDim,
			NKVHeads:          kvHeads,
			ValueDim:          valueDim,
			HasAttnV:          hasV,
			AttnQNorm:         qNorm,
			AttnKNorm:         kNorm,
			PostAttnNorm:      postAttnNorm,
			PostFFNNorm:       postFFNNorm,
			OutputScale:       outScale[0],
			IsSWA:             isSWA,
			HasKV:             hasKV,
			UsesKAsV:          hasKV && !hasV,
			KVCacheSlot:       kvSlots[l],
			FFNHiddenDim:      gateRows,
			PerLayerInputGate: perLayerGate,
			PerLayerProj:      perLayerProj,
			PerLayerPostNorm:  perLayerPostNorm,
		}
		if isSWA {
			layer.RopeDimension = min(localRopeDim, headDim)
			if localInv == nil {
				localInv = gemma4RopeInvFreq(localRopeTheta, headDim, layer.RopeDimension, nil)
			}
			layer.RopeInvFreq = localInv
		} else {
			layer.RopeDimension = min(globalRopeDim, headDim)
			if globalInv == nil {
				globalInv = gemma4RopeInvFreq(globalRopeTheta, headDim, layer.RopeDimension, ropeFactors)
			}
			if len(globalInv) < layer.RopeDimension/2 {
				return config, Gemma4Weights{}, fmt.Errorf("gemma4 global proportional RoPE has %d factors, need %d", len(globalInv), layer.RopeDimension/2)
			}
			layer.RopeInvFreq = globalInv
		}
		if layer.RopeDimension <= 0 || layer.RopeDimension%2 != 0 {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d invalid RoPE dimension %d", l, layer.RopeDimension)
		}
		layers = append(layers, layer)
		maxHeadDim = max(maxHeadDim, headDim)
		maxKVHeads = max(maxKVHeads, kvHeads)
		maxValueDim = max(maxValueDim, valueDim)
		maxFFNDim = max(maxFFNDim, gateRows)
		if l == 0 || (l+1)%8 == 0 || l+1 == config.NLayers {
			fmt.Fprintf(logw, "  Loaded Gemma 4 layer %d/%d\n", l+1, config.NLayers)
		}
	}
	for l, layer := range layers {
		if layer.KVCacheSlot < 0 || layer.KVCacheSlot >= len(layers) {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d has invalid KV cache slot %d", l, layer.KVCacheSlot)
		}
		source := layers[layer.KVCacheSlot]
		if !source.HasKV || source.IsSWA != layer.IsSWA || source.ValueDim != layer.HeadDim || source.NKVHeads <= 0 {
			return config, Gemma4Weights{}, fmt.Errorf("gemma4 layer %d shared-KV source %d is incompatible (source swa=%v kv=%v value=%d heads=%d; query swa=%v head=%d)", l, layer.KVCacheSlot, source.IsSWA, source.HasKV, source.ValueDim, source.NKVHeads, layer.IsSWA, layer.HeadDim)
		}
	}

	// DecodeBuffer and the generic workspace allocator need maxima, while the
	// native forward loop always reads the exact dimensions from each layer.
	config.NKVHeads = maxKVHeads
	config.HeadDim = maxHeadDim
	config.ValueDim = maxValueDim
	config.KVDim = maxKVHeads * maxValueDim
	config.KVMul = max(1, config.NHeads/maxKVHeads)
	// DecodeBuffer uses HiddenDim for its reusable activation slabs. Include
	// an expert or PLE gate if it is wider than the ordinary dense FFN, so the
	// first token of an otherwise valid future Gemma 4 variant does not grow
	// scratch on its hot path.
	config.HiddenDim = max(maxFFNDim, max(config.ExpertFeedForwardDim, perLayerDim))
	config.RopeDimensionCount = max(globalRopeDim, localRopeDim)
	config.AttentionScale = 1 // Gemma4Attention.self.scaling = 1.0
	config.UseGELU = true
	if config.EmbeddingScale == 0 {
		config.EmbeddingScale = float32(math.Sqrt(float64(config.Dim)))
	}
	return config, Gemma4Weights{
		TokenEmbd:  tokenEmbd,
		OutputNorm: outputNorm,
		Output:     output,
		Layers:     layers,
		MoE:        moes,
		PerLayer:   perLayer,
		Native:     true,
	}, nil
}

func gemma4MatrixShape(tensors map[string]TensorInfo, name string) (rows, cols int, err error) {
	info, ok := tensors[name]
	if !ok {
		return 0, 0, fmt.Errorf("missing tensor: %s", name)
	}
	if len(info.Dims) != 2 {
		return 0, 0, fmt.Errorf("gemma4 tensor %s must be a matrix, got dimensions %v", name, info.Dims)
	}
	rows, cols = int(info.Dims[1]), int(info.Dims[0])
	if rows <= 0 || cols <= 0 {
		return 0, 0, fmt.Errorf("gemma4 tensor %s has invalid shape [%d,%d]", name, rows, cols)
	}
	return rows, cols, nil
}

// gemma4KVCacheSlots maps native Gemma 4 blocks to the physical KV cache.
// The first n-shared blocks own K/V. Query-only tail blocks read the latest
// owned slot of their own attention class, so E2B's 35-20 layout is exactly
// [0..14 self], then local/SWA -> 13 and global/full -> 14. The mapping is
// derived from the declared SWA pattern rather than a fragile layer constant.
func gemma4KVCacheSlots(swaPattern []bool, shared int) ([]int, int, error) {
	nLayers := len(swaPattern)
	if nLayers == 0 || shared < 0 || shared >= nLayers {
		return nil, 0, fmt.Errorf("gemma4 invalid shared-KV layout: layers=%d shared=%d", nLayers, shared)
	}
	owned := nLayers - shared
	slots := make([]int, nLayers)
	lastByKind := map[bool]int{}
	for layer := 0; layer < owned; layer++ {
		slots[layer] = layer
		lastByKind[swaPattern[layer]] = layer
	}
	for layer := owned; layer < nLayers; layer++ {
		slot, ok := lastByKind[swaPattern[layer]]
		if !ok {
			kind := "global/full"
			if swaPattern[layer] {
				kind = "local/SWA"
			}
			return nil, 0, fmt.Errorf("gemma4 shared-KV tail layer %d has no preceding %s K/V source", layer, kind)
		}
		slots[layer] = slot
	}
	return slots, owned, nil
}

func nativeGemma4KVCacheLayerCount(weights Gemma4Weights) int {
	if !weights.Native || len(weights.Layers) == 0 {
		return 0
	}
	count := 0
	for _, layer := range weights.Layers {
		if layer.HasKV {
			count = max(count, layer.KVCacheSlot+1)
		}
	}
	return count
}

func gemma4RopeInvFreq(theta float32, headDim, ropeDim int, factors []float32) []float32 {
	config := Config{
		HeadDim:            headDim,
		RopeDimensionCount: ropeDim,
		RopeTheta:          theta,
		RopeFactorsShort:   factors,
	}
	inv, _ := buildRopeInvFreq(config, headDim)
	return inv
}

func forwardNativeGemma4BodyInto(config Config, weights Gemma4Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	dim := config.Dim
	weights.TokenEmbd.RowInto(int(token), dim, &buf.X)
	if config.EmbeddingScale != 1 {
		ScaleF32(buf.X[:dim], config.EmbeddingScale)
	}
	if weights.PerLayer != nil {
		prepareNativeGemma4PerLayerInputs(config, weights.PerLayer, token, buf)
	}
	for il := range weights.Layers {
		layer := weights.Layers[il]
		if !layer.Native || layer.KVCacheSlot < 0 || layer.KVCacheSlot >= len(weights.Layers) || cache == nil || layer.KVCacheSlot >= cache.layerCount() {
			panic("invalid native Gemma 4 layer")
		}
		kvSource := weights.Layers[layer.KVCacheSlot]
		if !kvSource.HasKV || kvSource.ValueDim != layer.HeadDim || kvSource.NKVHeads <= 0 {
			panic("invalid native Gemma 4 shared-KV source")
		}
		qLen := config.NHeads * layer.HeadDim
		kLen := kvSource.NKVHeads * kvSource.HeadDim
		vLen := kvSource.NKVHeads * kvSource.ValueDim
		normalizeDecoderInto(config, buf.X[:dim], layer.AttnNorm, nil, &buf.XN)
		layer.AttnQ.MatvecInto(buf.XN[:dim], &buf.Q)
		ensureLenNoClear(&buf.Q, qLen)
		perHeadRMSNormInPlace(buf.Q[:qLen], layer.HeadDim, config.NHeads, layer.AttnQNorm, config.RMSNormEps)
		ropeHalf, ropePairs := prepareRopeScratch(pos, layer.HeadDim, layer.RopeDimension, layer.RopeInvFreq, 1, &buf.RopeSin, &buf.RopeCos)
		// Gemma's HF layout uses the split-half (NeoX) ordering, including the
		// proportional global RoPE variant.
		applyPreparedRope(buf.Q[:qLen], layer.HeadDim, config.NHeads, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, false)
		if layer.HasKV {
			layer.AttnK.MatvecInto(buf.XN[:dim], &buf.K)
			ensureLenNoClear(&buf.K, kLen)
			ensureLenNoClear(&buf.V, vLen)
			if layer.UsesKAsV {
				copy(buf.V[:vLen], buf.K[:kLen])
			} else {
				layer.AttnV.MatvecInto(buf.XN[:dim], &buf.V)
			}
			perHeadRMSNormInPlace(buf.K[:kLen], kvSource.HeadDim, kvSource.NKVHeads, layer.AttnKNorm, config.RMSNormEps)
			perHeadRMSNormUnitInPlace(buf.V[:vLen], kvSource.ValueDim, kvSource.NKVHeads, config.RMSNormEps)
			applyPreparedRope(buf.K[:kLen], kvSource.HeadDim, kvSource.NKVHeads, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, false)
			cache.storeKV(layer.KVCacheSlot, pos, buf.K[:kLen], buf.V[:vLen])
		}

		attnOutLen := config.NHeads * kvSource.ValueDim
		clear(buf.AttnOut[:attnOutLen])
		attnStart := 0
		if layer.IsSWA {
			attnStart = max(0, pos-config.SlidingWindow)
		}
		kvMul := config.NHeads / kvSource.NKVHeads
		for h := range config.NHeads {
			kvH := h / kvMul
			qOff := h * layer.HeadDim
			outOff := h * kvSource.ValueDim
			cache.attendHead(layer.KVCacheSlot, kvH, buf.Q[qOff:qOff+layer.HeadDim], layer.HeadDim, kvSource.ValueDim,
				attnStart, pos, 1, 0, buf.AttnOut[outOff:outOff+kvSource.ValueDim])
		}
		layer.AttnOutput.MatvecInto(buf.AttnOut[:attnOutLen], &buf.Proj)
		rmsNormInto(buf.Proj[:dim], layer.PostAttnNorm, config.RMSNormEps, &buf.Proj)
		addInPlace(buf.X[:dim], buf.Proj[:dim])

		var moe *Gemma4MoEWeights
		if il < len(weights.MoE) {
			moe = weights.MoE[il]
		}
		if moe == nil {
			normalizeDecoderInto(config, buf.X[:dim], layer.FFNNorm, nil, &buf.XN2)
			layer.FFNGate.MatvecInto(buf.XN2[:dim], &buf.Gate)
			layer.FFNUp.MatvecInto(buf.XN2[:dim], &buf.Up)
			hiddenDim := layer.FFNHiddenDim
			ensureLenNoClear(&buf.Hidden, hiddenDim)
			for i := range hiddenDim {
				buf.Hidden[i] = geluTanh(buf.Gate[i]) * buf.Up[i]
			}
			layer.FFNDown.MatvecInto(buf.Hidden[:hiddenDim], &buf.Proj)
			rmsNormInto(buf.Proj[:dim], layer.PostFFNNorm, config.RMSNormEps, &buf.Proj)
			addInPlace(buf.X[:dim], buf.Proj[:dim])
		} else {
			forwardNativeGemma4MoE(config, layer, moe, buf)
		}
		if weights.PerLayer != nil {
			perLayerDim := weights.PerLayer.Dim
			off := il * perLayerDim
			forwardNativeGemma4PerLayerResidual(config, layer, buf.Gemma4PLE[off:off+perLayerDim], buf)
		}
		if layer.OutputScale != 1 {
			ScaleF32(buf.X[:dim], layer.OutputScale)
		}
	}
	rmsNormInto(buf.X[:dim], weights.OutputNorm, config.RMSNormEps, &buf.XN)
}

// prepareNativeGemma4PerLayerInputs implements E2B's token-conditioned PLE
// input graph. The regular token embedding is already sqrt(dim)-scaled when
// this runs. Each layer receives (RMSNorm(model_proj(x))/sqrt(dim) +
// sqrt(ple_dim)*per_layer_token_embedding[token, layer]) / sqrt(2).
func prepareNativeGemma4PerLayerInputs(config Config, weights *Gemma4PerLayerWeights, token uint32, buf *DecodeBuffer) {
	if weights == nil || weights.Dim <= 0 || len(weights.ProjNorm) != weights.Dim {
		panic("invalid native Gemma 4 per-layer embedding weights")
	}
	total := weights.Dim * config.NLayers
	if total <= 0 || total/weights.Dim != config.NLayers {
		panic("invalid native Gemma 4 per-layer embedding width")
	}
	weights.TokenEmbd.RowInto(int(token), total, &buf.Gemma4PLEInput)
	weights.ModelProj.MatvecInto(buf.X[:config.Dim], &buf.Gemma4PLE)
	if len(buf.Gemma4PLEInput) != total || len(buf.Gemma4PLE) != total {
		panic("native Gemma 4 per-layer embedding projection has an invalid shape")
	}
	projScale := float32(1 / math.Sqrt(float64(config.Dim)))
	inputScale := float32(math.Sqrt(float64(weights.Dim)))
	combineScale := float32(1 / math.Sqrt2)
	for layer := 0; layer < config.NLayers; layer++ {
		off := layer * weights.Dim
		projected := buf.Gemma4PLE[off : off+weights.Dim]
		rawInput := buf.Gemma4PLEInput[off : off+weights.Dim]
		gemma4RMSNormInPlace(projected, weights.ProjNorm, config.RMSNormEps)
		for i := range projected {
			projected[i] = (projected[i]*projScale + rawInput[i]*inputScale) * combineScale
		}
	}
}

func gemma4RMSNormInPlace(values, weight []float32, eps float32) {
	if len(values) == 0 || len(weight) != len(values) {
		panic("invalid native Gemma 4 RMS norm")
	}
	ss := DotF32(values, values)
	scale := float32(1 / math.Sqrt(float64(ss/float32(len(values))+eps)))
	mulScaleF32(values, weight, scale, values)
}

// forwardNativeGemma4PerLayerResidual is the per-block PLE residual after
// the normal FFN: GELU(inp_gate(x)) * prepared_layer_input -> proj -> RMSNorm
// -> add(x). It deliberately uses exact GELU, matching ggml_gelu in the
// reference E2B graph (the ordinary Gemma FFN remains tanh-GELU).
func forwardNativeGemma4PerLayerResidual(config Config, layer Gemma4LayerWeights, input []float32, buf *DecodeBuffer) {
	if len(input) == 0 || len(layer.PerLayerPostNorm) != config.Dim {
		panic("invalid native Gemma 4 per-layer residual")
	}
	layer.PerLayerInputGate.MatvecInto(buf.X[:config.Dim], &buf.Gate)
	if len(buf.Gate) != len(input) {
		panic("native Gemma 4 per-layer input gate has an invalid shape")
	}
	for i := range input {
		buf.Gate[i] = geluExact(buf.Gate[i]) * input[i]
	}
	layer.PerLayerProj.MatvecInto(buf.Gate[:len(input)], &buf.Proj)
	rmsNormInto(buf.Proj[:config.Dim], layer.PerLayerPostNorm, config.RMSNormEps, &buf.Proj)
	addInPlace(buf.X[:config.Dim], buf.Proj[:config.Dim])
}

// forwardNativeGemma4MoE implements Gemma 4 26B A4B's FFN block.  It is
// deliberately not sparseMoEForward: Gemma has an always-on dense GEGLU
// branch, uses a separately scaled router input, and applies three distinct
// branch/sum norms before the residual.
func forwardNativeGemma4MoE(config Config, layer Gemma4LayerWeights, moe *Gemma4MoEWeights, buf *DecodeBuffer) {
	dim := config.Dim
	if moe == nil || moe.ExpertUsed <= 0 || moe.ExpertUsed > moe.Gate.Experts ||
		moe.Gate.Experts != moe.Up.Experts || moe.Gate.Experts != moe.Down.Experts ||
		len(moe.RouterScale) != dim || len(moe.DownScale) != moe.Gate.Experts ||
		moe.Gate.Input != dim || moe.Up.Input != dim || moe.Down.Output != dim ||
		moe.Gate.Output != moe.Up.Output || moe.Down.Input != moe.Gate.Output {
		panic("invalid native Gemma 4 MoE weights")
	}

	// The normal ffn_* tensors stay meaningful in A4B: they form the shared,
	// always-on dense GEGLU branch, rather than a fallback expert.
	normalizeDecoderInto(config, buf.X[:dim], layer.FFNNorm, nil, &buf.XN2)
	layer.FFNGate.MatvecInto(buf.XN2[:dim], &buf.Gate)
	layer.FFNUp.MatvecInto(buf.XN2[:dim], &buf.Up)
	hiddenDim := layer.FFNHiddenDim
	ensureLenNoClear(&buf.Hidden, hiddenDim)
	for i := range hiddenDim {
		buf.Hidden[i] = geluTanh(buf.Gate[i]) * buf.Up[i]
	}
	layer.FFNDown.MatvecInto(buf.Hidden[:hiddenDim], &buf.AttnProj)
	rmsNormInto(buf.AttnProj[:dim], moe.PostNorm1, config.RMSNormEps, &buf.AttnProj)

	// The router sees RMSNorm(attn_out) * ffn_gate_inp.scale / sqrt(dim),
	// whereas the experts see the separately learned pre_ffw_norm_2 input.
	rmsNormInto(buf.X[:dim], nil, config.RMSNormEps, &buf.XN)
	mulScaleF32(buf.XN[:dim], moe.RouterScale, 1/float32(math.Sqrt(float64(dim))), buf.XN[:dim])
	moe.Router.MatvecInto(buf.XN[:dim], &buf.RouterLogits)
	if len(buf.RouterLogits) != moe.Gate.Experts {
		panic("native Gemma 4 MoE router has an invalid shape")
	}
	selected := selectTopExperts(buf.RouterLogits, moe.ExpertUsed, &buf.TopExperts)
	routing := sparseMoERoutingWeights(buf.RouterLogits, selected, true, &buf.ExpertProbs)

	normalizeDecoderInto(config, buf.X[:dim], moe.PreNorm2, nil, &buf.XN2)
	ensureLenNoClear(&buf.Proj, dim)
	clear(buf.Proj[:dim])
	for i, choice := range selected {
		if !expertMatvec2Into(moe.Gate, moe.Up, choice.Index, buf.XN2[:dim], &buf.Q4KXSums, &buf.Gate, &buf.Up) {
			expertMatvecInto(moe.Gate, choice.Index, buf.XN2[:dim], &buf.Gate, &buf.ExpertRow)
			expertMatvecInto(moe.Up, choice.Index, buf.XN2[:dim], &buf.Up, &buf.ExpertRow)
		}
		ensureLenNoClear(&buf.Hidden, moe.Gate.Output)
		for j := range moe.Gate.Output {
			buf.Hidden[j] = geluTanh(buf.Gate[j]) * buf.Up[j]
		}
		expertMatvecInto(moe.Down, choice.Index, buf.Hidden[:moe.Gate.Output], &buf.MOE, &buf.ExpertRow)
		ScaleF32(buf.MOE[:dim], moe.DownScale[choice.Index])
		AxpyF32(buf.Proj[:dim], routing[i], buf.MOE[:dim])
	}
	rmsNormInto(buf.Proj[:dim], moe.PostNorm2, config.RMSNormEps, &buf.Proj)
	addInPlace(buf.Proj[:dim], buf.AttnProj[:dim])
	rmsNormInto(buf.Proj[:dim], layer.PostFFNNorm, config.RMSNormEps, &buf.Proj)
	addInPlace(buf.X[:dim], buf.Proj[:dim])
}

func projectNativeGemma4Logits(config Config, weights Gemma4Weights, buf *DecodeBuffer, logits *[]float32) {
	weights.Output.MatvecInto(buf.XN[:config.Dim], logits)
	if config.LogitScale != 1 {
		ScaleF32(*logits, 1/config.LogitScale)
	}
	if config.FinalLogitSoftcap > 0 {
		softcapF32(*logits, config.FinalLogitSoftcap)
	}
}

// perHeadRMSNormUnitInPlace is Gemma 4's V normalization.  Unlike Q/K it
// has no learned norm weight, so RMS-normalization is applied directly.
func perHeadRMSNormUnitInPlace(vec []float32, headDim, nHeads int, eps float32) {
	if headDim <= 0 {
		return
	}
	for h := range nHeads {
		off := h * headDim
		if off+headDim > len(vec) {
			break
		}
		sub := vec[off : off+headDim]
		ss := DotF32(sub, sub)
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+eps)))
		ScaleF32(sub, scale)
	}
}

func releaseGemma4MetalWeights(weights *Gemma4Weights) {
	if weights == nil || !weights.Native {
		if weights != nil {
			releaseModelMetalWeights(&weights.Standard)
		}
		return
	}
	seen := map[*MetalWeight]bool{}
	release := func(w *Weight) {
		if w == nil || w.Metal == nil || seen[w.Metal] {
			return
		}
		releaseMetalWeight(w.Metal)
		seen[w.Metal] = true
		w.Metal = nil
	}
	release(&weights.TokenEmbd)
	release(&weights.Output)
	if weights.PerLayer != nil {
		release(&weights.PerLayer.TokenEmbd)
		release(&weights.PerLayer.ModelProj)
	}
	for i := range weights.Layers {
		layer := &weights.Layers[i]
		release(&layer.AttnQ)
		release(&layer.AttnK)
		release(&layer.AttnV)
		release(&layer.AttnOutput)
		release(&layer.FFNGate)
		release(&layer.FFNUp)
		release(&layer.FFNDown)
		release(&layer.PerLayerInputGate)
		release(&layer.PerLayerProj)
	}
	for _, moe := range weights.MoE {
		if moe == nil {
			continue
		}
		release(&moe.Router)
		release(&moe.Gate.Weight)
		release(&moe.Up.Weight)
		release(&moe.Down.Weight)
	}
}
