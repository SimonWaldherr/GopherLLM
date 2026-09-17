package gopherllm

import "os"

// Keep native batching separate from canBatchPrefill: embedding pooling and
// the standard-graph autotuner still require ModelWeights, not Gemma's graph.
func (r *Runner) canBatchNativeGemma4() bool {
	if r.kind != loadedGemma4 || !r.gemma4.Native || r.outOfCore || os.Getenv("GOPHERLLM_NO_BATCH_PREFILL") != "" {
		return false
	}
	for _, m := range r.gemma4.MoE {
		if m != nil {
			return false
		}
	}
	for _, l := range r.gemma4.Layers {
		for _, w := range []Weight{l.AttnQ, l.AttnK, l.AttnV, l.AttnOutput, l.FFNGate, l.FFNUp, l.FFNDown, l.PerLayerInputGate, l.PerLayerProj} {
			if (w.Metal != nil && !metalGemmaEnabled()) || w.GPU != nil {
				return false
			}
		}
	}
	return len(r.gemma4.Layers) > 0
}

type gemma4BatchScratch struct {
	geluInput, geluOutput []float32
	tokens                []DecodeBuffer
	inputs, outputs       [][]float32
}

// Only allocate the fields native Gemma uses. In particular, per-token
// vocabulary logits and generic decoder/MLA/MoE scratch are not needed.
func (b *gemma4BatchScratch) resize(n int) {
	if len(b.tokens) < n {
		b.tokens = append(b.tokens, make([]DecodeBuffer, n-len(b.tokens))...)
	}
	if cap(b.inputs) < n {
		b.inputs = make([][]float32, n)
		b.outputs = make([][]float32, n)
	}
	b.inputs, b.outputs = b.inputs[:n], b.outputs[:n]
}

func (b *gemma4BatchScratch) project(w Weight, input, output func(*DecodeBuffer) *[]float32) {
	cols, rows := w.Cols, w.Rows
	// Owned F32 weights may omit geometry; Weight.MatvecInto infers it
	// from the activation width, and native batch must use the same contract.
	if w.F32 != nil {
		cols = len(*input(&b.tokens[0]))
		rows = len(w.F32) / cols
	}
	for i := range b.inputs {
		v := &b.tokens[i]
		dst := output(v)
		ensureLenNoClear(dst, rows)
		b.inputs[i] = (*input(v))[:cols]
		b.outputs[i] = *dst
	}
	matvecBatch(w, b.inputs, b.outputs)
}

// Layer-wise prefill reuses each matrix across all tokens in the chunk. The
// scalar graph's norms, RoPE, K-as-V copy and per-layer residuals are unchanged.
// Dense native graphs only; routed MoE continues through token decode.
func forwardNativeGemma4BatchInto(config Config, weights Gemma4Weights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, computeLast bool, logits *[]float32) {
	if len(tokens) == 0 {
		return
	}
	b := &buf.gemmaBatch
	b.resize(len(tokens))
	dim := config.Dim
	for t, token := range tokens {
		v := &b.tokens[t]
		weights.TokenEmbd.RowInto(int(token), dim, &v.X)
		if config.EmbeddingScale != 1 {
			ScaleF32(v.X, config.EmbeddingScale)
		}
		if weights.PerLayer != nil {
			weights.PerLayer.TokenEmbd.RowInto(int(token), weights.PerLayer.Dim*config.NLayers, &v.Gemma4PLEInput)
		}
	}
	if weights.PerLayer != nil {
		b.project(weights.PerLayer.ModelProj, func(v *DecodeBuffer) *[]float32 { return &v.X }, func(v *DecodeBuffer) *[]float32 { return &v.Gemma4PLE })
		for t := range tokens {
			finishNativeGemma4PerLayerInputs(config, weights.PerLayer, &b.tokens[t])
		}
	}
	for il, layer := range weights.Layers {
		source := weights.Layers[layer.KVCacheSlot]
		for t := range tokens {
			v := &b.tokens[t]
			normalizeDecoderInto(config, v.X, layer.AttnNorm, nil, &v.XN)
		}
		b.project(layer.AttnQ, func(v *DecodeBuffer) *[]float32 { return &v.XN }, func(v *DecodeBuffer) *[]float32 { return &v.Q })
		if layer.HasKV {
			b.project(layer.AttnK, func(v *DecodeBuffer) *[]float32 { return &v.XN }, func(v *DecodeBuffer) *[]float32 { return &v.K })
			if !layer.UsesKAsV {
				b.project(layer.AttnV, func(v *DecodeBuffer) *[]float32 { return &v.XN }, func(v *DecodeBuffer) *[]float32 { return &v.V })
			}
		}
		for t := range tokens {
			gemma4AttentionInto(config, layer, source, cache, &b.tokens[t], startPos+t)
		}
		b.project(layer.AttnOutput, func(v *DecodeBuffer) *[]float32 { return &v.AttnOut }, func(v *DecodeBuffer) *[]float32 { return &v.Proj })
		for t := range tokens {
			v := &b.tokens[t]
			rmsNormInto(v.Proj, layer.PostAttnNorm, config.RMSNormEps, &v.Proj)
			addInPlace(v.X, v.Proj)
			normalizeDecoderInto(config, v.X, layer.FFNNorm, nil, &v.XN2)
		}
		if !b.geluFFN(layer) {
			b.project(layer.FFNGate, func(v *DecodeBuffer) *[]float32 { return &v.XN2 }, func(v *DecodeBuffer) *[]float32 { return &v.Gate })
			b.project(layer.FFNUp, func(v *DecodeBuffer) *[]float32 { return &v.XN2 }, func(v *DecodeBuffer) *[]float32 { return &v.Up })
			for t := range tokens {
				v := &b.tokens[t]
				ensureLenNoClear(&v.Hidden, layer.FFNHiddenDim)
				geluMulF32(v.Gate, v.Up, v.Hidden)
			}
			b.project(layer.FFNDown, func(v *DecodeBuffer) *[]float32 { return &v.Hidden }, func(v *DecodeBuffer) *[]float32 { return &v.Proj })
		}
		for t := range tokens {
			v := &b.tokens[t]
			rmsNormInto(v.Proj, layer.PostFFNNorm, config.RMSNormEps, &v.Proj)
			addInPlace(v.X, v.Proj)
		}
		if weights.PerLayer != nil {
			d := weights.PerLayer.Dim
			b.project(layer.PerLayerInputGate, func(v *DecodeBuffer) *[]float32 { return &v.X }, func(v *DecodeBuffer) *[]float32 { return &v.Gate })
			for t := range tokens {
				v := &b.tokens[t]
				input := v.Gemma4PLE[il*d : il*d+d]
				for i := range input {
					v.Gate[i] = geluExact(v.Gate[i]) * input[i]
				}
			}
			b.project(layer.PerLayerProj, func(v *DecodeBuffer) *[]float32 { return &v.Gate }, func(v *DecodeBuffer) *[]float32 { return &v.Proj })
			for t := range tokens {
				v := &b.tokens[t]
				rmsNormInto(v.Proj, layer.PerLayerPostNorm, config.RMSNormEps, &v.Proj)
				addInPlace(v.X, v.Proj)
			}
		}
		if layer.OutputScale != 1 {
			for t := range tokens {
				ScaleF32(b.tokens[t].X, layer.OutputScale)
			}
		}

	}
	last := &b.tokens[len(tokens)-1]
	ensureLenNoClear(&buf.X, dim)
	copy(buf.X, last.X)
	rmsNormInto(buf.X, weights.OutputNorm, config.RMSNormEps, &buf.XN)
	if computeLast && logits != nil {
		projectNativeGemma4Logits(config, weights, buf, logits)
	}
}

func (b *gemma4BatchScratch) geluFFN(l Gemma4LayerWeights) bool {
	if !metalGemmaEnabled() || l.FFNGate.Metal == nil || l.FFNUp.Metal == nil || l.FFNDown.Metal == nil || len(b.inputs) == 0 || len(b.inputs) > 256 {
		return false
	}
	count := len(b.inputs)
	cols := l.FFNGate.Cols
	rows := l.FFNDown.Rows
	if cols <= 0 || rows <= 0 {
		return false
	}
	ensureLenNoClear(&b.geluInput, count*cols)
	for i := 0; i < count; i++ {
		if len(b.tokens[i].XN2) != cols {
			return false
		}
		copy(b.geluInput[i*cols:(i+1)*cols], b.tokens[i].XN2)
	}
	if !matvecMetalGeGLUInto(l.FFNGate, l.FFNUp, l.FFNDown, b.geluInput, count, &b.geluOutput) {
		return false
	}
	for i := 0; i < count; i++ {
		ensureLenNoClear(&b.tokens[i].Proj, rows)
		copy(b.tokens[i].Proj, b.geluOutput[i*rows:(i+1)*rows])
	}
	return true
}
