package gopherllm

import (
	"fmt"
	"io"
	"math"
)

// BERTWeights is the encoder-only transformer graph used by GGUF BERT
// embeddings, including Nomic Embed and Granite Embedding models. Unlike the
// causal decoder graph, BERT attends over the complete input in both
// directions and uses LayerNorm after each residual addition.
type BERTWeights struct {
	TokenEmbd     Weight
	PositionEmbd  Weight
	TokenTypes    []float32
	EmbeddingNorm []float32
	EmbeddingBias []float32
	Layers        []BERTLayerWeights
	Epsilon       float32
	PoolingType   uint32
	UseRoPE       bool
}

type BERTLayerWeights struct {
	Q, K, V, Output              Weight
	QB, KB, VB, OutputB          []float32
	AttentionNorm, AttentionBias []float32
	FFNUp, FFNDown, FFNGate      Weight
	FFNUpB, FFNDownB             []float32
	OutputNorm, OutputBias       []float32
}

// LoadBERTModel loads an encoder-only BERT or Nomic-BERT GGUF. Both use the
// same canonical tensor names; their architecture-specific metadata namespace
// is read from ConfigFromGGUF. This makes Nomic and Granite embedding models
// usable by /embeddings and the history-RAG index without treating them as
// chat models.
func LoadBERTModel(data []byte, gguf *GGUFFile, borrowQuantized, prepareQuantized, useMetal bool, logw io.Writer) (Config, BERTWeights, error) {
	if logw == nil {
		logw = io.Discard
	}
	config := ConfigFromGGUF(gguf)
	if config.Dim <= 0 || config.NLayers <= 0 || config.NHeads <= 0 || config.HiddenDim <= 0 {
		return config, BERTWeights{}, fmt.Errorf("invalid bert configuration: dim=%d layers=%d heads=%d hidden=%d", config.Dim, config.NLayers, config.NHeads, config.HiddenDim)
	}
	if config.HeadDim <= 0 || config.HeadDim*config.NHeads != config.Dim {
		return config, BERTWeights{}, fmt.Errorf("invalid bert attention shape: dim=%d heads=%d head_dim=%d", config.Dim, config.NHeads, config.HeadDim)
	}
	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal)
	}

	tokenEmbd, err := load("token_embd.weight")
	if err != nil {
		return config, BERTWeights{}, err
	}
	var positionEmbd Weight
	if config.Arch == "bert" {
		positionEmbd, err = load("position_embd.weight")
		if err != nil {
			return config, BERTWeights{}, err
		}
	}
	embeddingNorm, err := loadF32Vec(data, gguf.DataOffset, "token_embd_norm.weight", tensors, inferred)
	if err != nil {
		return config, BERTWeights{}, err
	}
	embeddingBias := loadOptionalF32Vec(data, gguf.DataOffset, "token_embd_norm.bias", tensors, inferred, config.Dim)
	tokenTypes := loadOptionalF32VecNil(data, gguf.DataOffset, "token_types.weight", tensors, inferred)

	layers := make([]BERTLayerWeights, 0, config.NLayers)
	for l := range config.NLayers {
		prefix := fmt.Sprintf("blk.%d.", l)
		q, err := load(prefix + "attn_q.weight")
		if err != nil {
			return config, BERTWeights{}, err
		}
		k, err := load(prefix + "attn_k.weight")
		if err != nil {
			return config, BERTWeights{}, err
		}
		v, err := load(prefix + "attn_v.weight")
		if err != nil {
			return config, BERTWeights{}, err
		}
		output, err := load(prefix + "attn_output.weight")
		if err != nil {
			return config, BERTWeights{}, err
		}
		ffnUp, err := load(prefix + "ffn_up.weight")
		if err != nil {
			return config, BERTWeights{}, err
		}
		ffnDown, err := load(prefix + "ffn_down.weight")
		if err != nil {
			return config, BERTWeights{}, err
		}
		var ffnGate Weight
		if config.Arch == "nomic-bert" {
			ffnGate, err = load(prefix + "ffn_gate.weight")
			if err != nil {
				return config, BERTWeights{}, err
			}
		}
		attentionNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"attn_output_norm.weight", tensors, inferred)
		if err != nil {
			return config, BERTWeights{}, err
		}
		outputNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"layer_output_norm.weight", tensors, inferred)
		if err != nil {
			return config, BERTWeights{}, err
		}
		layers = append(layers, BERTLayerWeights{
			Q: q, K: k, V: v, Output: output,
			QB:            loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_q.bias", tensors, inferred, config.Dim),
			KB:            loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_k.bias", tensors, inferred, config.Dim),
			VB:            loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_v.bias", tensors, inferred, config.Dim),
			OutputB:       loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_output.bias", tensors, inferred, config.Dim),
			AttentionNorm: attentionNorm,
			AttentionBias: loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_output_norm.bias", tensors, inferred, config.Dim),
			FFNUp:         ffnUp, FFNDown: ffnDown, FFNGate: ffnGate,
			FFNUpB:     loadOptionalF32Vec(data, gguf.DataOffset, prefix+"ffn_up.bias", tensors, inferred, config.HiddenDim),
			FFNDownB:   loadOptionalF32Vec(data, gguf.DataOffset, prefix+"ffn_down.bias", tensors, inferred, config.Dim),
			OutputNorm: outputNorm,
			OutputBias: loadOptionalF32Vec(data, gguf.DataOffset, prefix+"layer_output_norm.bias", tensors, inferred, config.Dim),
		})
		if l == 0 || l+1 == config.NLayers || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded BERT layer %d/%d\n", l+1, config.NLayers)
		}
	}
	epsilon := gguf.GetF32(config.Arch+".attention.layer_norm_epsilon", 1e-5)
	if epsilon <= 0 {
		epsilon = 1e-5
	}
	return config, BERTWeights{
		TokenEmbd: tokenEmbd, PositionEmbd: positionEmbd, TokenTypes: tokenTypes,
		EmbeddingNorm: embeddingNorm, EmbeddingBias: embeddingBias, Layers: layers,
		Epsilon: epsilon, PoolingType: gguf.GetU32(config.Arch+".pooling_type", 1), UseRoPE: config.Arch == "nomic-bert",
	}, nil
}

func (r *Runner) embedBERT(text string) (EmbeddingResult, error) {
	tokens := r.tok.Encode(text)
	if len(tokens) == 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: input tokenised to zero tokens")
	}
	if r.config.MaxSeqLen > 0 && len(tokens) > r.config.MaxSeqLen {
		return EmbeddingResult{}, fmt.Errorf("embed: input (%d tokens) exceeds the model's context length (%d)", len(tokens), r.config.MaxSeqLen)
	}
	return EmbedBERT(r.config, r.bert, tokens)
}

// EmbedBERT executes one full bidirectional BERT encoder pass and applies the
// GGUF pooling mode (mean=1, CLS=2). It intentionally has no KV cache: every
// token needs to see all other tokens in the input.
func EmbedBERT(config Config, weights BERTWeights, tokens []uint32) (EmbeddingResult, error) {
	n := len(tokens)
	if n == 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: no tokens")
	}
	dim := config.Dim
	if !weights.UseRoPE && weights.PositionEmbd.F32 != nil && len(weights.PositionEmbd.F32) < n*dim {
		return EmbeddingResult{}, fmt.Errorf("embed: input (%d tokens) exceeds position embeddings", n)
	}
	x := make([][]float32, n)
	for i, token := range tokens {
		x[i] = make([]float32, dim)
		weights.TokenEmbd.RowInto(int(token), dim, &x[i])
		if !weights.UseRoPE {
			position := weights.PositionEmbd.Row(i, dim)
			addInPlace(x[i], position)
		}
		if len(weights.TokenTypes) >= dim {
			addInPlace(x[i], weights.TokenTypes[:dim])
		}
		layerNormInto(x[i], weights.EmbeddingNorm, weights.EmbeddingBias, weights.Epsilon, &x[i])
	}

	headDim := config.HeadDim
	scale := float32(1 / math.Sqrt(float64(headDim)))
	ropeInv, ropeMscale := buildRopeInvFreq(config, headDim)
	var ropeSin, ropeCos []float32
	for _, layer := range weights.Layers {
		q := make([][]float32, n)
		k := make([][]float32, n)
		v := make([][]float32, n)
		for i := range n {
			q[i] = layer.Q.Matvec(x[i])
			k[i] = layer.K.Matvec(x[i])
			v[i] = layer.V.Matvec(x[i])
			addInPlace(q[i], layer.QB)
			addInPlace(k[i], layer.KB)
			addInPlace(v[i], layer.VB)
			if weights.UseRoPE {
				half, cached := prepareRopeScratch(i, headDim, config.RopeDimensionCount, ropeInv, ropeMscale, &ropeSin, &ropeCos)
				applyPreparedRope(q[i], headDim, config.NHeads, half, cached, ropeSin, ropeCos, false)
				applyPreparedRope(k[i], headDim, config.NHeads, half, cached, ropeSin, ropeCos, false)
			}
		}
		attention := make([][]float32, n)
		for i := range n {
			attention[i] = make([]float32, dim)
			for h := range config.NHeads {
				off := h * headDim
				scores := make([]float32, n)
				maxScore := float32(-math.MaxFloat32)
				for j := range n {
					scores[j] = DotF32(q[i][off:off+headDim], k[j][off:off+headDim]) * scale
					maxScore = max(maxScore, scores[j])
				}
				denom := float32(0)
				for j := range n {
					scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
					denom += scores[j]
				}
				if denom > 0 {
					for j := range n {
						AxpyF32(attention[i][off:off+headDim], scores[j]/denom, v[j][off:off+headDim])
					}
				}
			}
		}
		for i := range n {
			projected := layer.Output.Matvec(attention[i])
			addInPlace(projected, layer.OutputB)
			addInPlace(projected, x[i])
			layerNormInto(projected, layer.AttentionNorm, layer.AttentionBias, weights.Epsilon, &x[i])
		}
		for i := range n {
			hidden := layer.FFNUp.Matvec(x[i])
			addInPlace(hidden, layer.FFNUpB)
			if weights.UseRoPE {
				gate := layer.FFNGate.Matvec(x[i])
				for j := range hidden {
					hidden[j] *= gate[j] / (1 + float32(math.Exp(float64(-gate[j]))))
				}
			} else {
				for j := range hidden {
					hidden[j] = geluExact(hidden[j])
				}
			}
			output := layer.FFNDown.Matvec(hidden)
			addInPlace(output, layer.FFNDownB)
			addInPlace(output, x[i])
			layerNormInto(output, layer.OutputNorm, layer.OutputBias, weights.Epsilon, &x[i])
		}
	}

	pooled := make([]float32, dim)
	if weights.PoolingType == 2 { // LLAMA_POOLING_TYPE_CLS
		copy(pooled, x[0])
	} else {
		for i := range n {
			addInPlace(pooled, x[i])
		}
		meanPoolInPlace(pooled, n)
	}
	l2NormalizeInPlace(pooled)
	return EmbeddingResult{Embedding: pooled, TokenCount: n}, nil
}

func geluExact(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}

// layerNormInto is BERT's mean-and-variance LayerNorm (as opposed to the
// RMSNorm used by decoder-only model families).
func layerNormInto(x, weight, bias []float32, epsilon float32, out *[]float32) {
	n := len(x)
	ensureLenNoClear(out, n)
	if n == 0 {
		return
	}
	mean := float32(0)
	for _, v := range x {
		mean += v
	}
	mean /= float32(n)
	variance := float32(0)
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	scale := float32(1 / math.Sqrt(float64(variance/float32(n)+epsilon)))
	for i, v := range x {
		w := float32(1)
		if i < len(weight) {
			w = weight[i]
		}
		b := float32(0)
		if i < len(bias) {
			b = bias[i]
		}
		(*out)[i] = (v-mean)*scale*w + b
	}
}

func releaseBERTMetalWeights(weights *BERTWeights) {
	if weights == nil {
		return
	}
	release := func(w *Weight) {
		if w != nil && w.Metal != nil {
			releaseMetalWeight(w.Metal)
			w.Metal = nil
		}
	}
	release(&weights.TokenEmbd)
	release(&weights.PositionEmbd)
	for i := range weights.Layers {
		layer := &weights.Layers[i]
		release(&layer.Q)
		release(&layer.K)
		release(&layer.V)
		release(&layer.Output)
		release(&layer.FFNUp)
		release(&layer.FFNDown)
		release(&layer.FFNGate)
	}
}
