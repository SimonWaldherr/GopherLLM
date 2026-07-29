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
	Q, K, V, Output Weight
	// QKV holds a single [Q; K; V] projection when the GGUF stores the
	// attention projections fused.  Nomic Embed v1.5 uses this layout: its
	// attn_qkv.weight has 3*Dim output rows instead of separate q/k/v tensors.
	QKV                          Weight
	HasQKV                       bool
	QB, KB, VB, QKVB, OutputB    []float32
	AttentionNorm, AttentionBias []float32
	FFNUp, FFNDown, FFNGate      Weight
	FFNUpB, FFNDownB             []float32
	OutputNorm, OutputBias       []float32
}

// bertEmbeddingScratch retains the encoder activation slabs for Runner.Embed.
// A Runner serializes embeddings with genLock, so these are safe to reuse and
// avoid re-allocating several MiB for every repeated API embedding request.
// The exported EmbedBERT function supplies a short-lived instance instead.
type bertEmbeddingScratch struct {
	XFlat, QFlat, KFlat, VFlat, QKVFlat, HiddenFlat, GateFlat []float32
	X, Q, K, V, QKV, Hidden, Gate                             [][]float32
	Position, RopeSin, RopeCos, Scores                        []float32
}

const maxReusableBERTScratchBytes int64 = 128 << 20

func reusableBERTScratch(n, dim, hidden int, useGate, useFusedQKV bool) bool {
	if n <= 0 || dim <= 0 || hidden < 0 {
		return false
	}
	values := int64(n) * (int64(dim)*4 + int64(hidden))
	if useFusedQKV {
		// The combined projection is transient (it is split into Q/K/V before
		// attention), but retaining it in Runner scratch avoids a large
		// per-request allocation for Nomic's fused QKV tensor.
		values += int64(n) * int64(dim) * 3
	}
	if useGate {
		values += int64(n) * int64(hidden)
	}
	return values*4 <= maxReusableBERTScratchBytes
}

// LoadBERTModel loads an encoder-only BERT or Nomic-BERT GGUF. Both use the
// same canonical tensor names; their architecture-specific metadata namespace
// is read from ConfigFromGGUF. This makes Nomic and Granite embedding models
// usable by /embeddings and the history-RAG index without treating them as
// chat models.
func LoadBERTModel(data []byte, gguf *GGUFFile, borrowQuantized, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, BERTWeights, error) {
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
	lazyScalarWeights := len(outOfCore) > 0 && outOfCore[0]
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrowQuantized, prepareQuantized, useMetal, lazyScalarWeights)
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
		var q, k, v, qkv Weight
		hasQKV := false
		if info, ok := tensors[prefix+"attn_qkv.weight"]; ok {
			if len(info.Dims) != 2 || int(info.Dims[0]) != config.Dim || int(info.Dims[1]) != 3*config.Dim {
				return config, BERTWeights{}, fmt.Errorf("invalid fused BERT QKV shape for layer %d: got %v, want [%d %d]", l, info.Dims, config.Dim, 3*config.Dim)
			}
			qkv, err = load(prefix + "attn_qkv.weight")
			if err != nil {
				return config, BERTWeights{}, err
			}
			hasQKV = true
		} else {
			q, err = load(prefix + "attn_q.weight")
			if err != nil {
				return config, BERTWeights{}, err
			}
			k, err = load(prefix + "attn_k.weight")
			if err != nil {
				return config, BERTWeights{}, err
			}
			v, err = load(prefix + "attn_v.weight")
			if err != nil {
				return config, BERTWeights{}, err
			}
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
			Q: q, K: k, V: v, QKV: qkv, HasQKV: hasQKV, Output: output,
			QB:            loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_q.bias", tensors, inferred, config.Dim),
			KB:            loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_k.bias", tensors, inferred, config.Dim),
			VB:            loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_v.bias", tensors, inferred, config.Dim),
			QKVB:          loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_qkv.bias", tensors, inferred, 3*config.Dim),
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
	var scratch *bertEmbeddingScratch
	if reusableBERTScratch(len(tokens), r.config.Dim, r.config.HiddenDim, r.bert.UseRoPE, bertHasFusedQKV(r.bert)) {
		scratch = &r.bertScratch
	}
	matvec := bertBatchMatvec(matvecBERTBatch)
	if r.outOfCore {
		// Batching trades more pages held concurrently for speed. The OOC path
		// intentionally streams one activation at a time so the VM can evict
		// cold model pages between projections.
		matvec = matvecBERTSequential
	}
	return embedBERTWithScratch(r.config, r.bert, tokens, matvec, scratch)
}

func bertHasFusedQKV(weights BERTWeights) bool {
	for i := range weights.Layers {
		if weights.Layers[i].HasQKV {
			return true
		}
	}
	return false
}

// bertBatchMatvec lets the BERT graph use one batched matrix traversal per
// projection while retaining a serial reference implementation for numerical
// regression tests and Metal-only fallbacks.
type bertBatchMatvec func(Weight, [][]float32, [][]float32)

func matvecBERTSequential(w Weight, xs, outs [][]float32) {
	for i := range xs {
		w.MatvecInto(xs[i], &outs[i])
	}
}

// matvecBERTBatch keeps a direct Metal projection on its established
// single-token route because the Metal backend currently exposes matvec, not
// matmul. CPU quantized and F32 weights use the batched prefill kernel, which
// dequantizes/streams every row once for the whole embedding sequence.
func matvecBERTBatch(w Weight, xs, outs [][]float32) {
	if len(xs) < 2 || metalWeightUsesDirect(w.Metal) {
		matvecBERTSequential(w, xs, outs)
		return
	}
	matvecBatch(w, xs, outs)
}

// EmbedBERT executes one full bidirectional BERT encoder pass and applies the
// GGUF pooling mode (mean=1, CLS=2). It intentionally has no KV cache: every
// token needs to see all other tokens in the input.
func EmbedBERT(config Config, weights BERTWeights, tokens []uint32) (EmbeddingResult, error) {
	return embedBERTWithScratch(config, weights, tokens, matvecBERTBatch, nil)
}

// embedBERTWithMatvec keeps the encoder graph independent from its projection
// strategy. The production path batches F32/quantized rows; tests use the
// single-token projection strategy as an exact graph-level regression oracle.
func embedBERTWithMatvec(config Config, weights BERTWeights, tokens []uint32, matvec bertBatchMatvec) (EmbeddingResult, error) {
	return embedBERTWithScratch(config, weights, tokens, matvec, nil)
}

func embedBERTWithScratch(config Config, weights BERTWeights, tokens []uint32, matvec bertBatchMatvec, scratch *bertEmbeddingScratch) (EmbeddingResult, error) {
	n := len(tokens)
	if n == 0 {
		return EmbeddingResult{}, fmt.Errorf("embed: no tokens")
	}
	dim := config.Dim
	if !weights.UseRoPE {
		positionRows := weights.PositionEmbd.Rows
		if weights.PositionEmbd.F32 != nil && dim > 0 {
			positionRows = len(weights.PositionEmbd.F32) / dim
		}
		if positionRows < n {
			return EmbeddingResult{}, fmt.Errorf("embed: input (%d tokens) exceeds position embeddings", n)
		}
	}
	if scratch == nil {
		scratch = &bertEmbeddingScratch{}
	}
	x := reuseBatchViews(&scratch.XFlat, &scratch.X, n, dim)
	ensureLenNoClear(&scratch.Position, dim)
	position := scratch.Position
	for i, token := range tokens {
		weights.TokenEmbd.RowInto(int(token), dim, &x[i])
		if !weights.UseRoPE {
			weights.PositionEmbd.RowInto(i, dim, &position)
			addInPlace(x[i], position)
		}
		if len(weights.TokenTypes) >= dim {
			addInPlace(x[i], weights.TokenTypes[:dim])
		}
		layerNormInto(x[i], weights.EmbeddingNorm, weights.EmbeddingBias, weights.Epsilon, &x[i])
	}

	headDim := config.HeadDim
	scale := float32(1 / math.Sqrt(float64(headDim)))
	var ropeInv []float32
	ropeMscale := float32(1)
	if weights.UseRoPE {
		ropeInv, ropeMscale = buildRopeInvFreq(config, headDim)
	}
	q := reuseBatchViews(&scratch.QFlat, &scratch.Q, n, dim)
	k := reuseBatchViews(&scratch.KFlat, &scratch.K, n, dim)
	v := reuseBatchViews(&scratch.VFlat, &scratch.V, n, dim)
	var qkv [][]float32
	if bertHasFusedQKV(weights) {
		qkv = reuseBatchViews(&scratch.QKVFlat, &scratch.QKV, n, 3*dim)
	}
	hidden := reuseBatchViews(&scratch.HiddenFlat, &scratch.Hidden, n, config.HiddenDim)
	var gate [][]float32
	if weights.UseRoPE {
		gate = reuseBatchViews(&scratch.GateFlat, &scratch.Gate, n, config.HiddenDim)
	}
	for _, layer := range weights.Layers {
		if layer.HasQKV {
			matvec(layer.QKV, x, qkv)
			for i := range n {
				copy(q[i], qkv[i][:dim])
				copy(k[i], qkv[i][dim:2*dim])
				copy(v[i], qkv[i][2*dim:])
			}
		} else {
			matvec(layer.Q, x, q)
			matvec(layer.K, x, k)
			matvec(layer.V, x, v)
		}
		for i := range n {
			if layer.HasQKV {
				if len(layer.QKVB) >= 3*dim {
					addInPlace(q[i], layer.QKVB[:dim])
					addInPlace(k[i], layer.QKVB[dim:2*dim])
					addInPlace(v[i], layer.QKVB[2*dim:])
				}
			} else {
				addInPlace(q[i], layer.QB)
				addInPlace(k[i], layer.KB)
				addInPlace(v[i], layer.VB)
			}
			if weights.UseRoPE {
				half, cached := prepareRopeScratch(i, headDim, config.RopeDimensionCount, ropeInv, ropeMscale, &scratch.RopeSin, &scratch.RopeCos)
				applyPreparedRope(q[i], headDim, config.NHeads, half, cached, scratch.RopeSin, scratch.RopeCos, false)
				applyPreparedRope(k[i], headDim, config.NHeads, half, cached, scratch.RopeSin, scratch.RopeCos, false)
			}
		}
		attendOne := func(i int, scores []float32) {
			for h := range config.NHeads {
				off := h * headDim
				query := q[i][off : off+headDim]
				maxScore := float32(-math.MaxFloat32)
				for j := range n {
					scores[j] = DotF32(query, k[j][off:off+headDim]) * scale
					maxScore = max(maxScore, scores[j])
				}
				denom := float32(0)
				for j := range n {
					scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
					denom += scores[j]
				}
				// The old implementation started each attention output at zero.
				// Clear only after the query has been fully consumed so the same
				// zero-output behavior is retained for non-finite/degenerate rows.
				clear(query)
				if denom > 0 {
					for j := range n {
						AxpyF32(query, scores[j]/denom, v[j][off:off+headDim])
					}
				}
			}
		}
		attend := func(start, end int) {
			scores := make([]float32, n)
			for i := start; i < end; i++ {
				attendOne(i, scores)
			}
		}
		if n >= 64 && config.NHeads > 1 {
			parallelChunks(n, attend)
		} else {
			ensureLenNoClear(&scratch.Scores, n)
			for i := range n {
				attendOne(i, scratch.Scores)
			}
		}
		// q now holds attention output. k is no longer needed, and v can
		// safely receive the projected residual because all attention rows
		// have consumed it.
		matvec(layer.Output, q, v)
		for i := range n {
			addInPlace(v[i], layer.OutputB)
			addInPlace(v[i], x[i])
			layerNormInto(v[i], layer.AttentionNorm, layer.AttentionBias, weights.Epsilon, &x[i])
		}
		matvec(layer.FFNUp, x, hidden)
		if weights.UseRoPE {
			matvec(layer.FFNGate, x, gate)
		}
		for i := range n {
			addInPlace(hidden[i], layer.FFNUpB)
			if weights.UseRoPE {
				for j := range hidden[i] {
					hidden[i][j] *= gate[i][j] / (1 + float32(math.Exp(float64(-gate[i][j]))))
				}
			} else {
				for j := range hidden[i] {
					hidden[i][j] = geluExact(hidden[i][j])
				}
			}
		}
		matvec(layer.FFNDown, hidden, v)
		for i := range n {
			addInPlace(v[i], layer.FFNDownB)
			addInPlace(v[i], x[i])
			layerNormInto(v[i], layer.OutputNorm, layer.OutputBias, weights.Epsilon, &x[i])
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
		release(&layer.QKV)
		release(&layer.Output)
		release(&layer.FFNUp)
		release(&layer.FFNDown)
		release(&layer.FFNGate)
	}
}
