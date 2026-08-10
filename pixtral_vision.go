package gopherllm

import (
	"fmt"
	"io"
	"math"
)

// PixtralVisionConfig holds a Pixtral-style vision encoder's hyperparameters,
// read from a companion "mmproj" GGUF's clip.*/clip.vision.* metadata. Field
// values below (patch_size=14, spatial_merge_size=2, etc.) were confirmed
// against a real mistralai/Ministral-3-3B-Instruct-2512 mmproj GGUF
// (Unsloth's F16 conversion) via --list-metadata/--list-tensors — they
// differ from vanilla Pixtral-12B (patch_size 16, no patch merging), which
// is exactly why every value here is read from metadata, never hardcoded.
type PixtralVisionConfig struct {
	EmbeddingLength   int     // clip.vision.embedding_length
	FeedForwardLength int     // clip.vision.feed_forward_length
	BlockCount        int     // clip.vision.block_count
	HeadCount         int     // clip.vision.attention.head_count (plain MHA -- no separate KV head count)
	HeadDim           int     // clip.vision.attention.head_dim, or EmbeddingLength/HeadCount if absent
	Epsilon           float32 // clip.vision.attention.layer_norm_epsilon
	ImageSize         int     // clip.vision.image_size (legacy/static preprocessing bound, pixels)
	PatchSize         int     // clip.vision.patch_size
	SpatialMergeSize  int     // clip.vision.spatial_merge_size (2x2 merge for Ministral-3)
	ImageMinTokens    int     // dynamic Pixtral input lower bound, after spatial merge
	ImageMaxTokens    int     // dynamic Pixtral input upper bound, after spatial merge
	ProjectionDim     int     // clip.vision.projection_dim -- must match the paired LLM's Dim
	ImageMean         [3]float32
	ImageStd          [3]float32
	UseSiLU           bool // clip.use_silu -- read explicitly, not inferred from projector type
	UseGELU           bool // clip.use_gelu -- read explicitly
	RopeTheta         float32
}

// PixtralVisionLayerWeights is one of the vision tower's transformer blocks:
// RMSNorm pre-norm residual, full (non-causal) multi-head attention with 2D
// RoPE, then a gated SiLU (or GELU, per UseGELU) FFN.
type PixtralVisionLayerWeights struct {
	AttnNorm                []float32 // v.blk.N.ln1
	Q, K, V, Out            Weight    // v.blk.N.attn_{q,k,v,out}.weight
	QB, KB, VB, OutB        []float32 // optional biases (absent in the verified Ministral-3 checkpoint)
	FFNNorm                 []float32 // v.blk.N.ln2
	FFNGate, FFNUp, FFNDown Weight    // v.blk.N.ffn_{gate,up,down}.weight
}

// PixtralVisionWeights is a loaded Pixtral vision tower + projector. See
// LoadPixtralVisionModel's doc comment for the full computational graph.
type PixtralVisionWeights struct {
	PatchEmbd   Weight    // v.patch_embd.weight, reinterpreted as [EmbeddingLength, 3*PatchSize^2] (see LoadPixtralVisionModel)
	PatchEmbdB  []float32 // optional
	PreNorm     []float32 // v.pre_ln.weight -- applied once, before block 0
	Layers      []PixtralVisionLayerWeights
	InputNorm   []float32 // mm.input_norm.weight -- functionally the vision tower's post-encoder norm (see LoadPixtralVisionModel)
	PatchMerger Weight    // mm.patch_merger.weight, [EmbeddingLength*SpatialMergeSize^2 -> EmbeddingLength], no bias, no activation
	Proj1       Weight    // mm.1.weight, [EmbeddingLength -> ProjectionDim]
	Proj1B      []float32
	Proj2       Weight // mm.2.weight, [ProjectionDim -> ProjectionDim]
	Proj2B      []float32
	ImgBreak    []float32 // v.token_embd.img_break -- lives in the TEXT decoder's embedding space, width == the paired LLM's Dim
}

// LoadPixtralVisionModel loads a Pixtral-projector "mmproj" GGUF (companion
// to, and structurally unrelated to, a text-decoder GGUF loaded separately
// via LoadModel/RunnerFromGGUFBytes*). It fails loudly if the file isn't a
// vision encoder or doesn't use the "pixtral" projector type -- this loader
// implements exactly that one graph, not a general CLIP loader.
//
// The full computational graph (confirmed against llama.cpp's own clip.cpp
// graph-builder and HF transformers' Mistral3 modeling code, both read only
// to extract this factual sequence of operations -- nothing below is
// transcribed from either):
//
//  1. Patch embedding: a stride-equals-kernel conv2d, which is mathematically
//     identical to a per-patch linear projection over a flattened
//     [channel][kh][kw] (channel slowest-varying) patch vector -- see the
//     shape check below for why this reinterpretation is safe.
//  2. v.pre_ln: RMSNorm, applied once, before the first transformer block.
//  3. BlockCount transformer blocks, each pre-norm: x += Attn(RMSNorm_ln1(x))
//     (attention uses 2D RoPE on Q/K, full/non-causal, no KV cache -- see
//     EncodeImagePixtral), then x += FFN(RMSNorm_ln2(x)) (gated SiLU/GELU).
//  4. mm.input_norm: RMSNorm on the final per-patch hidden states, applied to
//     every patch before merging -- despite being named/owned by the
//     projector, this is functionally the vision tower's post-encoder norm
//     (there is no separate v.post_ln tensor in this format).
//  5. Spatial merge: non-overlapping SpatialMergeSize x SpatialMergeSize
//     windows of the patch grid are packed into one vector in CHANNEL-MAJOR
//     order (merged[c*merge^2 + dy*merge + dx] = patch[dy,dx][c] -- NOT
//     naive whole-patch concatenation, which would silently produce a
//     plausible-looking but wrong result), then linearly projected back down
//     to EmbeddingLength via mm.patch_merger (no bias, no activation).
//  6. mm.1 (linear) -> GELU -> mm.2 (linear, no activation after) produces
//     the final ProjectionDim-wide embedding, one per merged patch.
//
// Callers splice these embeddings into the paired LLM's input sequence in
// place of image-placeholder tokens (see EncodeImagePixtral's return shape
// and runtime.go's renderMistralInstMessages).
func LoadPixtralVisionModel(data []byte, gguf *GGUFFile, useMetal bool, logw io.Writer) (PixtralVisionConfig, PixtralVisionWeights, error) {
	return loadPixtralVisionModel(data, gguf, useMetal, false, logw)
}

func loadPixtralVisionModel(data []byte, gguf *GGUFFile, useMetal, borrowQuantized bool, logw io.Writer) (PixtralVisionConfig, PixtralVisionWeights, error) {
	if logw == nil {
		logw = io.Discard
	}
	if !gguf.GetBool("clip.has_vision_encoder", false) {
		return PixtralVisionConfig{}, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: clip.has_vision_encoder is false or absent")
	}
	projType, _ := gguf.GetString("clip.projector_type")
	if projType != "pixtral" {
		return PixtralVisionConfig{}, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: unsupported clip.projector_type %q (only \"pixtral\" is implemented)", projType)
	}

	embLen := int(gguf.GetU32("clip.vision.embedding_length", 0))
	headCount := int(gguf.GetU32("clip.vision.attention.head_count", 0))
	blockCount := int(gguf.GetU32("clip.vision.block_count", 0))
	ffnLen := int(gguf.GetU32("clip.vision.feed_forward_length", 0))
	if embLen <= 0 || headCount <= 0 || blockCount <= 0 || ffnLen <= 0 {
		return PixtralVisionConfig{}, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: invalid config embedding_length=%d head_count=%d block_count=%d feed_forward_length=%d", embLen, headCount, blockCount, ffnLen)
	}
	headDim := int(gguf.GetU32("clip.vision.attention.head_dim", 0))
	if headDim <= 0 {
		if embLen%headCount != 0 {
			return PixtralVisionConfig{}, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: embedding_length %d not divisible by head_count %d and no explicit head_dim", embLen, headCount)
		}
		headDim = embLen / headCount
	}
	patchSize := int(gguf.GetU32("clip.vision.patch_size", 0))
	imageSize := int(gguf.GetU32("clip.vision.image_size", 0))
	if patchSize <= 0 || imageSize <= 0 {
		return PixtralVisionConfig{}, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: invalid patch_size=%d image_size=%d", patchSize, imageSize)
	}
	mergeSize := int(gguf.GetU32("clip.vision.spatial_merge_size", 1))
	if mergeSize <= 0 {
		mergeSize = 1
	}
	projDim := int(gguf.GetU32("clip.vision.projection_dim", 0))
	if projDim <= 0 {
		return PixtralVisionConfig{}, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: invalid projection_dim=%d", projDim)
	}

	imgMean := [3]float32{0.48145466, 0.4578275, 0.40821073}
	imgStd := [3]float32{0.26862954, 0.26130258, 0.27577711}
	if mean, ok := gguf.GetF32Array("clip.vision.image_mean"); ok && len(mean) >= 3 {
		copy(imgMean[:], mean)
	}
	if std, ok := gguf.GetF32Array("clip.vision.image_std"); ok && len(std) >= 3 {
		copy(imgStd[:], std)
	}
	// Defensive check against a known llama.cpp convert-script bug (PR
	// #13208) that once wrote image_mean into the image_std slot for
	// Pixtral specifically -- warn rather than silently trust.
	for i := range imgStd {
		if imgStd[i] == imgMean[i] || imgStd[i] <= 0 || imgStd[i] > 1 || imgMean[i] < 0 || imgMean[i] > 1 {
			fmt.Fprintf(logw, "  warning: vision image_mean/image_std for channel %d look suspicious (mean=%v std=%v) -- possible metadata bug (see llama.cpp PR #13208)\n", i, imgMean[i], imgStd[i])
		}
	}

	epsilon := gguf.GetF32("clip.vision.attention.layer_norm_epsilon", 1e-5)
	if epsilon <= 0 {
		epsilon = 1e-5
	}
	useSiLU := gguf.GetBool("clip.use_silu", false)
	useGELU := gguf.GetBool("clip.use_gelu", false)
	if !useSiLU && !useGELU {
		useSiLU = true // every published Pixtral/Ministral-3 checkpoint uses SiLU
	}

	config := PixtralVisionConfig{
		EmbeddingLength: embLen, FeedForwardLength: ffnLen, BlockCount: blockCount,
		HeadCount: headCount, HeadDim: headDim, Epsilon: epsilon,
		ImageSize: imageSize, PatchSize: patchSize, SpatialMergeSize: mergeSize,
		// Pixtral GGUFs do not currently carry these limits as metadata. They
		// are the model-family defaults used by the reference implementation.
		ImageMinTokens: 8, ImageMaxTokens: 1024,
		ProjectionDim: projDim, ImageMean: imgMean, ImageStd: imgStd,
		UseSiLU: useSiLU, UseGELU: useGELU, RopeTheta: 10000,
	}

	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrowQuantized, false, useMetal)
	}

	patchLen := 3 * patchSize * patchSize
	if info, ok := tensors["v.patch_embd.weight"]; !ok {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: tensor v.patch_embd.weight not found")
	} else if len(info.Dims) != 4 || int(info.Dims[3]) != embLen || int(info.Dims[0]*info.Dims[1]*info.Dims[2]) != patchLen {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: v.patch_embd.weight dims %v, want [patch,patch,3,%d] with patch=%d", info.Dims, embLen, patchSize)
	}
	patchEmbd, err := load("v.patch_embd.weight")
	if err != nil {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
	}
	patchEmbdB := loadOptionalF32Vec(data, gguf.DataOffset, "v.patch_embd.bias", tensors, inferred, embLen)

	preNorm, err := loadF32Vec(data, gguf.DataOffset, "v.pre_ln.weight", tensors, inferred)
	if err != nil {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
	}

	layers := make([]PixtralVisionLayerWeights, 0, blockCount)
	for l := range blockCount {
		prefix := fmt.Sprintf("v.blk.%d.", l)
		q, err := load(prefix + "attn_q.weight")
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		k, err := load(prefix + "attn_k.weight")
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		v, err := load(prefix + "attn_v.weight")
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		out, err := load(prefix + "attn_out.weight")
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		ffnGate, err := load(prefix + "ffn_gate.weight")
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		ffnUp, err := load(prefix + "ffn_up.weight")
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		ffnDown, err := load(prefix + "ffn_down.weight")
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		attnNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"ln1.weight", tensors, inferred)
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		ffnNorm, err := loadF32Vec(data, gguf.DataOffset, prefix+"ln2.weight", tensors, inferred)
		if err != nil {
			return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
		}
		layers = append(layers, PixtralVisionLayerWeights{
			AttnNorm: attnNorm, Q: q, K: k, V: v, Out: out,
			QB:      loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_q.bias", tensors, inferred, embLen),
			KB:      loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_k.bias", tensors, inferred, embLen),
			VB:      loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_v.bias", tensors, inferred, embLen),
			OutB:    loadOptionalF32Vec(data, gguf.DataOffset, prefix+"attn_out.bias", tensors, inferred, embLen),
			FFNNorm: ffnNorm, FFNGate: ffnGate, FFNUp: ffnUp, FFNDown: ffnDown,
		})
		if l == 0 || l+1 == blockCount || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded vision layer %d/%d\n", l+1, blockCount)
		}
	}

	inputNorm, err := loadF32Vec(data, gguf.DataOffset, "mm.input_norm.weight", tensors, inferred)
	if err != nil {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
	}

	mergedLen := embLen * mergeSize * mergeSize
	if info, ok := tensors["mm.patch_merger.weight"]; !ok {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: tensor mm.patch_merger.weight not found")
	} else if len(info.Dims) != 2 || int(info.Dims[0]) != mergedLen || int(info.Dims[1]) != embLen {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: mm.patch_merger.weight dims %v, want [%d,%d]", info.Dims, mergedLen, embLen)
	}
	patchMerger, err := load("mm.patch_merger.weight")
	if err != nil {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
	}

	proj1, err := load("mm.1.weight")
	if err != nil {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
	}
	proj2, err := load("mm.2.weight")
	if err != nil {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
	}
	proj1B := loadOptionalF32Vec(data, gguf.DataOffset, "mm.1.bias", tensors, inferred, projDim)
	proj2B := loadOptionalF32Vec(data, gguf.DataOffset, "mm.2.bias", tensors, inferred, projDim)

	imgBreak, err := loadF32Vec(data, gguf.DataOffset, "v.token_embd.img_break", tensors, inferred)
	if err != nil {
		return config, PixtralVisionWeights{}, fmt.Errorf("loading vision projector: %w", err)
	}

	weights := PixtralVisionWeights{
		PatchEmbd: patchEmbd, PatchEmbdB: patchEmbdB, PreNorm: preNorm, Layers: layers,
		InputNorm: inputNorm, PatchMerger: patchMerger,
		Proj1: proj1, Proj1B: proj1B, Proj2: proj2, Proj2B: proj2B, ImgBreak: imgBreak,
	}
	return config, weights, nil
}

// buildPixtralRope2DInvFreq builds the row- and column-axis inverse
// frequencies for Pixtral's split 2D RoPE. Pixtral first constructs its usual
// one-dimensional sequence f_i = theta^(-2i/headDim), then assigns alternating
// frequencies to the two axes: row_j=f_(2j), col_j=f_(2j+1).
//
// These must be distinct sequences. Reusing the row frequencies for the column
// axis leaves uniform images plausible but corrupts the geometry of real scenes.
func buildPixtralRope2DInvFreq(headDim int, theta float32) (rowInv, colInv []float32) {
	quarter := headDim / 4
	rowInv = make([]float32, quarter)
	colInv = make([]float32, quarter)
	if headDim <= 0 || theta <= 0 {
		return rowInv, colInv
	}
	for j := range quarter {
		rowInv[j] = float32(1 / math.Pow(float64(theta), float64(4*j)/float64(headDim)))
		colInv[j] = float32(1 / math.Pow(float64(theta), float64(4*j+2)/float64(headDim)))
	}
	return rowInv, colInv
}

// preparePixtralRope2DScratch packs one combined sin/cos buffer for a patch
// at (row, col): the first headDim/4 entries are the row axis's angles, the
// next headDim/4 are the column axis's alternating-frequency angles. Pixtral
// rotates adjacent pairs *within* each axis half: row pairs occupy dimensions
// 0..headDim/2-1 and column pairs headDim/2..headDim-1. This is distinct from
// the text decoder's split-half/NeoX convention, so callers must pass
// interleaved=true to applyPreparedRope.
func preparePixtralRope2DScratch(row, col, headDim int, rowInv, colInv []float32, sinScratch, cosScratch *[]float32) (half, nCache int) {
	quarter := min(len(rowInv), len(colInv))
	half = headDim / 2
	if quarter <= 0 || half <= 0 {
		return 0, 0
	}
	ensureLenNoClear(sinScratch, half)
	ensureLenNoClear(cosScratch, half)
	sin, cos := *sinScratch, *cosScratch
	for j := 0; j < quarter; j++ {
		s, c := math.Sincos(float64(float32(row) * rowInv[j]))
		sin[j], cos[j] = float32(s), float32(c)
	}
	for j := 0; j < quarter; j++ {
		s, c := math.Sincos(float64(float32(col) * colInv[j]))
		sin[quarter+j], cos[quarter+j] = float32(s), float32(c)
	}
	return half, half
}

// EncodeImagePixtral runs the whole vision tower + projector (see
// LoadPixtralVisionModel's doc comment for the full graph) over one image's
// preprocessed patches and returns one embedding per MERGED patch, sized
// PixtralVisionConfig.ProjectionDim, in row-major (merged row, then merged
// column) order, plus the merged grid shape (for IMG_BREAK placement by the
// caller). It has no KV cache and no causal mask: every patch of this one
// image attends to every other patch of this same image, matching the
// published Pixtral architecture -- callers must encode one image at a
// time (see the "no block-diagonal masking" simplification in the project
// plan).
func EncodeImagePixtral(vc PixtralVisionConfig, weights PixtralVisionWeights, img *PreprocessedImage) (embeds [][]float32, mergedRows, mergedCols int, err error) {
	n := len(img.Pixels)
	if n == 0 || img.Rows*img.Cols != n {
		return nil, 0, 0, fmt.Errorf("encoding image: invalid patch grid %dx%d for %d patches", img.Rows, img.Cols, n)
	}
	merge := max(1, vc.SpatialMergeSize)
	// The runtime's dynamic Pixtral preprocessor aligns images to
	// patchSize*merge, so production grids form complete merge windows. Keep
	// floor semantics here for callers of the legacy preprocessor: the
	// reference projector's unfold naturally drops an incomplete bottom/right
	// window rather than inventing padded visual tokens.
	mergedRows = img.Rows / merge
	mergedCols = img.Cols / merge
	if mergedRows == 0 || mergedCols == 0 {
		return nil, 0, 0, fmt.Errorf("encoding image: patch grid %dx%d is smaller than spatial merge size %d", img.Rows, img.Cols, merge)
	}
	dim := vc.EmbeddingLength
	headDim := vc.HeadDim
	heads := vc.HeadCount
	if dim <= 0 || headDim <= 0 || heads <= 0 || headDim*heads != dim {
		return nil, 0, 0, fmt.Errorf("encoding image: invalid vision config dim=%d headDim=%d heads=%d", dim, headDim, heads)
	}

	x := make([][]float32, n)
	{
		flat := make([]float32, n*dim)
		for i := range x {
			x[i] = flat[i*dim : (i+1)*dim : (i+1)*dim]
		}
		matvecBatch(weights.PatchEmbd, img.Pixels, x)
		if len(weights.PatchEmbdB) >= dim {
			for i := range x {
				addInPlace(x[i], weights.PatchEmbdB[:dim])
			}
		}
	}
	for i := range x {
		rmsNormInto(x[i], weights.PreNorm, vc.Epsilon, &x[i])
	}

	rowInvFreq, colInvFreq := buildPixtralRope2DInvFreq(headDim, vc.RopeTheta)
	scale := float32(1 / math.Sqrt(float64(headDim)))

	q := make([][]float32, n)
	k := make([][]float32, n)
	v := make([][]float32, n)
	attnOut := make([][]float32, n)
	normed := make([][]float32, n)
	ffnGateOut := make([][]float32, n)
	ffnHidden := make([][]float32, n)
	for i := range x {
		q[i] = make([]float32, dim)
		k[i] = make([]float32, dim)
		v[i] = make([]float32, dim)
		attnOut[i] = make([]float32, dim)
		normed[i] = make([]float32, dim)
		ffnGateOut[i] = make([]float32, vc.FeedForwardLength)
		ffnHidden[i] = make([]float32, vc.FeedForwardLength)
	}

	var sinScratch, cosScratch []float32
	for li := range weights.Layers {
		layer := &weights.Layers[li]
		for i := range x {
			rmsNormInto(x[i], layer.AttnNorm, vc.Epsilon, &normed[i])
		}
		matvecBatch(layer.Q, normed, q)
		matvecBatch(layer.K, normed, k)
		matvecBatch(layer.V, normed, v)
		for i := range x {
			if len(layer.QB) >= dim {
				addInPlace(q[i], layer.QB[:dim])
			}
			if len(layer.KB) >= dim {
				addInPlace(k[i], layer.KB[:dim])
			}
			if len(layer.VB) >= dim {
				addInPlace(v[i], layer.VB[:dim])
			}
			row, col := i/img.Cols, i%img.Cols
			half, nCache := preparePixtralRope2DScratch(row, col, headDim, rowInvFreq, colInvFreq, &sinScratch, &cosScratch)
			// Pixtral's two 2D axes each use conventional adjacent-pair RoPE.
			// Do not use the Mistral text decoder's NeoX/split-half rotation here:
			// that would rotate a row component against a column component.
			applyPreparedRope(q[i], headDim, heads, half, nCache, sinScratch, cosScratch, true)
			applyPreparedRope(k[i], headDim, heads, half, nCache, sinScratch, cosScratch, true)
		}

		attendOne := func(i int, scores []float32) {
			for h := 0; h < heads; h++ {
				off := h * headDim
				query := q[i][off : off+headDim]
				maxScore := float32(-math.MaxFloat32)
				for j := 0; j < n; j++ {
					scores[j] = DotF32(query, k[j][off:off+headDim]) * scale
					maxScore = max(maxScore, scores[j])
				}
				denom := float32(0)
				for j := 0; j < n; j++ {
					scores[j] = float32(math.Exp(float64(scores[j] - maxScore)))
					denom += scores[j]
				}
				out := attnOut[i][off : off+headDim]
				clear(out)
				if denom > 0 {
					for j := 0; j < n; j++ {
						AxpyF32(out, scores[j]/denom, v[j][off:off+headDim])
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
		if n >= 64 && heads > 1 {
			parallelChunks(n, attend)
		} else {
			scores := make([]float32, n)
			for i := range x {
				attendOne(i, scores)
			}
		}

		matvecBatch(layer.Out, attnOut, normed)
		for i := range x {
			if len(layer.OutB) >= dim {
				addInPlace(normed[i], layer.OutB[:dim])
			}
			addInPlace(x[i], normed[i])
		}

		for i := range x {
			rmsNormInto(x[i], layer.FFNNorm, vc.Epsilon, &normed[i])
		}
		matvecBatch(layer.FFNGate, normed, ffnGateOut)
		matvecBatch(layer.FFNUp, normed, ffnHidden)
		for i := range x {
			for j := range ffnHidden[i] {
				g := ffnGateOut[i][j]
				var act float32
				if vc.UseGELU && !vc.UseSiLU {
					act = geluExact(g)
				} else {
					act = g / (1 + float32(math.Exp(float64(-g))))
				}
				ffnHidden[i][j] = act * ffnHidden[i][j]
			}
		}
		matvecBatch(layer.FFNDown, ffnHidden, normed)
		for i := range x {
			addInPlace(x[i], normed[i])
		}
	}

	for i := range x {
		rmsNormInto(x[i], weights.InputNorm, vc.Epsilon, &x[i])
	}

	mergedN := mergedRows * mergedCols
	mergedVecLen := dim * merge * merge
	mergedIn := make([][]float32, mergedN)
	for mi := range mergedIn {
		mr, mc := mi/mergedCols, mi%mergedCols
		vec := make([]float32, mergedVecLen)
		for dy := 0; dy < merge; dy++ {
			for dx := 0; dx < merge; dx++ {
				src := x[(mr*merge+dy)*img.Cols+(mc*merge+dx)]
				slot := dy*merge + dx
				for c := 0; c < dim; c++ {
					vec[c*merge*merge+slot] = src[c]
				}
			}
		}
		mergedIn[mi] = vec
	}
	merged := make([][]float32, mergedN)
	for i := range merged {
		merged[i] = make([]float32, dim)
	}
	matvecBatch(weights.PatchMerger, mergedIn, merged)

	hidden := make([][]float32, mergedN)
	for i := range hidden {
		hidden[i] = make([]float32, vc.ProjectionDim)
	}
	matvecBatch(weights.Proj1, merged, hidden)
	for i := range hidden {
		if len(weights.Proj1B) > 0 {
			addInPlace(hidden[i], weights.Proj1B)
		}
		for j := range hidden[i] {
			hidden[i][j] = geluExact(hidden[i][j])
		}
	}

	out := make([][]float32, mergedN)
	for i := range out {
		out[i] = make([]float32, vc.ProjectionDim)
	}
	matvecBatch(weights.Proj2, hidden, out)
	if len(weights.Proj2B) > 0 {
		for i := range out {
			addInPlace(out[i], weights.Proj2B)
		}
	}
	return out, mergedRows, mergedCols, nil
}
