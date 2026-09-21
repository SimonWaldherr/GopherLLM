package gopherllm

import (
	"fmt"
	"math"
)

// This file implements the FastConformer encoder's per-block forward pass
// -- see parakeet.go's doc comment, point 3. Full (non-causal) attention
// over the whole clip: this is an offline-transcription implementation
// (asr.encoder.att_context_style=="regular", asr.encoder.cache_supported
// ==false in the checkpoint inspected -- no streaming/chunked masking is
// implemented here).
//
// Every linear projection below batches all T timesteps into one
// blasMatvecBatch call instead of looping Weight.MatvecInto per row: on
// Darwin+cgo this reaches Accelerate's GEMM kernel (the same win
// prepareVoxtralStreamWeights/blasMatvecBatch already give Voxtral's
// encoder), and even on the portable fallback matvecBatch amortizes
// dispatch overhead and parallelizes across the weight's rows. See
// parakeet_bench_test.go for the before/after numbers.
//
// The relative-position attention (ESPnet/NeMo's RelPositionMultiHeadAttention,
// itself Transformer-XL's formulation) is implemented via a direct index
// derivation rather than PyTorch's pad/reshape/drop "rel_shift" trick:
// given query position q and key position k (0-indexed, T total positions),
// the positional-score row NeMo's rel_shift would select is
// rawPosScore[q][(T-1)-q+k] -- derived and boundary-checked (q=0,k=0 ->
// relative offset 0 -> the table's middle row; q=0,k=T-1 -> offset -(T-1);
// q=T-1,k=0 -> offset +(T-1), matching RelPositionalEncoding's own
// descending-from-(T-1)-to-(T-1) position ordering) rather than mechanically
// replicated, since an off-by-one here would misalign every attention score
// silently rather than error. The per-head dot products and weighted sums
// use DotF32/AxpyF32 (this codebase's SIMD kernels) instead of scalar Go
// loops over headDim.

func swish(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// weightRows returns a Weight's output row count for a matvec over an input
// of length cols. Rows/Cols are only populated on raw (quantized or lazy
// scalar) weights; an owned F32 weight -- which is what every Parakeet
// tensor loads as -- infers its row count from len(F32)/cols instead (see
// Weight's doc comment), so batched callers that need to pre-size an output
// buffer before calling blasMatvecBatch/matvecBatch must compute it the same
// way MatvecInto does internally rather than reading w.Rows directly.
func weightRows(w Weight, cols int) int {
	if cols == 0 {
		return 0
	}
	if w.F32 != nil {
		return len(w.F32) / cols
	}
	return w.Rows
}

// parakeetFeedForwardBatch runs one macaron half-step FFN module over all T
// timesteps at once: LayerNorm -> Linear -> Swish -> Linear, residual added
// with a 0.5 scale (the "half-step" in "Macaron-Net" -- Conformer's own term
// for this pattern).
func parakeetFeedForwardBatch(xs [][]float32, normW, normB []float32, lin1, lin2 Weight, eps float32) [][]float32 {
	t := len(xs)
	result := make([][]float32, t)
	if t == 0 {
		return result
	}
	normed := make([][]float32, t)
	for i := range xs {
		layerNormInto(xs[i], normW, normB, eps, &normed[i])
	}
	dModel := len(xs[0])
	hiddenDim := weightRows(lin1, dModel)
	hidden := make([][]float32, t)
	for i := range hidden {
		hidden[i] = make([]float32, hiddenDim)
	}
	blasMatvecBatch(lin1, normed, hidden)
	for i := range hidden {
		row := hidden[i]
		for j, v := range row {
			row[j] = swish(v)
		}
	}
	outDim := weightRows(lin2, hiddenDim)
	out := make([][]float32, t)
	for i := range out {
		out[i] = make([]float32, outDim)
	}
	blasMatvecBatch(lin2, hidden, out)
	for i := range result {
		row := make([]float32, len(xs[i]))
		oi := out[i]
		for j := range row {
			row[j] = xs[i][j] + 0.5*oi[j]
		}
		result[i] = row
	}
	return result
}

// parakeetConvModule runs the Conformer convolution module over the full
// sequence [T][DModel]: LayerNorm -> pointwise_conv1 (DModel->2*DModel) ->
// GLU -> depthwise_conv (causal-free, same-padding, groups=DModel) ->
// batch_norm (eval-mode: running stats, no batch statistics) -> Swish ->
// pointwise_conv2 (DModel->DModel), residual added. The two pointwise convs
// are 1x1, i.e. per-timestep matvecs, so they batch across T the same way
// the FFN's linear layers do.
func parakeetConvModule(xs [][]float32, conv ParakeetEncoderConvLayer, normW, normB []float32, dModel, kernel int, eps float32) [][]float32 {
	t := len(xs)
	out := make([][]float32, t)
	if t == 0 {
		return out
	}
	normed := make([][]float32, t)
	for i := range xs {
		layerNormInto(xs[i], normW, normB, eps, &normed[i])
	}

	expandedDim := weightRows(conv.PointwiseConv1, dModel)
	expanded := make([][]float32, t)
	for i := range expanded {
		expanded[i] = make([]float32, expandedDim)
	}
	blasMatvecBatch(conv.PointwiseConv1, normed, expanded)

	half := expandedDim / 2
	glu := make([][]float32, t)
	for i := range glu {
		row := make([]float32, half)
		ex := expanded[i]
		for c := 0; c < half; c++ {
			gate := 1 / (1 + float32(math.Exp(float64(-ex[half+c]))))
			row[c] = ex[c] * gate
		}
		glu[i] = row
	}

	// Depthwise conv: same-padding (pad = (kernel-1)/2 on each side, kernel
	// is odd -- 9 for this checkpoint), one independent 1D kernel per
	// channel (groups==DModel), no bias (none shipped -- batch_norm
	// supplies the additive term instead).
	pad := (kernel - 1) / 2
	dwOut := make([][]float32, t)
	for i := range dwOut {
		dwOut[i] = make([]float32, dModel)
	}
	for c := 0; c < dModel; c++ {
		k := conv.DepthwiseConv[c*kernel : (c+1)*kernel]
		for ti := 0; ti < t; ti++ {
			var sum float32
			for ki := 0; ki < kernel; ki++ {
				si := ti - pad + ki
				if si < 0 || si >= t {
					continue
				}
				sum += glu[si][c] * k[ki]
			}
			dwOut[ti][c] = sum
		}
	}

	// Batch norm in eval mode: normalize with the checkpoint's running
	// mean/var (never batch statistics -- there is no "batch" at
	// inference), then affine scale/shift, then Swish.
	for i := range dwOut {
		row := dwOut[i]
		for c := range row {
			normed := (row[c] - conv.BatchNormMean[c]) / float32(math.Sqrt(float64(conv.BatchNormVar[c])+1e-5))
			row[c] = swish(normed*conv.BatchNormWeight[c] + conv.BatchNormBias[c])
		}
	}

	projDim := weightRows(conv.PointwiseConv2, dModel)
	proj := make([][]float32, t)
	for i := range proj {
		proj[i] = make([]float32, projDim)
	}
	blasMatvecBatch(conv.PointwiseConv2, dwOut, proj)

	for i := range out {
		row := make([]float32, dModel)
		pi := proj[i]
		for c := range row {
			row[c] = xs[i][c] + pi[c]
		}
		out[i] = row
	}
	return out
}

// parakeetRelPosSelfAttention runs one Conformer block's relative-position
// multi-head self-attention over the full sequence, full (non-causal)
// attention. posEnc is the encoder's whole precomputed table
// (ParakeetEncoderWeights.PosEnc, flat [PosEncMaxLen][DModel], row r =
// relative position (PosEncMaxLen/2)-r since the table's odd length
// 2*L-1 centers on L-1); this function extracts and projects only the
// 2*T-1 rows a sequence of length T actually needs.
func parakeetRelPosSelfAttention(xs [][]float32, attn ParakeetEncoderAttention, normW, normB []float32, nHeads, headDim int, posEnc []float32, posEncMaxLen int, eps float32) ([][]float32, error) {
	t := len(xs)
	center := posEncMaxLen / 2 // row index of relative position 0
	if t-1 > center {
		return nil, fmt.Errorf("parakeet attention: sequence length %d exceeds the positional encoding table's range (max relative position %d)", t, center)
	}
	dModel := nHeads * headDim
	result := make([][]float32, t)
	if t == 0 {
		return result, nil
	}

	normed := make([][]float32, t)
	for i := range xs {
		layerNormInto(xs[i], normW, normB, eps, &normed[i])
	}

	q := make([][]float32, t)
	k := make([][]float32, t)
	v := make([][]float32, t)
	for i := range q {
		q[i] = make([]float32, dModel)
		k[i] = make([]float32, dModel)
		v[i] = make([]float32, dModel)
	}
	blasMatvecBatch(attn.LinearQ, normed, q)
	blasMatvecBatch(attn.LinearK, normed, k)
	blasMatvecBatch(attn.LinearV, normed, v)

	// Project the 2T-1 positional rows [center-(T-1) : center+(T-1)]
	// (inclusive), i.e. relative positions from +(T-1) down to -(T-1),
	// through linear_pos (no bias).
	posStart := center - (t - 1)
	posLen := 2*t - 1
	posRows := make([][]float32, posLen)
	for i := 0; i < posLen; i++ {
		posRows[i] = posEnc[(posStart+i)*dModel : (posStart+i+1)*dModel]
	}
	posProj := make([][]float32, posLen)
	for i := range posProj {
		posProj[i] = make([]float32, dModel)
	}
	blasMatvecBatch(attn.LinearPos, posRows, posProj)

	scale := float32(1 / math.Sqrt(float64(headDim)))
	out := make([][]float32, t)
	for i := range out {
		out[i] = make([]float32, dModel)
	}
	scores := make([]float32, t)
	qu := make([]float32, headDim)
	qv := make([]float32, headDim)
	for h := 0; h < nHeads; h++ {
		off := h * headDim
		uBias := attn.PosBiasU[off : off+headDim]
		vBias := attn.PosBiasV[off : off+headDim]
		for qi := 0; qi < t; qi++ {
			qh := q[qi][off : off+headDim]
			for d := 0; d < headDim; d++ {
				qu[d] = qh[d] + uBias[d]
				qv[d] = qh[d] + vBias[d]
			}
			maxScore := negMaxF32
			for ki := 0; ki < t; ki++ {
				kh := k[ki][off : off+headDim]
				ac := DotF32(qu, kh)
				// See this file's doc comment: rawPosScore index for
				// (qi,ki) is (t-1)-qi+ki within the posProj rows this
				// function already extracted (index 0 == relative
				// position +(T-1)).
				bd := DotF32(qv, posProj[(t-1)-qi+ki])
				s := (ac + bd) * scale
				scores[ki] = s
				if s > maxScore {
					maxScore = s
				}
			}
			var sum float32
			for ki := 0; ki < t; ki++ {
				e := fastExpF32(scores[ki] - maxScore)
				scores[ki] = e
				sum += e
			}
			inv := float32(1)
			if sum > 0 {
				inv = 1 / sum
			}
			ctx := out[qi][off : off+headDim]
			for ki := 0; ki < t; ki++ {
				wgt := scores[ki] * inv
				if wgt == 0 {
					continue
				}
				AxpyF32(ctx, wgt, v[ki][off:off+headDim])
			}
		}
	}

	proj := make([][]float32, t)
	for i := range proj {
		proj[i] = make([]float32, dModel)
	}
	blasMatvecBatch(attn.LinearOut, out, proj)
	for i := range result {
		row := make([]float32, dModel)
		pi := proj[i]
		for c := range row {
			row[c] = xs[i][c] + pi[c]
		}
		result[i] = row
	}
	return result, nil
}

// ParakeetEncode runs the full FastConformer encoder: the mel frontend,
// dw_striding subsampling, then every Conformer block, returning one
// DModel-wide embedding per output frame (row-major [T][DModel]).
func ParakeetEncode(cfg ParakeetConfig, w ParakeetWeights, samples []float32) ([][]float32, error) {
	mel, nFrames, err := parakeetLogMelSpectrogram(cfg.Preprocessor, w.MelFilterbank, samples)
	if err != nil {
		return nil, fmt.Errorf("parakeet encode: %w", err)
	}
	flat, outT, err := parakeetSubsample(cfg, w.Encoder, mel, nFrames)
	if err != nil {
		return nil, fmt.Errorf("parakeet encode: %w", err)
	}
	dModel := cfg.Encoder.DModel
	xs := make([][]float32, outT)
	for i := range xs {
		xs[i] = append([]float32(nil), flat[i*dModel:(i+1)*dModel]...)
	}

	for li := range w.Encoder.Layers {
		layer := &w.Encoder.Layers[li]
		xs2 := parakeetFeedForwardBatch(xs, layer.NormFeedForward1Weight, layer.NormFeedForward1Bias, layer.FeedForward1Linear1, layer.FeedForward1Linear2, cfg.Encoder.Epsilon)
		xs3, err := parakeetRelPosSelfAttention(xs2, layer.SelfAttn, layer.NormSelfAttWeight, layer.NormSelfAttBias, cfg.Encoder.NHeads, cfg.Encoder.HeadDim, w.Encoder.PosEnc, w.Encoder.PosEncMaxLen, cfg.Encoder.Epsilon)
		if err != nil {
			return nil, fmt.Errorf("parakeet encode: layer %d: %w", li, err)
		}
		xs4 := parakeetConvModule(xs3, layer.Conv, layer.NormConvWeight, layer.NormConvBias, dModel, cfg.Encoder.ConvKernelSize, cfg.Encoder.Epsilon)
		xs5 := parakeetFeedForwardBatch(xs4, layer.NormFeedForward2Weight, layer.NormFeedForward2Bias, layer.FeedForward2Linear1, layer.FeedForward2Linear2, cfg.Encoder.Epsilon)
		for i := range xs5 {
			var normed []float32
			layerNormInto(xs5[i], layer.NormOutWeight, layer.NormOutBias, cfg.Encoder.Epsilon, &normed)
			xs5[i] = normed
		}
		xs = xs5
	}
	return xs, nil
}
