package gopherllm

import (
	"context"
	"fmt"
)

// qwen35BatchScratch bounds activation retention to at most 32 prompt tokens.
// Only the projections are batched: convolution and DeltaNet stay sequential
// per layer, and full attention reads each token's causal prefix as in decode.
type qwen35BatchScratch struct {
	XFlat, XNFlat, ProjFlat, MixedFlat     []float32
	QKVFlat, GateFlat, AlphaFlat, BetaFlat []float32
	QGateFlat, KFlat, VFlat                []float32
	FFNGateFlat, UpFlat, HiddenFlat        []float32
	RopeSinFlat, RopeCosFlat               []float32
	X, XN, Proj, Mixed, QKV, Gate          [][]float32
	Alpha, Beta, QGate, K, V               [][]float32
	FFNGate, Up, Hidden, RopeSin, RopeCos  [][]float32
}

// prefillQwen35At resumes an already initialized hybrid cache at startPos.
// As with token decode, the caller resets recurrent state for a cold request
// (generationWorkspace does so) or restores the complete matching snapshot.
func (r *Runner) prefillQwen35At(ctx context.Context, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, logits *[]float32) error {
	if !r.canBatchQwen35() {
		return fmt.Errorf("gopherllm: Qwen chunked prefill requires a dense CPU hybrid model")
	}
	if cache == nil || cache.Qwen35 == nil || startPos < 0 || startPos > cache.MaxLen || len(tokens) > cache.MaxLen-startPos {
		return fmt.Errorf("gopherllm: Qwen chunked prefill has an invalid cache range")
	}
	chunk := min(r.prefillChunkSize(), 32)
	for start := 0; start < len(tokens); start += chunk {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+chunk, len(tokens))
		if err := forwardQwen35BatchInto(ctx, r.config, r.qwen35, cache, buf, tokens[start:end], startPos+start, end == len(tokens), logits); err != nil {
			return err
		}
	}
	return ctx.Err()
}

func forwardQwen35BatchInto(ctx context.Context, cfg Config, weights Qwen35Weights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, last bool, logits *[]float32) error {
	_, err := forwardQwen35Batch(ctx, cfg, weights, cache, buf, tokens, startPos, false)
	if err != nil {
		return err
	}
	if len(tokens) > 0 && last && logits != nil {
		weights.Output.MatvecInto(buf.XN, logits)
	}
	return ctx.Err()
}

// forwardQwen35BatchStatesInto exposes the normalized target hidden state for
// every position in a short block. The rows borrow buf.qwenBatch.XN until its
// next use. In particular, callers must consume them before another batch.
// DeltaNet updates remain chronological within each layer; a speculative
// caller must restore the pre-block recurrent checkpoint before replaying an
// accepted prefix after a rejection or cancellation.
func forwardQwen35BatchStatesInto(ctx context.Context, cfg Config, weights Qwen35Weights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int) ([][]float32, error) {
	return forwardQwen35Batch(ctx, cfg, weights, cache, buf, tokens, startPos, true)
}

func forwardQwen35Batch(ctx context.Context, cfg Config, weights Qwen35Weights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, allStates bool) ([][]float32, error) {
	p := len(tokens)
	if p == 0 {
		return nil, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if p > 32 || cache == nil || cache.Qwen35 == nil || buf == nil || startPos < 0 || startPos > cache.MaxLen || p > cache.MaxLen-startPos {
		return nil, fmt.Errorf("gopherllm: Qwen block has an invalid cache range or exceeds 32 tokens")
	}
	b := &buf.qwenBatch
	X := reuseBatchViews(&b.XFlat, &b.X, p, cfg.Dim)
	XN := reuseBatchViews(&b.XNFlat, &b.XN, p, cfg.Dim)
	Proj := reuseBatchViews(&b.ProjFlat, &b.Proj, p, cfg.Dim)
	for token, id := range tokens {
		weights.TokenEmbd.RowInto(int(id), cfg.Dim, &X[token])
	}
	// Keep the same RoPE arithmetic as decode, but prepare each position once
	// per chunk instead of once for every full-attention layer.
	ropeDim := cfg.RopeDimensionCount
	if ropeDim <= 0 || ropeDim > cfg.HeadDim {
		ropeDim = cfg.HeadDim
	}
	ropeDim -= ropeDim % 2
	ropeHalf, ropePairs := ropeDim/2, min(ropeDim/2, len(buf.RopeInvFreq))
	if ropePairs > 0 {
		reuseBatchViews(&b.RopeSinFlat, &b.RopeSin, p, ropePairs)
		reuseBatchViews(&b.RopeCosFlat, &b.RopeCos, p, ropePairs)
		for token := range tokens {
			prepareRopeScratch(startPos+token, cfg.HeadDim, cfg.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &b.RopeSin[token], &b.RopeCos[token])
		}
	}
	for _, layer := range weights.Layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for token := range tokens {
			rmsNormInto(X[token], layer.Norm, cfg.RMSNormEps, &XN[token])
		}
		if layer.Kind == qwen35Attention {
			qLen, kLen, vLen := 2*cfg.NHeads*cfg.HeadDim, cfg.NKVHeads*cfg.HeadDim, cfg.NKVHeads*cfg.ValueDim
			QGate := reuseBatchViews(&b.QGateFlat, &b.QGate, p, qLen)
			K := reuseBatchViews(&b.KFlat, &b.K, p, kLen)
			V := reuseBatchViews(&b.VFlat, &b.V, p, vLen)
			Mixed := reuseBatchViews(&b.MixedFlat, &b.Mixed, p, cfg.NHeads*cfg.ValueDim)
			w := layer.Attention
			matvecBatch3Slabs(w.Q, w.K, w.V, XN, QGate, K, V, b.XNFlat, b.QGateFlat, b.KFlat, b.VFlat)
			ensureLenNoClear(&buf.QGate, qLen)
			ensureLenNoClear(&buf.K, kLen)
			ensureLenNoClear(&buf.V, vLen)
			for token := range tokens {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				copy(buf.QGate, QGate[token])
				copy(buf.K, K[token])
				copy(buf.V, V[token])
				if ropePairs > 0 {
					copy(buf.RopeSin, b.RopeSin[token])
					copy(buf.RopeCos, b.RopeCos[token])
				}
				qwen35AttentionMix(cfg, w, cache, layer.KVCacheSlot, startPos+token, ropeHalf, ropePairs, buf)
				copy(Mixed[token], buf.AttnOut)
			}
			matvecBatchSlabs(w.O, Mixed, Proj, b.MixedFlat, b.ProjFlat)
		} else {
			channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
			QKV := reuseBatchViews(&b.QKVFlat, &b.QKV, p, channels)
			Gate := reuseBatchViews(&b.GateFlat, &b.Gate, p, cfg.SSMInner)
			Alpha := reuseBatchViews(&b.AlphaFlat, &b.Alpha, p, cfg.SSMHeads)
			Beta := reuseBatchViews(&b.BetaFlat, &b.Beta, p, cfg.SSMHeads)
			Mixed := reuseBatchViews(&b.MixedFlat, &b.Mixed, p, cfg.SSMInner)
			w := layer.DeltaNet
			matvecBatch2Slabs(w.QKVConv, w.Gate, XN, QKV, Gate, b.XNFlat, b.QKVFlat, b.GateFlat)
			matvecBatch2Slabs(w.AlphaProj, w.BetaProj, XN, Alpha, Beta, b.XNFlat, b.AlphaFlat, b.BetaFlat)
			for token := range tokens {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				qwen35DeltaNetStep(cfg, w, cache.Qwen35, QKV[token], Gate[token], Alpha[token], Beta[token], layer.RecurrentCacheSlot, buf)
				copy(Mixed[token], buf.MambaY)
			}
			matvecBatchSlabs(w.Out, Mixed, Proj, b.MixedFlat, b.ProjFlat)
		}
		for token := range tokens {
			addInPlace(X[token], Proj[token])
			rmsNormInto(X[token], layer.PostAttnNorm, cfg.RMSNormEps, &XN[token])
		}
		// Dense SwiGLU projections share the established Q8/float batch paths;
		// activation arithmetic stays identical to qwen35FFNForward.
		hiddenDim := layer.FFN.Gate.Rows
		if layer.FFN.Gate.F32 != nil {
			hiddenDim = len(layer.FFN.Gate.F32) / cfg.Dim
		}
		Gate := reuseBatchViews(&b.FFNGateFlat, &b.FFNGate, p, hiddenDim)
		Up := reuseBatchViews(&b.UpFlat, &b.Up, p, hiddenDim)
		Hidden := reuseBatchViews(&b.HiddenFlat, &b.Hidden, p, hiddenDim)
		matvecBatch2Slabs(layer.FFN.Gate, layer.FFN.Up, XN, Gate, Up, b.XNFlat, b.FFNGateFlat, b.UpFlat)
		for token := range tokens {
			siluMulF32(Gate[token], Up[token], Hidden[token])
		}
		matvecBatchSlabs(layer.FFN.Down, Hidden, Proj, b.HiddenFlat, b.ProjFlat)
		for token := range tokens {
			addInPlace(X[token], Proj[token])
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	copy(buf.X, X[p-1])
	if allStates {
		for token := range tokens {
			rmsNormInto(X[token], weights.OutputNorm, cfg.RMSNormEps, &XN[token])
		}
		copy(buf.XN, XN[p-1])
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return XN, nil
	}
	rmsNormInto(X[p-1], weights.OutputNorm, cfg.RMSNormEps, &buf.XN)
	return nil, ctx.Err()
}
