package engine

import (
	"context"
	"fmt"
	"os"
)

// qwen35MTPProcessBatch catches up the draft head from known target hidden
// states. Only EH, K and V projections are needed: each next input uses the
// authoritative target hidden state, so catch-up attention, Q/gate, O and FFN
// outputs are never consumed. Draft generation still uses the full MTP body.
//
// The workspace belongs to state.Scratch, separately from the target graph's
// borrowed hidden rows. Scratch X/XN are temporary EH activations, not a full
// draft-body result. PendingHidden advances after each completed chunk;
// on error the caller must discard a partial prefill or restore its speculative
// boundary, as it does for target-state cancellation.
func qwen35MTPProcessBatch(ctx context.Context, cfg Config, weights *Qwen35MTPWeights, state *Qwen35MTPState, tokens []uint32, targetHidden [][]float32, startPos int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if weights == nil || state == nil {
		return nil
	}
	if len(tokens) != len(targetHidden) || cfg.Dim <= 0 || state.KV == nil || state.Scratch == nil || len(state.PendingHidden) < cfg.Dim ||
		startPos < 0 || startPos > state.KV.MaxLen || len(tokens) > state.KV.MaxLen-startPos {
		return fmt.Errorf("gopherllm: MTP catch-up has invalid target rows or cache range")
	}
	for i, token := range tokens {
		if len(targetHidden[i]) < cfg.Dim || uint64(token) >= uint64(cfg.VocabSize) {
			return fmt.Errorf("gopherllm: MTP catch-up has an invalid token or target hidden row at %d", i)
		}
	}
	if len(tokens) == 0 {
		return nil
	}
	// A one-token tail cannot amortize batch setup. Float-only quantized
	// kernels also retain their established operation tree: small differences
	// before compressed-KV rounding otherwise propagate into later drafting.
	if len(tokens) == 1 || os.Getenv("GOPHERLLM_NO_MTP_BATCH") != "" || !qwen35MTPBatchNumerics(*weights) {
		for i, token := range tokens {
			if err := ctx.Err(); err != nil {
				return err
			}
			qwen35MTPProcess(cfg, weights, state, token, targetHidden[i], startPos+i)
		}
		return ctx.Err()
	}
	execution := qwen35MTPExecutionWeights(*weights)
	for first := 0; first < len(tokens); first += 32 {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(first+32, len(tokens))
		if end-first == 1 {
			qwen35MTPProcess(cfg, weights, state, tokens[first], targetHidden[first], startPos+first)
			continue
		}
		if err := qwen35MTPProcessChunk(ctx, cfg, execution, state, tokens[first:end], targetHidden[first:end], startPos+first); err != nil {
			return err
		}
	}
	return ctx.Err()
}

// qwen35MTPProcessKVInto is the one-token catch-up path. No later catch-up or
// draft operation reads this token's MTP output hidden state: the accepted
// target hidden is the next boundary. Keep the public full-body operation as
// the independent reference and as the dependency-chained draft operation.
// The caller retains that full-body path for WebGPU projection views.
func qwen35MTPProcessKVInto(cfg Config, weights Qwen35MTPWeights, state *Qwen35MTPState, token uint32, targetHidden []float32, pos int) {
	if state.KV == nil || state.Scratch == nil {
		panic("Qwen MTP catch-up requires an MTP cache")
	}
	if len(state.PendingHidden) < cfg.Dim {
		panic("Qwen MTP catch-up received a short pending target hidden state")
	}
	weights = qwen35MTPExecutionWeights(weights)
	buf := state.Scratch
	weights.TokenEmbd.RowInto(int(token), cfg.Dim, &buf.XN2)
	rmsNormInto(buf.XN2, weights.EmbeddingNorm, cfg.RMSNormEps, &buf.X)
	rmsNormInto(state.PendingHidden[:cfg.Dim], weights.HiddenNorm, cfg.RMSNormEps, &buf.XN)
	ensureLenNoClear(&buf.MTPInput, 2*cfg.Dim)
	copy(buf.MTPInput[:cfg.Dim], buf.X[:cfg.Dim])
	copy(buf.MTPInput[cfg.Dim:], buf.XN[:cfg.Dim])
	weights.EHProj.MatvecInto(buf.MTPInput, &buf.X)
	rmsNormInto(buf.X, weights.Norm, cfg.RMSNormEps, &buf.XN)
	if !tryMatvec2Into(weights.Attention.K, weights.Attention.V, buf.XN, &buf.Q4KXSums, &buf.K, &buf.V) {
		weights.Attention.K.MatvecInto(buf.XN, &buf.K)
		weights.Attention.V.MatvecInto(buf.XN, &buf.V)
	}
	qwen35MTPStoreKV(cfg, weights.Attention, state, pos)
	copy(state.PendingHidden, targetHidden[:cfg.Dim])
}

func qwen35MTPStoreKV(cfg Config, weights Qwen35AttentionWeights, state *Qwen35MTPState, pos int) {
	buf := state.Scratch
	ropeHalf, ropePairs := prepareRopeScratch(pos, cfg.HeadDim, cfg.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &buf.RopeSin, &buf.RopeCos)
	perHeadRMSNormInPlace(buf.K, cfg.HeadDim, cfg.NKVHeads, weights.KNorm, cfg.RMSNormEps)
	applyPreparedRope(buf.K, cfg.HeadDim, cfg.NKVHeads, ropeHalf, ropePairs, buf.RopeSin, buf.RopeCos, false)
	state.KV.storeKV(0, pos, buf.K, buf.V)
}

func qwen35MTPBatchNumerics(w Qwen35MTPWeights) bool {
	if !qwen35MTPKVOnlyEligible(w) {
		return false
	}
	if useQ8Activations.Load() {
		return true
	}
	for _, matrix := range [...]Weight{w.EHProj, w.Attention.K, w.Attention.V} {
		if matrix.F32 == nil && len(matrix.Raw) > 0 && !rawScalarWeight(matrix) {
			return false
		}
	}
	return true
}

// The full attention graph can fuse Q/K/V on the CPU even when individual
// tensors have WebGPU views. Removing Q changes the available fusion groups:
// mixed K/V would instead reach their individual GPU matvecs. Preserve the
// original full-body arithmetic for those views, including GPU EH input mixing.
func qwen35MTPKVOnlyEligible(w Qwen35MTPWeights) bool {
	return w.EHProj.GPU == nil && w.Attention.Q.GPU == nil && w.Attention.K.GPU == nil && w.Attention.V.GPU == nil
}

func qwen35MTPProcessChunk(ctx context.Context, cfg Config, weights Qwen35MTPWeights, state *Qwen35MTPState, tokens []uint32, targetHidden [][]float32, startPos int) error {
	p, dim := len(tokens), cfg.Dim
	buf := state.Scratch
	b := &buf.qwenBatch
	// Stage authoritative target states before reusing any activation rows.
	// Gate/QKV are otherwise unused by the MTP K/V catch-up projections.
	Target := reuseBatchViews(&b.GateFlat, &b.Gate, p, dim)
	for i := range tokens {
		copy(Target[i], targetHidden[i][:dim])
	}
	X := reuseBatchViews(&b.XFlat, &b.X, p, dim)
	XN := reuseBatchViews(&b.XNFlat, &b.XN, p, dim)
	Pair := reuseBatchViews(&b.QKVFlat, &b.QKV, p, 2*dim)
	for i, token := range tokens {
		weights.TokenEmbd.RowInto(int(token), dim, &X[i])
		embedding, previous := Pair[i][:dim], Pair[i][dim:]
		rmsNormInto(X[i], weights.EmbeddingNorm, cfg.RMSNormEps, &embedding)
		prev := state.PendingHidden[:dim]
		if i > 0 {
			prev = Target[i-1]
		}
		rmsNormInto(prev, weights.HiddenNorm, cfg.RMSNormEps, &previous)
	}
	matvecBatchSlabs(weights.EHProj, Pair, X, b.QKVFlat, b.XFlat)
	for i := range tokens {
		rmsNormInto(X[i], weights.Norm, cfg.RMSNormEps, &XN[i])
	}
	kLen, vLen := cfg.NKVHeads*cfg.HeadDim, cfg.NKVHeads*cfg.ValueDim
	K := reuseBatchViews(&b.KFlat, &b.K, p, kLen)
	V := reuseBatchViews(&b.VFlat, &b.V, p, vLen)
	w := weights.Attention
	matvecBatch2Slabs(w.K, w.V, XN, K, V, b.XNFlat, b.KFlat, b.VFlat)
	ensureLenNoClear(&buf.K, kLen)
	ensureLenNoClear(&buf.V, vLen)
	for i := range tokens {
		if err := ctx.Err(); err != nil {
			return err
		}
		copy(buf.K, K[i])
		copy(buf.V, V[i])
		qwen35MTPStoreKV(cfg, w, state, startPos+i)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	copy(buf.X, X[p-1])
	copy(buf.XN, XN[p-1])
	copy(state.PendingHidden, Target[p-1])
	return ctx.Err()
}
