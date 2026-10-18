package gopherllm

import (
	"context"
	"fmt"
	"os"
)

// canBatchQwen35Metal admits the FFN batching schedule below. Mixer
// projections retain their original GPU/CPU arithmetic choices.
// Compressed KV and unsupported FFNs remain on the existing serial path.
func (r *Runner) canBatchQwen35Metal(cache *KVCache) bool {
	return r != nil && r.kind == loadedQwen35 && !r.outOfCore &&
		os.Getenv("GOPHERLLM_NO_BATCH_PREFILL") == "" &&
		qwen35MetalBatchEligible(r.config, r.qwen35, cache)
}

func qwen35MetalBatchEligible(cfg Config, weights Qwen35Weights, cache *KVCache) bool {
	if cfg.Arch != "qwen35" || cfg.NLayers <= 0 || len(weights.Layers) != cfg.NLayers ||
		cache == nil || cache.F16 || cache.I8 || cache.Qwen35 == nil ||
		cache.layerCount() != qwen35AttentionLayerCount(weights) || cache.Qwen35.Layers != qwen35RecurrentLayerCount(weights) ||
		cache.PerPosKDim != cfg.NKVHeads*cfg.HeadDim || cache.PerPosVDim != cfg.NKVHeads*cfg.ValueDim ||
		weights.TokenEmbd.GPU != nil || weights.Output.GPU != nil {
		return false
	}
	for _, layer := range weights.Layers {
		w := layer.FFN
		if w.MoE != nil || w.Gate.GPU != nil || w.Up.GPU != nil || w.Down.GPU != nil ||
			w.Gate.Cols != cfg.Dim || w.Up.Cols != cfg.Dim || w.Down.Rows != cfg.Dim ||
			!metalQwen35FFNBatchReady(w.Gate.Metal, w.Up.Metal, w.Down.Metal) {
			return false
		}
		if layer.Kind == qwen35Attention {
			w := layer.Attention
			if w.Q.GPU != nil || w.K.GPU != nil || w.V.GPU != nil || w.O.GPU != nil {
				return false
			}
		} else {
			w := layer.DeltaNet
			if w.QKVConv.GPU != nil || w.Gate.GPU != nil || w.AlphaProj.GPU != nil || w.BetaProj.GPU != nil || w.Out.GPU != nil {
				return false
			}
		}
	}
	return true
}

// forwardQwen35MetalBatchStatesInto keeps every recurrent and full-attention
// step in its established chronological CPU/Metal path. FFNs stream all
// positions together on Metal; eligible DeltaNet output projections reuse
// their CPU-Q8 weights across positions. This preserves the mixed execution model:
// a CPU-Q8 projection is never silently replaced by a GPU-float projection.
// The returned normalized rows borrow buf.qwenBatch.XN. As with the CPU hybrid
// batch, callers must restore the recurrent checkpoint after cancellation or
// rejection; K/V rows beyond the accepted prefix remain causally invisible.
func forwardQwen35MetalBatchStatesInto(ctx context.Context, cfg Config, weights Qwen35Weights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, start int) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p := len(tokens)
	if p == 0 {
		return nil, nil
	}
	if p > 32 || buf == nil || !qwen35MetalBatchEligible(cfg, weights, cache) || start < 0 || start > cache.MaxLen || p > cache.MaxLen-start {
		return nil, fmt.Errorf("gopherllm: Qwen Metal block requires supported dense FFNs, F32 KV and a valid range of at most 32 tokens")
	}
	b := &buf.qwenBatch
	X := reuseBatchViews(&b.XFlat, &b.X, p, cfg.Dim)
	XN := reuseBatchViews(&b.XNFlat, &b.XN, p, cfg.Dim)
	Proj := reuseBatchViews(&b.ProjFlat, &b.Proj, p, cfg.Dim)
	for i, token := range tokens {
		weights.TokenEmbd.RowInto(int(token), cfg.Dim, &X[i])
	}
	replay := &buf.speculative.qwenReplay
	replay.begin(cfg, cache.Qwen35.Layers, p, buf.speculative.active)
	for _, layer := range weights.Layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		batchDeltaOut := p > 1 && layer.Kind == qwen35DeltaNet && qwen35MetalBatchCPUOut(layer.DeltaNet.Out)
		var Mixed [][]float32
		if batchDeltaOut {
			Mixed = reuseBatchViews(&b.MixedFlat, &b.Mixed, p, cfg.SSMInner)
		}
		var QKV, Gate, Alpha, Beta [][]float32
		if layer.Kind == qwen35DeltaNet && replay.valid {
			channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
			QKV = reuseBatchViews(&b.QKVFlat, &b.QKV, p, channels)
			Gate = reuseBatchViews(&b.GateFlat, &b.Gate, p, cfg.SSMInner)
			Alpha = reuseBatchViews(&b.AlphaFlat, &b.Alpha, p, cfg.SSMHeads)
			Beta = reuseBatchViews(&b.BetaFlat, &b.Beta, p, cfg.SSMHeads)
		}
		for i := range tokens {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			rmsNormInto(X[i], layer.Norm, cfg.RMSNormEps, &XN[i])
			if layer.Kind == qwen35Attention {
				half, pairs := prepareRopeScratch(start+i, cfg.HeadDim, cfg.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &buf.RopeSin, &buf.RopeCos)
				qwen35AttentionForward(cfg, layer.Attention, cache, XN[i], layer.KVCacheSlot, start+i, half, pairs, buf)
			} else {
				if batchDeltaOut {
					qwen35DeltaNetMix(cfg, layer.DeltaNet, cache.Qwen35, XN[i], layer.RecurrentCacheSlot, buf)
					copy(Mixed[i], buf.MambaY)
				} else {
					qwen35DeltaNetForward(cfg, layer.DeltaNet, cache.Qwen35, XN[i], layer.RecurrentCacheSlot, buf)
				}
				if replay.valid {
					copy(QKV[i], buf.MambaIn)
					copy(Gate[i], buf.MambaZ)
					copy(Alpha[i], buf.MambaDT)
					copy(Beta[i], buf.MambaBeta)
				}
			}
			if !batchDeltaOut {
				copy(Proj[i], buf.Proj)
				addInPlace(X[i], Proj[i])
				rmsNormInto(X[i], layer.PostAttnNorm, cfg.RMSNormEps, &XN[i])
			}
		}
		if batchDeltaOut {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			matvecBatchSlabs(layer.DeltaNet.Out, Mixed, Proj, b.MixedFlat, b.ProjFlat)
			for i := range tokens {
				addInPlace(X[i], Proj[i])
				rmsNormInto(X[i], layer.PostAttnNorm, cfg.RMSNormEps, &XN[i])
			}
		}
		if layer.Kind == qwen35DeltaNet && replay.valid {
			replay.capture(layer.RecurrentCacheSlot, b.QKVFlat, b.GateFlat, b.AlphaFlat, b.BetaFlat)
		}
		// Preserve the serial GPU reduction before the next CPU-Q8 mixer:
		// even a 1e-8 FFN difference can cross an activation-rounding tie.
		if !matvecMetalSwiGLUDecodeBatchInto(layer.FFN.Gate.Metal, layer.FFN.Up.Metal, layer.FFN.Down.Metal, b.XNFlat, p, &b.ProjFlat) {
			// A failed GPU submission writes no projection output. Retain the
			// original row schedule, including its existing fallback behavior.
			for i := range tokens {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				qwen35FFNForward(layer.FFN, XN[i], buf)
				copy(Proj[i], buf.Proj)
			}
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for i := range tokens {
			addInPlace(X[i], Proj[i])
		}
	}
	copy(buf.X, X[p-1])
	for i := range tokens {
		rmsNormInto(X[i], weights.OutputNorm, cfg.RMSNormEps, &XN[i])
	}
	copy(buf.XN, XN[p-1])
	return XN, ctx.Err()
}

func qwen35MetalBatchCPUOut(w Weight) bool {
	if !useQ8Activations.Load() || w.F32 != nil || len(w.Raw) == 0 || w.Rows <= 0 || w.Cols <= 0 || w.Cols%256 != 0 ||
		w.GPU != nil || metalWeightUsesDirect(w.Metal) {
		return false
	}
	switch w.Type {
	case GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0:
		return true
	default:
		return false
	}
}
