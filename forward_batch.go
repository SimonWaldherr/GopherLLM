package gopherllm

import "math"

// batchDecodeBuffer owns the large activation slabs used by batched prefill.
// It hangs off DecodeBuffer so prompt chunks and subsequent requests reuse the
// same backing arrays instead of feeding tens or hundreds of MiB to the GC.
type batchDecodeBuffer struct {
	XFlat, XNFlat, QFlat, KFlat, VFlat []float32
	AttnOutFlat, ProjFlat              []float32
	GateFlat, UpFlat, HiddenFlat       []float32
	QKVFlat, GateUpFlat                []float32
	X, XN, Q, K, V, AttnOut, Proj      [][]float32
	Gate, Up, Hidden, QKV, GateUp      [][]float32
}

func reuseBatchViews(flat *[]float32, views *[][]float32, p, stride int) [][]float32 {
	ensureLenNoClear(flat, p*stride)
	if cap(*views) < p {
		*views = make([][]float32, p)
	} else {
		*views = (*views)[:p]
	}
	for i := 0; i < p; i++ {
		(*views)[i] = (*flat)[i*stride : (i+1)*stride : (i+1)*stride]
	}
	return *views
}

// Batched (prefill) matvec and forward pass. During prompt processing the
// per-token path re-streams every weight from memory once per token; batching
// reads each weight row once and applies it to all prompt tokens, so a P-token
// prompt reads the weights ~once instead of P times. Prefill is memory-bandwidth
// bound, so this is close to a P-fold speedup for the matvecs.

// matvecBatch computes outs[p][r] = dot(weightRow_r, xs[p]) for every token p and
// row r. For quantized weights it dequantizes each row ONCE (the expensive
// nibble-unpack + scale step) into a scratch buffer, then does P cheap AVX2 float
// dots against it. Prefill matvecs are compute-bound, so amortizing the
// dequantization across the whole prompt chunk is the win. outs[p] must be
// pre-sized to the weight's row count.
func matvecBatch(w Weight, xs, outs [][]float32) {
	p := len(xs)
	if p == 0 {
		return
	}
	cols := len(xs[0])
	if cols == 0 {
		return
	}

	if w.F32 != nil {
		rows := len(w.F32) / cols
		parallelRows(rows, func(start, end int) {
			for r := start; r < end; r++ {
				row := w.F32[r*cols : (r+1)*cols]
				for t := 0; t < p; t++ {
					outs[t][r] = DotF32(row, xs[t])
				}
			}
		})
		return
	}

	if useQ8Activations && matvecBatchQ8(w, xs, outs) {
		return
	}

	dequant := dequantRowInto(w, cols)
	if dequant == nil {
		// No batched dequant for this type: fall back to per-token matvec.
		for t := 0; t < p; t++ {
			w.MatvecInto(xs[t], &outs[t])
		}
		return
	}
	rowBytes := len(w.Raw) / w.Rows
	parallelRows(w.Rows, func(start, end int) {
		deq := make([]float32, cols) // one scratch row per worker chunk
		for r := start; r < end; r++ {
			dequant(w.Raw[r*rowBytes:(r+1)*rowBytes], cols, deq)
			for t := 0; t < p; t++ {
				outs[t][r] = DotF32(deq, xs[t])
			}
		}
	})
}

// dequantRowInto returns the row-dequant function for a quantized weight, or nil
// if cols is incompatible or the type has no dequantizer.
func dequantRowInto(w Weight, cols int) func(row []byte, cols int, out []float32) {
	switch w.Type {
	case GGMLTypeQ4_1:
		if cols%32 == 0 {
			return DequantRowQ4_1Into
		}
	case GGMLTypeQ5_0:
		if cols%32 == 0 {
			return DequantRowQ5_0Into
		}
	case GGMLTypeQ5_1:
		if cols%32 == 0 {
			return DequantRowQ5_1Into
		}
	case GGMLTypeQ8_1:
		if cols%32 == 0 {
			return DequantRowQ8_1Into
		}
	case GGMLTypeQ2_K:
		if cols%256 == 0 {
			return DequantRowQ2KInto
		}
	case GGMLTypeQ3_K:
		if cols%256 == 0 {
			return DequantRowQ3KInto
		}
	case GGMLTypeQ4_K:
		if cols%256 == 0 {
			return DequantRowQ4KInto
		}
	case GGMLTypeQ6_K:
		if cols%256 == 0 {
			return DequantRowQ6KInto
		}
	}
	return nil
}

// ForwardBatchInto processes a chunk of prompt tokens (positions
// startPos..startPos+len(tokens)-1) through the standard transformer, populating
// the KV cache. The matvecs are batched so each weight is streamed once for the
// whole chunk. When computeLast is set, the final token's logits are written to
// logits. Only the non-fused standard path is supported (callers must check).
func ForwardBatchInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, computeLast bool, logits *[]float32) {
	p := len(tokens)
	if p == 0 {
		return
	}
	dim := config.Dim
	headDim := config.HeadDim
	valueDim := config.ValueDim
	kvMul := max(1, config.KVMul)
	qLen := config.NHeads * headDim
	kLen := config.NKVHeads * headDim
	vLen := config.NKVHeads * valueDim
	attnLen := config.NHeads * valueDim
	hDim := config.HiddenDim
	interleaved := ropeInterleaved(config.Arch)
	eps := config.RMSNormEps

	b := &buf.batch
	X := reuseBatchViews(&b.XFlat, &b.X, p, dim)
	XN := reuseBatchViews(&b.XNFlat, &b.XN, p, dim)
	Q := reuseBatchViews(&b.QFlat, &b.Q, p, qLen)
	K := reuseBatchViews(&b.KFlat, &b.K, p, kLen)
	V := reuseBatchViews(&b.VFlat, &b.V, p, vLen)
	AttnOut := reuseBatchViews(&b.AttnOutFlat, &b.AttnOut, p, attnLen)
	Proj := reuseBatchViews(&b.ProjFlat, &b.Proj, p, dim)
	Gate := reuseBatchViews(&b.GateFlat, &b.Gate, p, hDim)
	Up := reuseBatchViews(&b.UpFlat, &b.Up, p, hDim)
	QKV := [][]float32(nil)
	GateUp := [][]float32(nil)
	Hidden := reuseBatchViews(&b.HiddenFlat, &b.Hidden, p, hDim)

	for t := 0; t < p; t++ {
		weights.TokenEmbd.RowInto(int(tokens[t]), dim, &X[t])
		if config.EmbeddingScale != 1 {
			ScaleF32(X[t], config.EmbeddingScale)
		}
	}

	scale := config.AttentionScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(headDim)))
	}
	var sinS, cosS []float32

	for l := 0; l < config.NLayers; l++ {
		layer := weights.Layers[l]
		for t := 0; t < p; t++ {
			rmsNormInto(X[t], layer.AttnNorm, eps, &XN[t])
		}
		if layer.HasQKV {
			qkvLen := qLen + kLen + vLen
			QKV = reuseBatchViews(&b.QKVFlat, &b.QKV, p, qkvLen)
			matvecBatch(layer.WQKV, XN, QKV)
			for t := 0; t < p; t++ {
				copy(Q[t], QKV[t][:qLen])
				copy(K[t], QKV[t][qLen:qLen+kLen])
				copy(V[t], QKV[t][qLen+kLen:qLen+kLen+vLen])
			}
		} else {
			matvecBatch(layer.WQ, XN, Q)
			matvecBatch(layer.WK, XN, K)
			matvecBatch(layer.WV, XN, V)
		}

		// RoPE + KV cache write are sequential: RoPE reuses shared sin/cos
		// scratch, and all K/V must be resident before attention so a token can
		// attend to earlier tokens in the same chunk.
		for t := 0; t < p; t++ {
			addInPlace(Q[t], layer.BQ)
			addInPlace(K[t], layer.BK)
			addInPlace(V[t], layer.BV)
			pos := startPos + t
			half, nCache := prepareRopeScratch(pos, headDim, config.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &sinS, &cosS)
			applyPreparedRope(Q[t], headDim, config.NHeads, half, nCache, sinS, cosS, interleaved)
			applyPreparedRope(K[t], headDim, config.NKVHeads, half, nCache, sinS, cosS, interleaved)
			kStart := pos * cache.PerPosKDim
			vStart := pos * cache.PerPosVDim
			copy(cache.K[l][kStart:kStart+min(len(K[t]), cache.PerPosKDim)], K[t])
			copy(cache.V[l][vStart:vStart+min(len(V[t]), cache.PerPosVDim)], V[t])
		}

		// Attention is independent per token, so spread the chunk across workers.
		attend := func(ts, te int) {
			for t := ts; t < te; t++ {
				pos := startPos + t
				attnStart := 0
				if config.layerUsesSWA(l) {
					attnStart = max(0, pos-config.SlidingWindow)
				}
				clear(AttnOut[t])
				for h := 0; h < config.NHeads; h++ {
					kvH := h / kvMul
					onlineAttention(
						Q[t][h*headDim:h*headDim+headDim],
						cache.K[l][kvH*headDim:],
						cache.V[l][kvH*valueDim:],
						cache.PerPosKDim, cache.PerPosVDim,
						headDim, valueDim,
						attnStart, pos, scale,
						config.AttnLogitSoftcap,
						AttnOut[t][h*valueDim:h*valueDim+valueDim],
					)
				}
			}
		}
		if p > 1 {
			parallelChunks(p, attend)
		} else {
			attend(0, p)
		}

		matvecBatch(layer.WO, AttnOut, Proj)
		for t := 0; t < p; t++ {
			if config.ResidualScale != 1 {
				ScaleF32(Proj[t], config.ResidualScale)
			}
			addInPlace(X[t], Proj[t])
			rmsNormInto(X[t], layer.FFNNorm, eps, &XN[t])
		}
		if layer.HasGateUp {
			gateUpLen := hDim * 2
			GateUp = reuseBatchViews(&b.GateUpFlat, &b.GateUp, p, gateUpLen)
			matvecBatch(layer.WGateUp, XN, GateUp)
			for t := 0; t < p; t++ {
				copy(Gate[t], GateUp[t][:hDim])
				copy(Up[t], GateUp[t][hDim:gateUpLen])
			}
		} else {
			matvecBatch(layer.W1, XN, Gate)
			matvecBatch(layer.W3, XN, Up)
		}
		if p > 1 {
			parallelChunks(p, func(ts, te int) {
				for t := ts; t < te; t++ {
					siluMulF32(Gate[t][:hDim], Up[t][:hDim], Hidden[t][:hDim])
				}
			})
		} else {
			siluMulF32(Gate[0][:hDim], Up[0][:hDim], Hidden[0][:hDim])
		}
		matvecBatch(layer.W2, Hidden, Proj)
		for t := 0; t < p; t++ {
			if config.ResidualScale != 1 {
				ScaleF32(Proj[t], config.ResidualScale)
			}
			addInPlace(X[t], Proj[t])
		}
	}

	if computeLast {
		last := p - 1
		rmsNormInto(X[last], weights.OutputNorm, eps, &buf.XN)
		weights.Output.MatvecInto(buf.XN, logits)
		if config.LogitScale != 1 {
			ScaleF32(*logits, 1/config.LogitScale)
		}
		if config.FinalLogitSoftcap > 0 {
			softcapF32(*logits, config.FinalLogitSoftcap)
		}
	}
}
