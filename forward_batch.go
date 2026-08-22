package gopherllm

import (
	"math"
	"sync"
)

// Batched quantized matvecs previously allocated one dequantized row per
// worker chunk and projection, producing thousands of short-lived arrays over
// a 26-layer prefill. Reusing those independent scratch rows removes GC work
// without changing arithmetic. Risk is bounded retention of the largest row
// seen per active chunk; rollback is to allocate make([]float32, cols) inline.
var batchDequantScratchPool = sync.Pool{New: func() any {
	scratch := make([]float32, 0)
	return &scratch
}}

// batchDecodeBuffer owns the large activation slabs used by batched prefill.
// It hangs off DecodeBuffer so prompt chunks and subsequent requests reuse the
// same backing arrays instead of feeding tens or hundreds of MiB to the GC.
type batchDecodeBuffer struct {
	XFlat, XNFlat, QFlat, KFlat, VFlat      []float32
	AttnOutFlat, ProjFlat, AttnProjFlat     []float32
	GateFlat, UpFlat, HiddenFlat            []float32
	QKVFlat, GateUpFlat                     []float32
	RopeSinFlat, RopeCosFlat                []float32
	RopeSWASinFlat, RopeSWACosFlat          []float32
	X, XN, Q, K, V, AttnOut, Proj, AttnProj [][]float32
	Gate, Up, Hidden, QKV, GateUp           [][]float32
	RopeSin, RopeCos                        [][]float32
	RopeSWASin, RopeSWACos                  [][]float32
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

// dotRowIntoBatch writes one weight row's dot product against every token into
// column r of each output.
//
// The naive form — DotF32(row, xs[t]) per token — is what every target other
// than arm64 still runs, and it is the right choice there. On arm64, dotF32x4
// is a hand-written 4-wide NEON kernel that holds the shared row in registers
// while consuming four token vectors against it (attention_gqa_arm64.s, written
// for the transposed case: four query heads against one shared K row), so
// tiling the token loop through it reuses each row four times per load.
//
// The tiled flag is gated on hasFastDotF32x4 rather than applied everywhere,
// because where dotF32x4 is a portable composition of four DotF32 calls the
// tiling is pure overhead: measured on amd64 it cost ~1.4x on the vision tower.
// The win is register-level row reuse, not an algorithmic change, so it exists
// only where the kernel does.
//
// This is the batched/prefill path, so it covers vision-tower encoding — where
// every block runs seven of these over the full patch grid — as well as text
// prefill.
func dotRowIntoBatch(row []float32, xs, outs [][]float32, r, p, cols int, tiled bool) {
	t := 0
	if tiled {
		for ; t+4 <= p; t += 4 {
			s0, s1, s2, s3 := dotF32x4(&xs[t][0], &xs[t+1][0], &xs[t+2][0], &xs[t+3][0], &row[0], cols)
			outs[t][r], outs[t+1][r] = s0, s1
			outs[t+2][r], outs[t+3][r] = s2, s3
		}
	}
	for ; t < p; t++ {
		outs[t][r] = DotF32(row, xs[t])
	}
}

// matvecBatch computes outs[p][r] = dot(weightRow_r, xs[p]) for every token p and
// row r. For quantized weights it dequantizes each row ONCE (the expensive
// nibble-unpack + scale step) into a scratch buffer, then does P cheap float
// dots against it. Prefill matvecs are compute-bound, so amortizing the
// dequantization across the whole prompt chunk is the win. outs[p] must be
// pre-sized to the weight's row count.
func matvecBatch(w Weight, xs, outs [][]float32) {
	matvecBatchWithQ8(w, xs, outs, useQ8Activations.Load())
}

// matvecBatchNoQ8 retains the weight-stationary batched traversal while
// deliberately using the dequantize-once + float/NEON dot path for quantized
// rows. Embedding models can select it per Runner without flipping the
// process-wide Q8 activation setting used by simultaneously active decoders.
func matvecBatchNoQ8(w Weight, xs, outs [][]float32) {
	matvecBatchWithQ8(w, xs, outs, false)
}

// matvecBatchWithQ8 is matvecBatch with an explicit choice for the optional
// Q8-activation kernel. Keeping the choice as an argument is important for
// embedding runners: useQ8Activations is process-global and cannot safely be
// toggled around one model invocation.
func matvecBatchWithQ8(w Weight, xs, outs [][]float32, useQ8 bool) {
	p := len(xs)
	if p == 0 {
		return
	}
	cols := len(xs[0])
	if cols == 0 {
		return
	}

	// tiled reports whether the 4-wide token kernel can be used: it takes bare
	// pointers, so every token vector must be at least cols long.
	tiled := hasFastDotF32x4 && p >= 4
	if tiled {
		for t := 0; t < p; t++ {
			if len(xs[t]) < cols {
				tiled = false
				break
			}
		}
	}

	if w.F32 != nil {
		rows := len(w.F32) / cols
		parallelRowsBatched(rows, func(start, end int) {
			for r := start; r < end; r++ {
				row := w.F32[r*cols : (r+1)*cols]
				dotRowIntoBatch(row, xs, outs, r, p, cols, tiled)
			}
		})
		return
	}

	// Raw (unexpanded) F16/BF16/F32 weights: keep the same weight-stationary
	// shape as the F32 path above rather than falling through to the
	// per-token fallback, which would re-read the whole matrix once per token.
	// For f16 the row dot is a real vector kernel (see rawScalarDot), so this
	// reads half the bytes of an expanded F32 copy and converts in-register.
	//
	// Routing these through dequantRowInto instead -- decoding each row once
	// into an f32 scratch and then doing p plain dots, as the quantized path
	// does -- was measured on the vision tower and is 1.5x SLOWER (47s -> 70s
	// median, interleaved, 448 patches). Amortizing the conversion is not
	// worth it here: dotF32F16AVX2 fuses VCVTPH2PS into the FMA loop, so the
	// conversion is nearly free in-register, while the f32 scratch doubles the
	// bytes the token loop reads per row. Hoisting the conversion trades a
	// free operation for real memory traffic.
	//
	// This is the path out-of-core models take for every scalar matrix, and
	// the one a vision tower's f16 mmproj takes for all seven of its
	// per-block projections.
	if rawScalarWeight(w) && w.Rows > 0 && w.Cols == cols {
		if width := scalarBytesPerElement(w.Type); width > 0 {
			rowBytes := cols * width
			if len(w.Raw) >= w.Rows*rowBytes {
				parallelRowsBatched(w.Rows, func(start, end int) {
					for r := start; r < end; r++ {
						off := r * rowBytes
						for t := 0; t < p; t++ {
							outs[t][r] = rawScalarDot(w.Raw, w.Type, off, xs[t], cols)
						}
					}
				})
				return
			}
		}
	}

	if useQ8 && matvecBatchQ8(w, xs, outs) {
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
	parallelRowsBatched(w.Rows, func(start, end int) {
		scratch := batchDequantScratchPool.Get().(*[]float32)
		if cap(*scratch) < cols {
			*scratch = make([]float32, cols)
		} else {
			*scratch = (*scratch)[:cols]
		}
		deq := *scratch
		for r := start; r < end; r++ {
			dequant(w.Raw[r*rowBytes:(r+1)*rowBytes], cols, deq)
			dotRowIntoBatch(deq, xs, outs, r, p, cols, tiled)
		}
		*scratch = deq[:0]
		batchDequantScratchPool.Put(scratch)
	})
}

// matvecBatch2/3 keep projections with a shared activation batch together.
// On targets with the Q8K batch path this avoids re-quantizing every token and
// collapsing the worker pool between Q/K/V or gate/up projections. Unsupported
// formats retain the ordinary independently-dispatched batch implementation.
func matvecBatch2(a, b Weight, xs, aOut, bOut [][]float32) {
	if matvecBatchQ8Fused2(a, b, xs, aOut, bOut) {
		return
	}
	matvecBatch(a, xs, aOut)
	matvecBatch(b, xs, bOut)
}

// matvecBatch2NoQ8 is the per-Runner float counterpart to matvecBatch2. It
// intentionally does not enter the fused Q8 route: both projections keep the
// same batched row traversal, just without quantizing their shared activations.
func matvecBatch2NoQ8(a, b Weight, xs, aOut, bOut [][]float32) {
	matvecBatchNoQ8(a, xs, aOut)
	matvecBatchNoQ8(b, xs, bOut)
}

func matvecBatch3(a, b, c Weight, xs, aOut, bOut, cOut [][]float32) {
	if matvecBatchQ8Fused3(a, b, c, xs, aOut, bOut, cOut) {
		return
	}
	matvecBatch(a, xs, aOut)
	matvecBatch(b, xs, bOut)
	matvecBatch(c, xs, cOut)
}

// dequantRowInto returns the row-dequant function for a quantized weight, or nil
// if cols is incompatible or the type has no dequantizer.
func dequantRowInto(w Weight, cols int) func(row []byte, cols int, out []float32) {
	switch w.Type {
	case GGMLTypeQ8_0:
		if cols%32 == 0 {
			return DequantRowQ8_0Into
		}
	case GGMLTypeQ4_0:
		if cols%32 == 0 {
			return DequantRowQ4_0Into
		}
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
	case GGMLTypeQ5_K:
		if cols%256 == 0 {
			return DequantRowQ5KInto
		}
	case GGMLTypeQ6_K:
		if cols%256 == 0 {
			return DequantRowQ6KInto
		}
	case GGMLTypeQ8_K:
		if cols%256 == 0 {
			return DequantRowQ8KInto
		}
	case GGMLTypeMXFP4:
		if cols%32 == 0 {
			return DequantRowMXFP4Into
		}
	case GGMLTypeTQ1_0:
		if cols%256 == 0 {
			return DequantRowTQ1_0Into
		}
	case GGMLTypeTQ2_0:
		if cols%256 == 0 {
			return DequantRowTQ2_0Into
		}
	case GGMLTypeQ1_0:
		if cols%128 == 0 {
			return DequantRowQ1_0Into
		}
	case GGMLTypeQ2_0:
		if cols%64 == 0 {
			return DequantRowQ2_0Into
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
	forwardBatchInto(config, weights, cache, buf, tokens, startPos, computeLast, logits, nil)
}

// forwardBatchPoolInto is the embedding counterpart of ForwardBatchInto. It
// adds the output-normalized hidden state for every token in the chunk to sum.
// Keeping pooling inside the batched graph avoids re-streaming model weights
// once per token just to obtain the intermediate states needed for mean pooling.
// sum must have config.Dim elements; it is intentionally internal because the
// caller owns aggregation across chunks and final L2 normalization.
func forwardBatchPoolInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, sum []float32) {
	if len(sum) != config.Dim {
		panic("gopherllm: batch embedding sum has wrong dimension")
	}
	forwardBatchInto(config, weights, cache, buf, tokens, startPos, false, nil, sum)
}

// forwardBatchInto is the common batched transformer implementation. poolSum,
// when non-nil, requests final hidden states for all tokens; otherwise the
// normal generation path only normalizes/projects the final token.
func forwardBatchInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, computeLast bool, logits *[]float32, poolSum []float32) {
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

	b := &buf.batch
	X := reuseBatchViews(&b.XFlat, &b.X, p, dim)
	XN := reuseBatchViews(&b.XNFlat, &b.XN, p, dim)
	Q := reuseBatchViews(&b.QFlat, &b.Q, p, qLen)
	K := reuseBatchViews(&b.KFlat, &b.K, p, kLen)
	V := reuseBatchViews(&b.VFlat, &b.V, p, vLen)
	AttnOut := reuseBatchViews(&b.AttnOutFlat, &b.AttnOut, p, attnLen)
	Proj := reuseBatchViews(&b.ProjFlat, &b.Proj, p, dim)
	var AttnProj [][]float32
	if config.ParallelResidual {
		AttnProj = reuseBatchViews(&b.AttnProjFlat, &b.AttnProj, p, dim)
	}
	Gate := reuseBatchViews(&b.GateFlat, &b.Gate, p, hDim)
	Up := reuseBatchViews(&b.UpFlat, &b.Up, p, hDim)
	QKV := [][]float32(nil)
	GateUp := [][]float32(nil)
	Hidden := reuseBatchViews(&b.HiddenFlat, &b.Hidden, p, hDim)

	usesPosEmbd := config.usesAbsolutePositionEmbd()
	for t := 0; t < p; t++ {
		weights.TokenEmbd.RowInto(int(tokens[t]), dim, &X[t])
		if emb, ok := buf.ImageEmbeds[startPos+t]; ok {
			copy(X[t][:dim], emb)
		} else if config.EmbeddingScale != 1 {
			ScaleF32(X[t], config.EmbeddingScale)
		}
		if usesPosEmbd {
			weights.PositionEmbd.RowInto(startPos+t, dim, &buf.PosEmbd)
			addInPlace(X[t][:dim], buf.PosEmbd[:dim])
		}
	}

	// RoPE depends on position, but not on layer. Computing its sin/cos pairs
	// here instead of inside every layer avoids NLayers identical Sincos sweeps
	// during prompt prefill. The tables share DecodeBuffer's bounded batch
	// scratch and are overwritten by the next chunk.
	ropeDim := config.RopeDimensionCount
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}
	ropeDim -= ropeDim % 2
	ropeHalf := ropeDim / 2
	ropePairs := min(ropeHalf, len(buf.RopeInvFreq))
	var ropeSin, ropeCos [][]float32
	if ropePairs > 0 {
		ropeSin = reuseBatchViews(&b.RopeSinFlat, &b.RopeSin, p, ropePairs)
		ropeCos = reuseBatchViews(&b.RopeCosFlat, &b.RopeCos, p, ropePairs)
		for t := 0; t < p; t++ {
			prepareRopeScratch(startPos+t, headDim, config.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &ropeSin[t], &ropeCos[t])
		}
	}
	swaRopePairs := min(ropeHalf, len(buf.RopeSWAInvFreq))
	var swaRopeSin, swaRopeCos [][]float32
	if swaRopePairs > 0 {
		swaRopeSin = reuseBatchViews(&b.RopeSWASinFlat, &b.RopeSWASin, p, swaRopePairs)
		swaRopeCos = reuseBatchViews(&b.RopeSWACosFlat, &b.RopeSWACos, p, swaRopePairs)
		for t := 0; t < p; t++ {
			prepareRopeScratch(startPos+t, headDim, config.RopeDimensionCount, buf.RopeSWAInvFreq, buf.RopeSWAMscale, &swaRopeSin[t], &swaRopeCos[t])
		}
	}

	scale := config.AttentionScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(float64(headDim)))
	}
	qScale := float32(1)
	if config.Arch == "phi2" {
		qScale, scale = scale, 1
	}
	for l := 0; l < config.NLayers; l++ {
		layer := weights.Layers[l]
		for t := 0; t < p; t++ {
			if config.usesPostNormOnly() {
				copy(XN[t], X[t])
			} else {
				normalizeDecoderInto(config, X[t], layer.AttnNorm, layer.AttnNormBias, &XN[t])
			}
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
			matvecBatch3(layer.WQ, layer.WK, layer.WV, XN, Q, K, V)
		}

		// RoPE + KV cache write are sequential: RoPE reuses shared sin/cos
		// scratch, and all K/V must be resident before attention so a token can
		// attend to earlier tokens in the same chunk.
		for t := 0; t < p; t++ {
			addInPlace(Q[t], layer.BQ)
			addInPlace(K[t], layer.BK)
			addInPlace(V[t], layer.BV)
			normalizeProjectedQKInPlace(config, layer, Q[t], K[t])
			pos := startPos + t
			if ropePairs > 0 && config.layerUsesRoPE(l) {
				activePairs := ropePairs
				activeSin, activeCos := ropeSin[t], ropeCos[t]
				if config.layerUsesSWA(l) && swaRopePairs > 0 {
					activePairs = swaRopePairs
					activeSin, activeCos = swaRopeSin[t], swaRopeCos[t]
				}
				applyPreparedRope(Q[t], headDim, config.NHeads, ropeHalf, activePairs, activeSin, activeCos, interleaved)
				applyPreparedRope(K[t], headDim, config.NKVHeads, ropeHalf, activePairs, activeSin, activeCos, interleaved)
			}
			if temperature := attentionTemperatureAt(config, pos); temperature != 1 {
				ScaleF32(Q[t], temperature)
			}
			if qScale != 1 {
				ScaleF32(Q[t], qScale)
			}
			cache.storeKV(l, pos, K[t], V[t])
		}

		// Attention is independent per token, so spread the chunk across workers.
		alibi := config.usesALiBi()
		attend := func(ts, te int) {
			for t := ts; t < te; t++ {
				pos := startPos + t
				attnStart := 0
				if config.layerUsesSWA(l) {
					attnStart = max(0, pos-config.SlidingWindow)
				}
				clear(AttnOut[t])
				if useGroupedGQAAttention && kvMul > 1 && config.NKVHeads > 0 && len(layer.AttnSinks) == 0 && !alibi {
					for kvH := 0; kvH < config.NKVHeads; kvH++ {
						hStart := kvH * kvMul
						hEnd := min(hStart+kvMul, config.NHeads)
						if hStart >= hEnd {
							break
						}
						cache.attendHeadGroup(l, kvH,
							Q[t][hStart*headDim:hEnd*headDim], hEnd-hStart,
							headDim, valueDim, attnStart, pos, scale, config.AttnLogitSoftcap,
							AttnOut[t][hStart*valueDim:hEnd*valueDim])
					}
				} else {
					for h := 0; h < config.NHeads; h++ {
						kvH := h / kvMul
						var alibiSlope float32
						if alibi {
							alibiSlope = aLiBiSlope(h, config.NHeads, config.ALiBiMaxBias)
						}
						cache.attendHeadWithSink(l, kvH, Q[t][h*headDim:h*headDim+headDim],
							headDim, valueDim, attnStart, pos, scale,
							config.AttnLogitSoftcap, alibiSlope, 0, false,
							AttnOut[t][h*valueDim:h*valueDim+valueDim])
					}
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
			addInPlace(Proj[t], layer.BO)
			if layer.PostAttnNorm != nil {
				rmsNormInto(Proj[t], layer.PostAttnNorm, config.RMSNormEps, &Proj[t])
			}
			if config.ResidualScale != 1 {
				ScaleF32(Proj[t], config.ResidualScale)
			}
			if config.ParallelResidual {
				copy(AttnProj[t], Proj[t])
			} else {
				addInPlace(X[t], Proj[t])
			}
			if config.sharesParallelBranchNorm() {
				// Parallel StableLM and Phi-2 blocks feed both residual
				// branches from the same normalized block input.
			} else if config.usesPostNormOnly() {
				copy(XN[t], X[t])
			} else {
				normalizeDecoderInto(config, X[t], layer.FFNNorm, layer.FFNNormBias, &XN[t])
			}
		}
		if config.usesPlainMLP() {
			matvecBatch(layer.W3, XN, Up)
			for t := 0; t < p; t++ {
				addInPlace(Up[t], layer.FFNUpBias)
			}
		} else if layer.HasGateUp {
			gateUpLen := hDim * 2
			GateUp = reuseBatchViews(&b.GateUpFlat, &b.GateUp, p, gateUpLen)
			matvecBatch(layer.WGateUp, XN, GateUp)
			for t := 0; t < p; t++ {
				copy(Gate[t], GateUp[t][:hDim])
				copy(Up[t], GateUp[t][hDim:gateUpLen])
			}
		} else {
			matvecBatch2(layer.W1, layer.W3, XN, Gate, Up)
		}
		activateFFN := func(ts, te int) {
			for t := ts; t < te; t++ {
				if config.usesPlainMLP() {
					if config.UseExactGELU {
						for i := 0; i < hDim; i++ {
							Hidden[t][i] = geluExact(Up[t][i])
						}
					} else {
						for i := 0; i < hDim; i++ {
							Hidden[t][i] = geluTanhScalar(Up[t][i])
						}
					}
				} else if config.UseGELU {
					geluMulF32(Gate[t][:hDim], Up[t][:hDim], Hidden[t][:hDim])
				} else {
					siluMulF32(Gate[t][:hDim], Up[t][:hDim], Hidden[t][:hDim])
				}
			}
		}
		if p > 1 {
			parallelChunks(p, func(ts, te int) {
				activateFFN(ts, te)
			})
		} else {
			activateFFN(0, 1)
		}
		matvecBatch(layer.W2, Hidden, Proj)
		for t := 0; t < p; t++ {
			addInPlace(Proj[t], layer.FFNDownBias)
			if layer.PostFFNNorm != nil {
				rmsNormInto(Proj[t], layer.PostFFNNorm, config.RMSNormEps, &Proj[t])
			}
			if config.ResidualScale != 1 {
				ScaleF32(Proj[t], config.ResidualScale)
			}
			addInPlace(X[t], Proj[t])
			if config.ParallelResidual {
				addInPlace(X[t], AttnProj[t])
			}
		}
	}

	if poolSum != nil {
		for t := 0; t < p; t++ {
			normalizeDecoderInto(config, X[t], weights.OutputNorm, weights.OutputNormBias, &XN[t])
			addInPlace(poolSum, XN[t])
		}
	}
	if computeLast {
		last := p - 1
		normalizeDecoderInto(config, X[last], weights.OutputNorm, weights.OutputNormBias, &buf.XN)
		weights.Output.MatvecInto(buf.XN, logits)
		addInPlace(*logits, weights.OutputBias)
		if config.LogitScale != 1 {
			ScaleF32(*logits, 1/config.LogitScale)
		}
		if config.FinalLogitSoftcap > 0 {
			softcapF32(*logits, config.FinalLogitSoftcap)
		}
	}
}
