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

// batchAttentionTask holds the per-layer state for independent prompt-token
// attention work. Keeping it in a pool lets ForwardBatchInto reuse the worker
// pool without creating an escaping closure for every transformer layer.
type batchAttentionTask struct {
	cache      *KVCache
	q, attnOut [][]float32

	l, startPos                                int
	headDim, valueDim, nHeads, nKVHeads, kvMul int
	slidingWindow                              int
	scale, softcap, alibiMaxBias               float32
	usesSWA, groupedGQA, alibi                 bool
}

func (t *batchAttentionTask) runRows(start, end int) {
	for token := start; token < end; token++ {
		pos := t.startPos + token
		attnStart := 0
		if t.usesSWA {
			attnStart = max(0, pos-t.slidingWindow)
		}
		clear(t.attnOut[token])
		if t.groupedGQA {
			for kvH := 0; kvH < t.nKVHeads; kvH++ {
				hStart := kvH * t.kvMul
				hEnd := min(hStart+t.kvMul, t.nHeads)
				if hStart >= hEnd {
					break
				}
				t.cache.attendHeadGroup(t.l, kvH,
					t.q[token][hStart*t.headDim:hEnd*t.headDim], hEnd-hStart,
					t.headDim, t.valueDim, attnStart, pos, t.scale, t.softcap,
					t.attnOut[token][hStart*t.valueDim:hEnd*t.valueDim])
			}
			continue
		}
		for h := 0; h < t.nHeads; h++ {
			kvH := h / t.kvMul
			var alibiSlope float32
			if t.alibi {
				alibiSlope = aLiBiSlope(h, t.nHeads, t.alibiMaxBias)
			}
			t.cache.attendHeadWithSink(t.l, kvH, t.q[token][h*t.headDim:h*t.headDim+t.headDim],
				t.headDim, t.valueDim, attnStart, pos, t.scale, t.softcap,
				alibiSlope, 0, false,
				t.attnOut[token][h*t.valueDim:h*t.valueDim+t.valueDim])
		}
	}
}

var batchAttentionTaskPool = sync.Pool{New: func() any { return new(batchAttentionTask) }}

// batchActivationTask is the allocation-free equivalent of the short
// per-token FFN activation closure in ForwardBatchInto.
type batchActivationTask struct {
	gate, up, hidden             [][]float32
	hiddenDim                    int
	plainMLP, exactGELU, useGELU bool
}

func (t *batchActivationTask) runRows(start, end int) {
	for token := start; token < end; token++ {
		if t.plainMLP {
			if t.exactGELU {
				for i := 0; i < t.hiddenDim; i++ {
					t.hidden[token][i] = geluExact(t.up[token][i])
				}
			} else {
				for i := 0; i < t.hiddenDim; i++ {
					t.hidden[token][i] = geluTanhScalar(t.up[token][i])
				}
			}
		} else if t.useGELU {
			geluMulF32(t.gate[token][:t.hiddenDim], t.up[token][:t.hiddenDim], t.hidden[token][:t.hiddenDim])
		} else {
			siluMulF32(t.gate[token][:t.hiddenDim], t.up[token][:t.hiddenDim], t.hidden[token][:t.hiddenDim])
		}
	}
}

var batchActivationTaskPool = sync.Pool{New: func() any { return new(batchActivationTask) }}

// batchF32MatvecTask streams an expanded f32 matrix once across a prompt
// batch. Its pooled state replaces the callback allocation on every dense
// batch projection.
type batchF32MatvecTask struct {
	weights  []float32
	xs, outs [][]float32
	p, cols  int
	tiled    bool
}

func (t *batchF32MatvecTask) runRows(start, end int) {
	for rowIndex := start; rowIndex < end; rowIndex++ {
		row := t.weights[rowIndex*t.cols : (rowIndex+1)*t.cols]
		dotRowIntoBatch(row, t.xs, t.outs, rowIndex, t.p, t.cols, t.tiled)
	}
}

var batchF32MatvecTaskPool = sync.Pool{New: func() any { return new(batchF32MatvecTask) }}

// batchRawScalarMatvecTask keeps raw f16/bf16/f32 weights in their compact
// representation while applying every row to the prompt batch.
type batchRawScalarMatvecTask struct {
	raw      []byte
	typeID   GGMLType
	xs, outs [][]float32
	p, cols  int
	rowBytes int
}

func (t *batchRawScalarMatvecTask) runRows(start, end int) {
	for rowIndex := start; rowIndex < end; rowIndex++ {
		offset := rowIndex * t.rowBytes
		for token := 0; token < t.p; token++ {
			t.outs[token][rowIndex] = rawScalarDot(t.raw, t.typeID, offset, t.xs[token], t.cols)
		}
	}
}

var batchRawScalarMatvecTaskPool = sync.Pool{New: func() any { return new(batchRawScalarMatvecTask) }}

// batchDequantMatvecTask owns the row range for formats that are dequantized
// once then dotted against every prompt token. The scratch remains local to a
// worker range exactly as in the callback implementation.
type batchDequantMatvecTask struct {
	raw      []byte
	dequant  func(src []byte, cols int, dst []float32)
	xs, outs [][]float32
	p, cols  int
	rowBytes int
	tiled    bool
}

func (t *batchDequantMatvecTask) runRows(start, end int) {
	scratch := batchDequantScratchPool.Get().(*[]float32)
	if cap(*scratch) < t.cols {
		*scratch = make([]float32, t.cols)
	} else {
		*scratch = (*scratch)[:t.cols]
	}
	dequantized := *scratch
	for rowIndex := start; rowIndex < end; rowIndex++ {
		t.dequant(t.raw[rowIndex*t.rowBytes:(rowIndex+1)*t.rowBytes], t.cols, dequantized)
		dotRowIntoBatch(dequantized, t.xs, t.outs, rowIndex, t.p, t.cols, t.tiled)
	}
	*scratch = dequantized[:0]
	batchDequantScratchPool.Put(scratch)
}

var batchDequantMatvecTaskPool = sync.Pool{New: func() any { return new(batchDequantMatvecTask) }}

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
		task := batchF32MatvecTaskPool.Get().(*batchF32MatvecTask)
		task.weights, task.xs, task.outs = w.F32, xs, outs
		task.p, task.cols, task.tiled = p, cols, tiled
		parallelRowsBatchedTask(rows, task)
		*task = batchF32MatvecTask{}
		batchF32MatvecTaskPool.Put(task)
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
				task := batchRawScalarMatvecTaskPool.Get().(*batchRawScalarMatvecTask)
				task.raw, task.typeID, task.xs, task.outs = w.Raw, w.Type, xs, outs
				task.p, task.cols, task.rowBytes = p, cols, rowBytes
				parallelRowsBatchedTask(w.Rows, task)
				*task = batchRawScalarMatvecTask{}
				batchRawScalarMatvecTaskPool.Put(task)
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
	task := batchDequantMatvecTaskPool.Get().(*batchDequantMatvecTask)
	task.raw, task.dequant, task.xs, task.outs = w.Raw, dequant, xs, outs
	task.p, task.cols, task.rowBytes, task.tiled = p, cols, rowBytes, tiled
	parallelRowsBatchedTask(w.Rows, task)
	*task = batchDequantMatvecTask{}
	batchDequantMatvecTaskPool.Put(task)
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
		attendTask := batchAttentionTaskPool.Get().(*batchAttentionTask)
		attendTask.cache, attendTask.q, attendTask.attnOut = cache, Q, AttnOut
		attendTask.l, attendTask.startPos = l, startPos
		attendTask.headDim, attendTask.valueDim = headDim, valueDim
		attendTask.nHeads, attendTask.nKVHeads, attendTask.kvMul = config.NHeads, config.NKVHeads, kvMul
		attendTask.slidingWindow = config.SlidingWindow
		attendTask.scale, attendTask.softcap, attendTask.alibiMaxBias = scale, config.AttnLogitSoftcap, config.ALiBiMaxBias
		attendTask.usesSWA = config.layerUsesSWA(l)
		attendTask.groupedGQA = useGroupedGQAAttention && kvMul > 1 && config.NKVHeads > 0 && len(layer.AttnSinks) == 0 && !alibi
		attendTask.alibi = alibi
		parallelChunksTask(p, attendTask)
		*attendTask = batchAttentionTask{}
		batchAttentionTaskPool.Put(attendTask)

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
		activationTask := batchActivationTaskPool.Get().(*batchActivationTask)
		activationTask.gate, activationTask.up, activationTask.hidden = Gate, Up, Hidden
		activationTask.hiddenDim = hDim
		activationTask.plainMLP, activationTask.exactGELU, activationTask.useGELU = config.usesPlainMLP(), config.UseExactGELU, config.UseGELU
		parallelChunksTask(p, activationTask)
		*activationTask = batchActivationTask{}
		batchActivationTaskPool.Put(activationTask)
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
