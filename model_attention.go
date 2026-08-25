package gopherllm

import (
	"sync"
)

// attnScoresPool holds per-head score scratch for the two-pass attention
// below. Heads run concurrently (parallelChunks), so each in-flight head
// borrows its own buffer.
var attnScoresPool = sync.Pool{New: func() any { s := make([]float32, 0, 4096); return &s }}

// onlineAttention computes softmax(q·K/scale)·V for one head over positions
// startT..endT, accumulating into out (which the caller has zeroed).
//
// Two-pass structure: pass 1 computes every score as an independent dot
// product — nothing but loads and FMAs in the dependency chain, so
// out-of-order execution overlaps positions freely; pass 2 takes the exact
// global max, exponentiates (iterations independent, so the exp latency
// pipelines too), and accumulates the weighted V rows. The previous
// single-pass online-softmax rescaled the accumulator inside the loop,
// chaining dot -> exp -> branch -> rescale serially per position; measured
// on the dev laptop the two-pass form is ~1.15x faster at 4k-16k context
// and numerically it uses the true maximum rather than a running one.
func onlineAttention(query, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	onlineAttentionWithSink(query, keys, values, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, 0, 0, false, out)
}

func onlineAttentionWithSink(query, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap, alibiSlope, sink float32, hasSink bool, out []float32) {
	span := endT - startT + 1
	if span <= 0 {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	ensureLenNoClear(scratch, span)
	scores := (*scratch)[:span]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if kOff+keyHeadDim > len(keys) {
			break
		}
		scores[n] = DotF32(query, keys[kOff:kOff+keyHeadDim])*scale + alibiSlope*float32(t-endT)
		n++
	}
	weightedVSumWithSink(scores[:n], values, valueStride, valueHeadDim, startT, softcap, sink, hasSink, out)
	attnScoresPool.Put(scratch)
}

// onlineAttentionGroup is the IO-aware GQA/MQA path. Query heads sharing one
// KV head are kept together through both attention passes. The arithmetic for
// each head is unchanged, but K/V rows remain hot across the group instead of
// being streamed independently for every query head.
func onlineAttentionGroup(queries, keys, values []float32, queryHeads, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	if queryHeads == 4 {
		onlineAttentionGroup4(queries, keys, values, keyStride, valueStride,
			keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
		return
	}
	if queryHeads == 6 {
		// Qwen3.8's 24 query heads and four KV heads form six-way groups.
		// Process the first four through the shared-row SIMD primitive and
		// retain two ordinary heads, keeping every K/V row live across all six.
		onlineAttentionGroup6(queries, keys, values, keyStride, valueStride,
			keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
		return
	}
	onlineAttentionGroupEither(queries, keys, nil, nil, values, nil, nil, queryHeads,
		keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
}

// onlineAttentionGroup4 is the Ministral-style GQA specialization. Its NEON
// primitives load each shared K/V row once for four query heads.
func onlineAttentionGroup4(queries, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	const queryHeads = 4
	span := endT - startT + 1
	if span <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || len(queries) < queryHeads*keyHeadDim || len(out) < queryHeads*valueHeadDim {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	scoreLen := queryHeads * span
	ensureLenNoClear(scratch, scoreLen+queryHeads)
	scores := (*scratch)[:scoreLen]
	denoms := (*scratch)[scoreLen : scoreLen+queryHeads]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if kOff+keyHeadDim > len(keys) {
			break
		}
		s0, s1, s2, s3 := dotF32x4(
			&queries[0], &queries[keyHeadDim], &queries[2*keyHeadDim], &queries[3*keyHeadDim],
			&keys[kOff], keyHeadDim)
		scores[n] = s0 * scale
		scores[span+n] = s1 * scale
		scores[2*span+n] = s2 * scale
		scores[3*span+n] = s3 * scale
		n++
	}
	if n == 0 {
		attnScoresPool.Put(scratch)
		return
	}
	for h := 0; h < queryHeads; h++ {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}

	out0 := out[:valueHeadDim]
	out1 := out[valueHeadDim : 2*valueHeadDim]
	out2 := out[2*valueHeadDim : 3*valueHeadDim]
	out3 := out[3*valueHeadDim : 4*valueHeadDim]
	for i := 0; i < n; i++ {
		vOff := (startT + i) * valueStride
		if vOff+valueHeadDim > len(values) {
			break
		}
		axpyF32x4(&out0[0], &out1[0], &out2[0], &out3[0],
			scores[i], scores[span+i], scores[2*span+i], scores[3*span+i],
			&values[vOff], valueHeadDim)
	}
	ScaleF32(out0, 1/denoms[0])
	ScaleF32(out1, 1/denoms[1])
	ScaleF32(out2, 1/denoms[2])
	ScaleF32(out3, 1/denoms[3])
	attnScoresPool.Put(scratch)
}

// onlineAttentionGroup6 is Qwen3.8's six-query-head GQA specialization. Four
// heads use the existing shared-row SIMD instructions while the remaining two
// finish against the same cacheline. This avoids falling back to six unrelated
// dot/AXPY calls for Qwen's 24:4 attention layout.
func onlineAttentionGroup6(queries, keys, values []float32, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	const queryHeads = 6
	span := endT - startT + 1
	if span <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || len(queries) < queryHeads*keyHeadDim || len(out) < queryHeads*valueHeadDim {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	scoreLen := queryHeads * span
	ensureLenNoClear(scratch, scoreLen+queryHeads)
	scores := (*scratch)[:scoreLen]
	denoms := (*scratch)[scoreLen : scoreLen+queryHeads]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if kOff+keyHeadDim > len(keys) {
			break
		}
		s0, s1, s2, s3 := dotF32x4(
			&queries[0], &queries[keyHeadDim], &queries[2*keyHeadDim], &queries[3*keyHeadDim],
			&keys[kOff], keyHeadDim)
		scores[n] = s0 * scale
		scores[span+n] = s1 * scale
		scores[2*span+n] = s2 * scale
		scores[3*span+n] = s3 * scale
		scores[4*span+n] = DotF32(queries[4*keyHeadDim:5*keyHeadDim], keys[kOff:kOff+keyHeadDim]) * scale
		scores[5*span+n] = DotF32(queries[5*keyHeadDim:6*keyHeadDim], keys[kOff:kOff+keyHeadDim]) * scale
		n++
	}
	if n == 0 {
		attnScoresPool.Put(scratch)
		return
	}
	for h := 0; h < queryHeads; h++ {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}

	out0 := out[:valueHeadDim]
	out1 := out[valueHeadDim : 2*valueHeadDim]
	out2 := out[2*valueHeadDim : 3*valueHeadDim]
	out3 := out[3*valueHeadDim : 4*valueHeadDim]
	out4 := out[4*valueHeadDim : 5*valueHeadDim]
	out5 := out[5*valueHeadDim : 6*valueHeadDim]
	for i := 0; i < n; i++ {
		vOff := (startT + i) * valueStride
		if vOff+valueHeadDim > len(values) {
			break
		}
		value := values[vOff : vOff+valueHeadDim]
		axpyF32x4(&out0[0], &out1[0], &out2[0], &out3[0],
			scores[i], scores[span+i], scores[2*span+i], scores[3*span+i],
			&value[0], valueHeadDim)
		AxpyF32(out4, scores[4*span+i], value)
		AxpyF32(out5, scores[5*span+i], value)
	}
	for h := 0; h < queryHeads; h++ {
		ScaleF32(out[h*valueHeadDim:(h+1)*valueHeadDim], 1/denoms[h])
	}
	attnScoresPool.Put(scratch)
}

func onlineAttentionGroupF16(queries []float32, keys, values []uint16, queryHeads, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	if queryHeads == 4 {
		onlineAttentionGroup4F16(queries, keys, values, keyStride, valueStride,
			keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
		return
	}
	if queryHeads == 6 {
		onlineAttentionGroup6F16(queries, keys, values, keyStride, valueStride,
			keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
		return
	}
	onlineAttentionGroupEither(queries, nil, keys, nil, nil, values, nil, queryHeads,
		keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
}

// onlineAttentionGroup4F16 is the compact-KV equivalent of
// onlineAttentionGroup4. Mistral/Ministral/Devstral use four query heads per
// KV head, so its ARM64 NEON primitives decode each shared half-precision K/V
// row once instead of once per query head.
func onlineAttentionGroup4F16(queries []float32, keys, values []uint16, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	const queryHeads = 4
	span := endT - startT + 1
	if span <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || len(queries) < queryHeads*keyHeadDim || len(out) < queryHeads*valueHeadDim {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	scoreLen := queryHeads * span
	ensureLenNoClear(scratch, scoreLen+queryHeads)
	scores := (*scratch)[:scoreLen]
	denoms := (*scratch)[scoreLen : scoreLen+queryHeads]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if kOff+keyHeadDim > len(keys) {
			break
		}
		s0, s1, s2, s3 := dotF32F16x4(
			&queries[0], &queries[keyHeadDim], &queries[2*keyHeadDim], &queries[3*keyHeadDim],
			&keys[kOff], keyHeadDim)
		scores[n] = s0 * scale
		scores[span+n] = s1 * scale
		scores[2*span+n] = s2 * scale
		scores[3*span+n] = s3 * scale
		n++
	}
	if n == 0 {
		attnScoresPool.Put(scratch)
		return
	}
	for h := 0; h < queryHeads; h++ {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}

	out0 := out[:valueHeadDim]
	out1 := out[valueHeadDim : 2*valueHeadDim]
	out2 := out[2*valueHeadDim : 3*valueHeadDim]
	out3 := out[3*valueHeadDim : 4*valueHeadDim]
	for i := 0; i < n; i++ {
		vOff := (startT + i) * valueStride
		if vOff+valueHeadDim > len(values) {
			break
		}
		axpyF32F16x4(&out0[0], &out1[0], &out2[0], &out3[0],
			scores[i], scores[span+i], scores[2*span+i], scores[3*span+i],
			&values[vOff], valueHeadDim)
	}
	ScaleF32(out0, 1/denoms[0])
	ScaleF32(out1, 1/denoms[1])
	ScaleF32(out2, 1/denoms[2])
	ScaleF32(out3, 1/denoms[3])
	attnScoresPool.Put(scratch)
}

// onlineAttentionGroup6F16 is the compact-KV counterpart to
// onlineAttentionGroup6. Qwen3.8's six-head groups take the four-wide NEON
// fast path for most heads and reuse the already-hot f16 K/V row for the last
// two, instead of converting it independently six times.
func onlineAttentionGroup6F16(queries []float32, keys, values []uint16, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	const queryHeads = 6
	span := endT - startT + 1
	if span <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || len(queries) < queryHeads*keyHeadDim || len(out) < queryHeads*valueHeadDim {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	scoreLen := queryHeads * span
	ensureLenNoClear(scratch, scoreLen+queryHeads)
	scores := (*scratch)[:scoreLen]
	denoms := (*scratch)[scoreLen : scoreLen+queryHeads]

	n := 0
	for t := startT; t <= endT; t++ {
		kOff := t * keyStride
		if kOff+keyHeadDim > len(keys) {
			break
		}
		s0, s1, s2, s3 := dotF32F16x4(
			&queries[0], &queries[keyHeadDim], &queries[2*keyHeadDim], &queries[3*keyHeadDim],
			&keys[kOff], keyHeadDim)
		scores[n] = s0 * scale
		scores[span+n] = s1 * scale
		scores[2*span+n] = s2 * scale
		scores[3*span+n] = s3 * scale
		scores[4*span+n] = dotF32F16(queries[4*keyHeadDim:5*keyHeadDim], keys[kOff:kOff+keyHeadDim]) * scale
		scores[5*span+n] = dotF32F16(queries[5*keyHeadDim:6*keyHeadDim], keys[kOff:kOff+keyHeadDim]) * scale
		n++
	}
	if n == 0 {
		attnScoresPool.Put(scratch)
		return
	}
	for h := 0; h < queryHeads; h++ {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}

	out0 := out[:valueHeadDim]
	out1 := out[valueHeadDim : 2*valueHeadDim]
	out2 := out[2*valueHeadDim : 3*valueHeadDim]
	out3 := out[3*valueHeadDim : 4*valueHeadDim]
	out4 := out[4*valueHeadDim : 5*valueHeadDim]
	out5 := out[5*valueHeadDim : 6*valueHeadDim]
	for i := 0; i < n; i++ {
		vOff := (startT + i) * valueStride
		if vOff+valueHeadDim > len(values) {
			break
		}
		value := values[vOff : vOff+valueHeadDim]
		axpyF32F16x4(&out0[0], &out1[0], &out2[0], &out3[0],
			scores[i], scores[span+i], scores[2*span+i], scores[3*span+i],
			&value[0], valueHeadDim)
		axpyF16(out4, scores[4*span+i], value)
		axpyF16(out5, scores[5*span+i], value)
	}
	for h := 0; h < queryHeads; h++ {
		ScaleF32(out[h*valueHeadDim:(h+1)*valueHeadDim], 1/denoms[h])
	}
	attnScoresPool.Put(scratch)
}

// onlineAttentionGroupI8 is the Q8_0-KV-cache counterpart of
// onlineAttentionGroupF16. keys8/values8 must already be sliced to the
// correct KV head's byte offset (q8RowBytes(kvH*keyHeadDim), see
// attendHeadGroup) — keyStride/valueStride stay in element units exactly as
// for f32/f16, and are converted to Q8_0 byte strides internally.
func onlineAttentionGroupI8(queries []float32, keys8, values8 []byte, queryHeads, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	onlineAttentionGroupEither(queries, nil, nil, keys8, nil, nil, values8, queryHeads,
		keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, out)
}

func onlineAttentionGroupEither(queries []float32, keys []float32, keys16 []uint16, keys8 []byte, values []float32, values16 []uint16, values8 []byte, queryHeads, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	span := endT - startT + 1
	if span <= 0 || queryHeads <= 0 || len(queries) < queryHeads*keyHeadDim || len(out) < queryHeads*valueHeadDim {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	scoreLen := queryHeads * span
	ensureLenNoClear(scratch, scoreLen+queryHeads)
	scores := (*scratch)[:scoreLen]
	denoms := (*scratch)[scoreLen : scoreLen+queryHeads]

	// Only meaningful (and only computed) when keys8/values8 are the active
	// format — see q8RowBytes's doc comment for why byte and element offsets
	// are not interchangeable for Q8_0-packed rows.
	keyByteStride := q8RowBytes(keyStride)
	keyBlockBytes := q8RowBytes(keyHeadDim)

	n := 0
scorePass:
	for t := startT; t <= endT; t++ {
		switch {
		case keys != nil:
			kOff := t * keyStride
			if kOff+keyHeadDim > len(keys) {
				break scorePass
			}
			for h := 0; h < queryHeads; h++ {
				query := queries[h*keyHeadDim : (h+1)*keyHeadDim]
				scores[h*span+n] = DotF32(query, keys[kOff:kOff+keyHeadDim]) * scale
			}
		case keys16 != nil:
			kOff := t * keyStride
			if kOff+keyHeadDim > len(keys16) {
				break scorePass
			}
			for h := 0; h < queryHeads; h++ {
				query := queries[h*keyHeadDim : (h+1)*keyHeadDim]
				scores[h*span+n] = dotF32F16(query, keys16[kOff:kOff+keyHeadDim]) * scale
			}
		default:
			kOff8 := t * keyByteStride
			if kOff8+keyBlockBytes > len(keys8) {
				break scorePass
			}
			for h := 0; h < queryHeads; h++ {
				query := queries[h*keyHeadDim : (h+1)*keyHeadDim]
				scores[h*span+n] = DotQ8_0F32(keys8[kOff8:kOff8+keyBlockBytes], query, keyHeadDim) * scale
			}
		}
		n++
	}
	if n == 0 {
		attnScoresPool.Put(scratch)
		return
	}

	for h := 0; h < queryHeads; h++ {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}

	valueByteStride := q8RowBytes(valueStride)
	valueBlockBytes := q8RowBytes(valueHeadDim)

valuePass:
	for i := 0; i < n; i++ {
		switch {
		case values != nil:
			vOff := (startT + i) * valueStride
			if vOff+valueHeadDim > len(values) {
				break valuePass
			}
			for h := 0; h < queryHeads; h++ {
				AxpyF32(out[h*valueHeadDim:(h+1)*valueHeadDim], scores[h*span+i], values[vOff:vOff+valueHeadDim])
			}
		case values16 != nil:
			vOff := (startT + i) * valueStride
			if vOff+valueHeadDim > len(values16) {
				break valuePass
			}
			for h := 0; h < queryHeads; h++ {
				axpyF16(out[h*valueHeadDim:(h+1)*valueHeadDim], scores[h*span+i], values16[vOff:vOff+valueHeadDim])
			}
		default:
			vOff8 := (startT + i) * valueByteStride
			if vOff8+valueBlockBytes > len(values8) {
				break valuePass
			}
			for h := 0; h < queryHeads; h++ {
				axpyQ8Row(out[h*valueHeadDim:(h+1)*valueHeadDim], scores[h*span+i], values8[vOff8:vOff8+valueBlockBytes])
			}
		}
	}
	for h := 0; h < queryHeads; h++ {
		if denoms[h] > 0 {
			ScaleF32(out[h*valueHeadDim:(h+1)*valueHeadDim], 1/denoms[h])
		}
	}
	attnScoresPool.Put(scratch)
}

func attentionWeightsInPlace(scores []float32, softcap float32) float32 {
	if softcap > 0 {
		for i, s := range scores {
			// Attention runs this once per cached position and head on every
			// decode token. Keep it in f32 just like the activation and final
			// logit-softcap paths rather than widening every score into libm.
			scores[i] = softcap * fastTanhF32(s/softcap)
		}
	}
	maxScore := scores[0]
	for _, s := range scores[1:] {
		if s > maxScore {
			maxScore = s
		}
	}
	var denom float32
	for i, s := range scores {
		w := fastExpF32(s - maxScore)
		scores[i] = w
		denom += w
	}
	return denom
}

// weightedVSum finishes attention pass 2 shared by the f32, f16, and int8
// K-row variants: optional softcap, max-stabilized softmax weights in place,
// then out += sum(w_i * V_row_i) / denom. values16/values8 are used when
// values is nil.
func weightedVSum(scores []float32, values []float32, valueStride, valueHeadDim, startT int, softcap float32, out []float32) {
	weightedVSumWithSink(scores, values, valueStride, valueHeadDim, startT, softcap, 0, false, out)
}

func weightedVSumWithSink(scores []float32, values []float32, valueStride, valueHeadDim, startT int, softcap, sink float32, hasSink bool, out []float32) {
	weightedVSumEitherWithSink(scores, values, nil, nil, valueStride, valueHeadDim, startT, softcap, sink, hasSink, out)
}

func weightedVSumEither(scores []float32, values []float32, values16 []uint16, valueStride, valueHeadDim, startT int, softcap float32, out []float32) {
	weightedVSumEitherWithSink(scores, values, values16, nil, valueStride, valueHeadDim, startT, softcap, 0, false, out)
}

func weightedVSumEitherWithSink(scores []float32, values []float32, values16 []uint16, values8 []byte, valueStride, valueHeadDim, startT int, softcap, sink float32, hasSink bool, out []float32) {
	n := len(scores)
	if n == 0 {
		return
	}
	if softcap > 0 {
		for i, s := range scores {
			scores[i] = softcap * fastTanhF32(s/softcap)
		}
	}
	maxScore := scores[0]
	for _, s := range scores[1:] {
		if s > maxScore {
			maxScore = s
		}
	}
	if hasSink && sink > maxScore {
		maxScore = sink
	}
	var denom float32
	for i, s := range scores {
		w := fastExpF32(s - maxScore)
		scores[i] = w
		denom += w
	}
	if hasSink {
		denom += fastExpF32(sink - maxScore)
	}
	valueByteStride := q8RowBytes(valueStride)
	valueBlockBytes := q8RowBytes(valueHeadDim)
	for i := 0; i < n; i++ {
		if values != nil {
			vOff := (startT + i) * valueStride
			if vOff+valueHeadDim > len(values) {
				break
			}
			AxpyF32(out[:valueHeadDim], scores[i], values[vOff:vOff+valueHeadDim])
		} else if values16 != nil {
			vOff := (startT + i) * valueStride
			if vOff+valueHeadDim > len(values16) {
				break
			}
			axpyF16(out[:valueHeadDim], scores[i], values16[vOff:vOff+valueHeadDim])
		} else {
			vOff8 := (startT + i) * valueByteStride
			if vOff8+valueBlockBytes > len(values8) {
				break
			}
			axpyQ8Row(out[:valueHeadDim], scores[i], values8[vOff8:vOff8+valueBlockBytes])
		}
	}
	if denom > 0 {
		ScaleF32(out[:valueHeadDim], 1/denom)
	}
}
