package gopherllm

import "math"

// f16 KV cache: on hardware with fast half<->single conversion the KV cache
// stores rows as IEEE 754 half floats, halving both the cache's memory
// footprint and — more importantly for decode speed — the bytes streamed
// through attention per generated token, which grow linearly with context.
// The activation math itself stays f32; only the cached K/V rows are
// rounded, the same tradeoff llama.cpp makes by default. Portable scalar
// fallbacks below keep every platform correct; the amd64 F16C kernels make
// it fast, and useF16KVCache gates the feature to platforms where it wins.

// F32ToF16 converts a float32 to IEEE 754 half-precision bits with
// round-to-nearest-even, the scalar counterpart of F16ToF32.
func F32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23&0xff) - 127 + 15
	mant := bits & 0x7fffff
	switch {
	case exp >= 0x1f: // overflow or inf/NaN
		if bits&0x7fffffff > 0x7f800000 { // NaN: keep a mantissa bit set
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp <= 0: // subnormal or zero
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		rem := mant & (1<<shift - 1)
		mid := uint32(1) << (shift - 1)
		if rem > mid || (rem == mid && half&1 == 1) {
			half++
		}
		return sign | half
	default:
		half := sign | uint16(exp)<<10 | uint16(mant>>13)
		rem := mant & 0x1fff
		if rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
			half++ // carry may roll into the exponent; that is correct rounding
		}
		return half
	}
}

func dotF32F16Scalar(a []float32, b []uint16, start int) float32 {
	n := min(len(a), len(b))
	var sum float32
	for i := start; i < n; i++ {
		sum += a[i] * F16ToF32(b[i])
	}
	return sum
}

func axpyF16Scalar(out []float32, alpha float32, x []uint16, start int) {
	n := min(len(out), len(x))
	for i := start; i < n; i++ {
		out[i] += alpha * F16ToF32(x[i])
	}
}

func scaleAddF16Scalar(out []float32, alpha float32, x []uint16, start int) {
	n := min(len(out), len(x))
	for i := start; i < n; i++ {
		out[i] = out[i]*alpha + F16ToF32(x[i])
	}
}

func f32ToF16RowScalar(dst []uint16, src []float32, start int) {
	n := min(len(dst), len(src))
	for i := start; i < n; i++ {
		dst[i] = F32ToF16(src[i])
	}
}

// onlineAttentionF16 is onlineAttention over f16-stored K/V rows: the same
// two-pass structure (see onlineAttention's doc comment), with the cached
// rows converted to f32 inside the fused kernels rather than materialized.
func onlineAttentionF16(query []float32, keys, values []uint16, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
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
		scores[n] = dotF32F16(query, keys[kOff:kOff+keyHeadDim]) * scale
		n++
	}
	weightedVSumEither(scores[:n], nil, values, valueStride, valueHeadDim, startT, softcap, out)
	attnScoresPool.Put(scratch)
}
