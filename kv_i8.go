package gopherllm

import "os"

// int8 (Q8_0-block) KV cache: an optional third, more aggressive storage
// tier alongside f32 and f16 (see kv_f16.go). K/V rows are stored using
// QuantizeRowQ8_0's exact byte layout (quantize_rtn.go) — 34-byte blocks,
// each a 2-byte f16 scale plus 32 signed int8 values — reusing that
// already-tested encoder/decoder pair rather than inventing a new format.
//
// This is a memory-capacity feature first: Q8_0 is only ~1.88x smaller than
// f16 (vs. f16's clean 2x over f32), so unlike f16-over-f32 there is no
// context length at which int8-over-f16 is an unambiguous default; it stays
// opt-in everywhere via GOPHERLLM_KV_I8, off by default, and is not part of
// --auto's search in this version.
//
// Scalar only for now — no AVX2/NEON kernels — so this file carries no build
// tag and is identical on every platform, unlike kv_f16's three per-platform
// variants (which exist only because f16 conversion has genuine SIMD paths).

// q8RowBytes converts an element count to the number of whole Q8_0 blocks'
// worth of bytes it occupies (34 bytes per 32 elements). Any remainder
// smaller than one block is dropped, matching QuantizeRowQ8_0/DotQ8_0F32's
// own block-count derivation (cols/32) — callers that care about a remainder
// must reject it before calling into this format at all (see kvI8Eligible);
// nothing downstream of that gate should ever see a non-block-aligned
// dimension.
func q8RowBytes(n int) int { return (n / 32) * 34 }

// q8RowElems is q8RowBytes's inverse: how many whole-block elements fit in a
// given byte capacity.
func q8RowElems(byteLen int) int { return (byteLen / 34) * 32 }

// kvI8Eligible reports whether every dimension the int8 KV cache touches is a
// multiple of 32 (one Q8_0 block). K/V rows are sliced per KV head before any
// dot/axpy runs (attendHeadWithSink/attendHeadGroup); a byte-packed Q8_0 row
// is only addressable at a block boundary, so if headDim/valueDim aren't
// themselves block-aligned, a per-head byte offset lands mid-block and there
// is no correct fixed-stride formula for it — silently reading a slice of one
// head's data as another's scale (or vice versa). This must be checked before
// ever selecting the int8 format; NewKVCacheI8 itself trusts its caller.
func kvI8Eligible(kDim, vDim, headDim, valueDim int) bool {
	return kDim > 0 && vDim > 0 && headDim > 0 && valueDim > 0 &&
		kDim%32 == 0 && vDim%32 == 0 && headDim%32 == 0 && valueDim%32 == 0
}

// f32ToQ8Row quantizes src into dst using QuantizeRowQ8_0's exact block
// format, writing only as many whole 32-element blocks as both src and dst's
// block capacity support — mirroring f32ToF16Row's own len-based truncation
// behavior rather than panicking on a short slice.
func f32ToQ8Row(dst []byte, src []float32) {
	n := len(src)
	if cap := q8RowElems(len(dst)); cap < n {
		n = cap
	}
	if n <= 0 {
		return
	}
	copy(dst, QuantizeRowQ8_0(src, n))
}

// axpyQ8Row adds alpha * dequant(row) to out. row must hold at least
// len(out)/32 whole 34-byte Q8_0 blocks (2-byte f16 scale + 32 signed int8
// values each), the same layout QuantizeRowQ8_0/DotQ8_0F32 use. Trailing
// bytes beyond that, or a trailing remainder of out shorter than one block,
// are left untouched — mirroring DotQ8_0F32's own block-count derivation
// (len(out)/32), not a new truncation rule.
func axpyQ8Row(out []float32, alpha float32, row []byte) {
	blocks := len(out) / 32
	for b := 0; b < blocks; b++ {
		base := b * 34
		if base+34 > len(row) {
			return
		}
		scale := alpha * F16ToF32(binaryLE16(row[base:]))
		rBlock := row[base+2 : base+34]
		oBlock := out[b*32 : b*32+32]
		_ = rBlock[31]
		_ = oBlock[31]
		for i := 0; i < 32; i++ {
			oBlock[i] += scale * float32(int8(rBlock[i]))
		}
	}
}

// useI8KVCache gates the int8 KV cache tier. Unlike useF16KVCache, this has
// one definition for every platform (no per-platform SIMD path exists yet)
// and defaults off everywhere: there is no context regime where int8-over-f16
// is an unambiguous win the way f16-over-f32 already measured to be on
// amd64, so this stays an explicit opt-in rather than a measured default.
var useI8KVCache = newAtomicBool(os.Getenv("GOPHERLLM_KV_I8") == "1")

// kvI8Available reports whether the int8 KV cache is implemented on this
// build. Always true — there is no hardware dependency to gate on (matches
// q8ActivationsAvailable's "available means correct and selectable"
// convention); eligibility per-model is a separate, dimension-based check
// (kvI8Eligible), not a platform capability.
func kvI8Available() bool { return true }

func kvI8Enabled() bool { return useI8KVCache.Load() }

func setKVI8(on bool) { useI8KVCache.Store(on) }

// onlineAttentionI8WithSink is the int8-KV-cache counterpart of
// onlineAttentionF16WithSink. keys8/values8 must already be sliced to the
// correct KV head's byte offset (q8RowBytes(kvH*keyHeadDim), see
// attendHeadWithSink) — keyStride/valueStride stay in element units exactly
// as for f32/f16 and are converted to Q8_0 byte strides internally.
func onlineAttentionI8WithSink(query []float32, keys8, values8 []byte, keyStride, valueStride, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap, alibiSlope, sink float32, hasSink bool, out []float32) {
	span := endT - startT + 1
	if span <= 0 {
		return
	}
	scratch := attnScoresPool.Get().(*[]float32)
	ensureLenNoClear(scratch, span)
	scores := (*scratch)[:span]

	keyByteStride := q8RowBytes(keyStride)
	keyBlockBytes := q8RowBytes(keyHeadDim)

	n := 0
	for t := startT; t <= endT; t++ {
		kOff8 := t * keyByteStride
		if kOff8+keyBlockBytes > len(keys8) {
			break
		}
		scores[n] = DotQ8_0F32(keys8[kOff8:kOff8+keyBlockBytes], query, keyHeadDim)*scale + alibiSlope*float32(t-endT)
		n++
	}
	weightedVSumEitherWithSink(scores[:n], nil, nil, values8, valueStride, valueHeadDim, startT, softcap, sink, hasSink, out)
	attnScoresPool.Put(scratch)
}
