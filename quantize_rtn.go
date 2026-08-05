package gopherllm

// Round-to-nearest weight quantization: the inverse of the mainline
// DequantRow* decoders in simd.go. Every function here takes one row's
// float32 values and produces exactly DType.DataSize(len(row)) bytes,
// matching DequantRowXInto's own per-row calling convention (not the
// whole-tensor dequantTensor convention) — GPTQ needs per-row processing
// anyway, since every row's quantization residual is independent even
// though rows sharing a weight matrix share one calibration Hessian.
//
// These are plain nearest-fit encoders: no calibration data, no iterative
// error search (unlike llama.cpp's make_qkx2_quants for Q4_K/Q6_K, which
// additionally searches candidate scales to minimize weighted squared
// error). They produce a valid, self-consistent GGUF whose values
// approximate the input — the GPTQ/AWQ/SmoothQuant layers built on top of
// this file replace "which value gets stored" with something smarter;
// this file only owns "how a chosen value gets packed into the block
// format," which every one of those methods still needs.

import "math"

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// QuantizeRowQ8_0 encodes cols float32 values as Q8_0: one f16 scale plus
// 32 signed int8 values per 32-element block (simd.go:1466-1485). Symmetric,
// no zero-point.
func QuantizeRowQ8_0(row []float32, cols int) []byte {
	nBlocks := cols / 32
	out := make([]byte, nBlocks*34)
	for b := 0; b < nBlocks; b++ {
		x := row[b*32 : b*32+32]
		var amax float32
		for _, v := range x {
			if a := abs32(v); a > amax {
				amax = a
			}
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		base := b * 34
		putF16(out[base:], d)
		for i := 0; i < 32; i++ {
			q := clampInt(int(roundHalfAwayFromZero(x[i]*id)), -127, 127)
			out[base+2+i] = byte(int8(q))
		}
	}
	return out
}

// QuantizeRowQ4_0 encodes cols float32 values as Q4_0: one f16 scale plus
// 16 nibble-packed bytes per 32-element block, decoded value = scale *
// (nibble-8) (simd.go:1487-1508).
//
// nibble-8 is an ASYMMETRIC range (-8..+7, not -8..+8: a 4-bit unsigned
// code only has 16 values total). Scaling by plain amax/8 therefore clips
// the block's extreme value whenever it's positive (its ideal code, +8,
// doesn't exist). The fix — matching llama.cpp's reference
// quantize_row_q4_0 — is to let the scale take the SIGN of whichever
// signed extreme has the largest magnitude, not just its absolute value:
// d = extremeSigned/-8. That maps the extreme to code 0 exactly regardless
// of its sign (code 0 decodes to d*(0-8) = -8d = extremeSigned either way),
// fully using the available range instead of silently clipping one side.
func QuantizeRowQ4_0(row []float32, cols int) []byte {
	nBlocks := cols / 32
	out := make([]byte, nBlocks*18)
	for b := 0; b < nBlocks; b++ {
		x := row[b*32 : b*32+32]
		var amax, extreme float32
		for _, v := range x {
			if a := abs32(v); a > amax {
				amax = a
				extreme = v
			}
		}
		d := extreme / -8
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		base := b * 18
		putF16(out[base:], d)
		for i := 0; i < 16; i++ {
			lo := clampInt(int(roundHalfAwayFromZero(x[i]*id))+8, 0, 15)
			hi := clampInt(int(roundHalfAwayFromZero(x[16+i]*id))+8, 0, 15)
			out[base+2+i] = byte(lo) | byte(hi)<<4
		}
	}
	return out
}

// packScaleMinK4All is the inverse of getScaleMinK4 (simd.go:1658-1663):
// given all 8 sub-block 6-bit scale and min codes, produce the 12-byte
// packed table Q4_K/Q5_K store them in. See quantize_rtn_test.go for the
// exhaustive round-trip check against the real decoder this mirrors.
func packScaleMinK4All(sc, m [8]byte) [12]byte {
	var q [12]byte
	for i := 0; i < 4; i++ {
		q[i] = (sc[i] & 0x3f) | ((sc[i+4]>>4)&0x03)<<6
		q[i+4] = (m[i] & 0x3f) | ((m[i+4]>>4)&0x03)<<6
		q[i+8] = (sc[i+4] & 0x0f) | (m[i+4]&0x0f)<<4
	}
	return q
}

// QuantizeRowQ4K encodes cols float32 values as Q4_K: 144-byte superblocks
// of 256 elements (f16 d + f16 dmin + 12B packed 6-bit scale/min table + 128
// nibble-packed bytes), decoded value = d*sc*nibble - dmin*m (simd.go:1510-1543).
//
// Two passes per superblock, not one: the 12-byte scale/min table packs all
// 8 sub-blocks' 6-bit codes jointly (packScaleMinK4All above), so no
// sub-block's stored scale is knowable until all 8 have been computed. Pass
// 1 computes and freezes those codes; pass 2 rounds nibbles against the
// scales the decoder will actually reconstruct from them, not the ideal
// float scale — otherwise a caller (GPTQ) computing an error term from this
// function's output would compute it against a scale the file doesn't
// actually store.
func QuantizeRowQ4K(row []float32, cols int) []byte {
	nSuper := cols / 256
	out := make([]byte, nSuper*144)
	for s := 0; s < nSuper; s++ {
		super := row[s*256 : s*256+256]

		// Pass 1: per-sub-block raw (scale, min1) where decoded value ==
		// rawScale*nibble - rawMin1, nibble in [0,15].
		var rawScale, rawMin1 [8]float32
		for i := 0; i < 8; i++ {
			sub := super[i*32 : i*32+32]
			lo, hi := sub[0], sub[0]
			for _, v := range sub {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			sc := float32(0)
			if hi > lo {
				sc = (hi - lo) / 15
			}
			rawScale[i] = sc
			rawMin1[i] = -lo
		}
		maxScale, maxMin1 := rawScale[0], rawMin1[0]
		for i := 1; i < 8; i++ {
			if rawScale[i] > maxScale {
				maxScale = rawScale[i]
			}
			if rawMin1[i] > maxMin1 {
				maxMin1 = rawMin1[i]
			}
		}
		d := maxScale / 63
		dmin := maxMin1 / 63
		invD, invDmin := float32(0), float32(0)
		if d != 0 {
			invD = 1 / d
		}
		if dmin != 0 {
			invDmin = 1 / dmin
		}
		var sc, m [8]byte
		for i := 0; i < 8; i++ {
			sc[i] = byte(clampInt(int(roundHalfAwayFromZero(rawScale[i]*invD)), 0, 63))
			m[i] = byte(clampInt(int(roundHalfAwayFromZero(rawMin1[i]*invDmin)), 0, 63))
		}
		packed := packScaleMinK4All(sc, m)

		base := s * 144
		putF16(out[base:], d)
		putF16(out[base+2:], dmin)
		copy(out[base+4:base+16], packed[:])

		// Pass 2: round against the scales the decoder will actually use
		// (d*sc[i], dmin*m[i] — the quantized codes from pass 1), then pack
		// nibbles in the same low/high-nibble-pair-of-sub-blocks layout
		// DequantRowQ4KInto reads (simd.go:1516-1542): q advances 32 bytes
		// per 64 output elements, sub-block 2k in the low nibble, 2k+1 in
		// the high nibble.
		q := out[base+16 : base+144]
		for pair := 0; pair < 4; pair++ {
			i0, i1 := pair*2, pair*2+1
			d1 := d * float32(sc[i0])
			min1_ := dmin * float32(m[i0])
			d2 := d * float32(sc[i1])
			min2_ := dmin * float32(m[i1])
			invD1, invD2 := float32(0), float32(0)
			if d1 != 0 {
				invD1 = 1 / d1
			}
			if d2 != 0 {
				invD2 = 1 / d2
			}
			sub0 := super[i0*32 : i0*32+32]
			sub1 := super[i1*32 : i1*32+32]
			qq := q[pair*32 : pair*32+32]
			for l := 0; l < 32; l++ {
				lo := clampInt(int(roundHalfAwayFromZero((sub0[l]+min1_)*invD1)), 0, 15)
				hi := clampInt(int(roundHalfAwayFromZero((sub1[l]+min2_)*invD2)), 0, 15)
				qq[l] = byte(lo) | byte(hi)<<4
			}
		}
	}
	return out
}

// QuantizeRowQ5K encodes cols float32 values as Q5_K: 176-byte superblocks of
// 256 elements (f16 d + f16 dmin + the same 12B packed 6-bit scale/min table
// as Q4_K + 32B high-bit plane + 128B low-nibble plane), decoded value =
// d*sc*(nibble|hibit<<4) - dmin*m, a 5-bit unsigned code (0..31) split into a
// 4-bit low nibble and one bit of the 32-byte high-bit plane
// (simd.go:1545-1593). Shares Q4_K's joint scale/min packing exactly
// (getScaleMinK4/packScaleMinK4All), so the same two-pass approach applies:
// freeze the 8 sub-blocks' 6-bit scale/min codes first, then round 5-bit
// values against the scales the decoder will actually reconstruct.
func QuantizeRowQ5K(row []float32, cols int) []byte {
	nSuper := cols / 256
	out := make([]byte, nSuper*176)
	for s := 0; s < nSuper; s++ {
		super := row[s*256 : s*256+256]

		// Pass 1: per-sub-block raw (scale, min1), 5-bit range (0..31) this
		// time, not Q4_K's 4-bit (0..15).
		var rawScale, rawMin1 [8]float32
		for i := 0; i < 8; i++ {
			sub := super[i*32 : i*32+32]
			lo, hi := sub[0], sub[0]
			for _, v := range sub {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			sc := float32(0)
			if hi > lo {
				sc = (hi - lo) / 31
			}
			rawScale[i] = sc
			rawMin1[i] = -lo
		}
		maxScale, maxMin1 := rawScale[0], rawMin1[0]
		for i := 1; i < 8; i++ {
			if rawScale[i] > maxScale {
				maxScale = rawScale[i]
			}
			if rawMin1[i] > maxMin1 {
				maxMin1 = rawMin1[i]
			}
		}
		d := maxScale / 63
		dmin := maxMin1 / 63
		invD, invDmin := float32(0), float32(0)
		if d != 0 {
			invD = 1 / d
		}
		if dmin != 0 {
			invDmin = 1 / dmin
		}
		var sc, m [8]byte
		for i := 0; i < 8; i++ {
			sc[i] = byte(clampInt(int(roundHalfAwayFromZero(rawScale[i]*invD)), 0, 63))
			m[i] = byte(clampInt(int(roundHalfAwayFromZero(rawMin1[i]*invDmin)), 0, 63))
		}
		packed := packScaleMinK4All(sc, m)

		base := s * 176
		putF16(out[base:], d)
		putF16(out[base+2:], dmin)
		copy(out[base+4:base+16], packed[:])
		qh := out[base+16 : base+48]
		q := out[base+48 : base+176]

		// Pass 2: same sub-block pairing as Q4_K (is=0,2,4,6 across 4
		// iterations), but each 5-bit code splits into q's low/high nibble
		// plus one bit of qh — u1/u2 walk the same two-bit-bits-per-iteration
		// pattern DequantRowQ5KInto reads them back with.
		u1, u2 := byte(1), byte(2)
		for pair := 0; pair < 4; pair++ {
			i0, i1 := pair*2, pair*2+1
			d1 := d * float32(sc[i0])
			min1_ := dmin * float32(m[i0])
			d2 := d * float32(sc[i1])
			min2_ := dmin * float32(m[i1])
			invD1, invD2 := float32(0), float32(0)
			if d1 != 0 {
				invD1 = 1 / d1
			}
			if d2 != 0 {
				invD2 = 1 / d2
			}
			sub0 := super[i0*32 : i0*32+32]
			sub1 := super[i1*32 : i1*32+32]
			qq := q[pair*32 : pair*32+32]
			for l := 0; l < 32; l++ {
				code1 := clampInt(int(roundHalfAwayFromZero((sub0[l]+min1_)*invD1)), 0, 31)
				code2 := clampInt(int(roundHalfAwayFromZero((sub1[l]+min2_)*invD2)), 0, 31)
				qq[l] = byte(code1&0x0f) | byte(code2&0x0f)<<4
				if code1&0x10 != 0 {
					qh[l] |= u1
				}
				if code2&0x10 != 0 {
					qh[l] |= u2
				}
			}
			u1 <<= 2
			u2 <<= 2
		}
	}
	return out
}

// QuantizeRowQ6K encodes cols float32 values as Q6_K: 210-byte superblocks
// of 256 elements (128B low-nibble plane + 64B 2-bit-high plane + 16 signed
// int8 sub-block scales + f16 d at the end), decoded value =
// d*int8(scale)*(code-32), code a 6-bit unsigned value (simd.go:1595-1629).
// Unlike Q4_K, the 16 sub-block scales are independent int8 values with no
// joint packing, so this needs only one pass.
func QuantizeRowQ6K(row []float32, cols int) []byte {
	nSuper := cols / 256
	out := make([]byte, nSuper*210)
	for s := 0; s < nSuper; s++ {
		super := row[s*256 : s*256+256]

		// extremeSub keeps the SIGN of each sub-block's largest-magnitude
		// value, not just its absolute value: the per-element code range
		// (code-32, code in [0,63]) is asymmetric, -32..+31, exactly like
		// Q4_0's nibble-8. Letting the signed int8 sub-block scale take
		// that sign — scale_i = extreme_i/-32 — maps the sub-block's own
		// extreme to code 0 or 63 exactly regardless of which sign it has,
		// instead of always aiming for the +31 end and silently clipping
		// negative extremes by up to a full step (mirrors QuantizeRowQ4_0's
		// fix above).
		var extremeSub [16]float32
		var globalAbsmax float32
		for i := 0; i < 16; i++ {
			sub := super[i*16 : i*16+16]
			var a float32
			for _, v := range sub {
				if m := abs32(v); m > a {
					a = m
					extremeSub[i] = v
				}
			}
			if a > globalAbsmax {
				globalAbsmax = a
			}
		}
		d := globalAbsmax / (32 * 127)
		invD := float32(0)
		if d != 0 {
			invD = 1 / d
		}
		var scInt8 [16]int8
		for i := 0; i < 16; i++ {
			target := extremeSub[i] / -32
			raw := clampInt(int(roundHalfAwayFromZero(target*invD)), -127, 127)
			scInt8[i] = int8(raw)
		}

		base := s * 210
		ql := out[base : base+128]
		qh := out[base+128 : base+192]
		sc := out[base+192 : base+208]
		for i := 0; i < 16; i++ {
			sc[i] = byte(scInt8[i])
		}
		putF16(out[base+208:], d)

		// Inverse of DequantRowQ6KInto's q1..q4 layout (simd.go:1595-1629):
		// two 128-element halves, each covering 4 sub-blocks (is=0..1 per
		// half via l/16), code split into a 4-bit low nibble (ql) and a
		// 2-bit high field (qh) per one of four bit-shift slots. The
		// decoder reslices its local `sc` by 8 elements between halves
		// (`sc = sc[8:]` in DequantRowQ6KInto), so half 1's sub-block
		// indices are offset by +8 here to match — without it, half 1
		// silently reused half 0's sub-block scales.
		for half := 0; half < 2; half++ {
			n := half * 128
			scOff := half * 8
			qlh := ql[half*64 : half*64+64]
			qhh := qh[half*32 : half*32+32]
			for l := 0; l < 32; l++ {
				is := l/16 + scOff
				scale1 := d * float32(scInt8[is])
				scale2 := d * float32(scInt8[is+2])
				scale3 := d * float32(scInt8[is+4])
				scale4 := d * float32(scInt8[is+6])
				code1 := quantizeQ6KCode(super[n+l], scale1)
				code2 := quantizeQ6KCode(super[n+32+l], scale2)
				code3 := quantizeQ6KCode(super[n+64+l], scale3)
				code4 := quantizeQ6KCode(super[n+96+l], scale4)
				qlh[l] |= code1 & 0x0f
				qlh[l+32] |= code2 & 0x0f
				qlh[l] |= (code3 & 0x0f) << 4
				qlh[l+32] |= (code4 & 0x0f) << 4
				qhh[l] |= (code1 >> 4) & 0x03
				qhh[l] |= ((code2 >> 4) & 0x03) << 2
				qhh[l] |= ((code3 >> 4) & 0x03) << 4
				qhh[l] |= ((code4 >> 4) & 0x03) << 6
			}
		}
	}
	return out
}

// quantizeQ6KCode rounds a value against a per-sub-block scale into a 6-bit
// unsigned code biased by 32 (decoded as scale*(code-32)), the inverse of
// DequantRowQ6KInto's "- 32" step.
func quantizeQ6KCode(x, scale float32) byte {
	inv := float32(0)
	if scale != 0 {
		inv = 1 / scale
	}
	return byte(clampInt(int(roundHalfAwayFromZero(x*inv))+32, 0, 63))
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// roundHalfAwayFromZero matches llama.cpp's nearest_int (round half away
// from zero), which differs from Go's math.Round only in that math.Round
// already rounds half away from zero too — kept as a named seam so the
// rounding rule is documented at every call site rather than inlined.
func roundHalfAwayFromZero(v float32) float32 {
	return float32(math.Round(float64(v)))
}

// putF16 writes v as a little-endian IEEE 754 half-precision float at the
// start of b, matching binaryLE16(row) on the read side (simd.go).
func putF16(b []byte, v float32) {
	bits := F32ToF16(v)
	b[0] = byte(bits)
	b[1] = byte(bits >> 8)
}
