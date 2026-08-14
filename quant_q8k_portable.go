package gopherllm

import "math"

// Portable int8-activation ("Q8K") kernels.
//
// The fast matvec path quantizes the activation vector once to int8 with one
// scale per 256-element block (llama.cpp's Q8_K convention) and then dots each
// weight row against it in integer arithmetic, folding the block scales back in
// at the end. Until now that path existed only as amd64 assembly, so
// kernels_portable_tunables.go compiled it out entirely and every other
// architecture — including Apple Silicon — fell back to dequantizing weights to
// f32 and doing float FMAs.
//
// These functions are the architecture-independent implementation of exactly
// that arithmetic. They are deliberately written as straight-line scalar Go:
// their job is to be the readable definition of the contract, the fallback on
// targets without a hand-written kernel, and the oracle that per-architecture
// assembly is differentially tested against. They are promoted, unchanged in
// behaviour, from the scalar references that already guarded the amd64
// assembly, so the amd64 kernels' existing test suite doubles as validation
// that this arithmetic is right (see quant_q8k_portable_amd64_test.go, which
// asserts these agree with those references).
//
// Every function here takes the same arguments in the same order as its amd64
// assembly counterpart, so the two are drop-in interchangeable.

// q8kQuantizePortable quantizes x to int8 per 256-element block using
// symmetric absmax scaling, writing one scale per block. blocks*256 elements
// are read from x and written to q8.
//
// Rounding is round-to-nearest-even to match VCVTPS2DQ under the default
// MXCSR rounding mode, which is what the amd64 kernel emits. A block whose
// absmax is zero or NaN is left as zeros with a zero scale rather than
// producing infinities.
func q8kQuantizePortable(x []float32, q8 []int8, scales []float32, blocks int) {
	for b := range blocks {
		xb := x[b*256 : b*256+256]
		var amax float32
		for _, v := range xb {
			// NaN fails this comparison and is therefore skipped, so a block
			// containing one still scales off its finite elements. absF32 keeps
			// that property: clearing the sign bit leaves a NaN a NaN.
			if a := absF32(v); a > amax {
				amax = a
			}
		}
		if amax == 0 || amax != amax {
			// Leave scale zero: the row dots multiply by it, so the block
			// contributes nothing rather than a NaN.
			scales[b] = 0
			clear(q8[b*256 : b*256+256])
			continue
		}
		scales[b] = amax / 127
		inv := 127 / amax
		qb := q8[b*256 : b*256+256]
		for i, v := range xb {
			qb[i] = int8(roundToEvenF32(v * inv))
		}
	}
}

// absF32 and roundToEvenF32 exist because this loop runs over the whole
// activation vector on every matvec, and the obvious spellings
// (math.Abs(float64(v)) and math.RoundToEven(float64(v))) widen to float64 and
// back once per element for a result that is exactly representable in float32.
// Both of these are bit-identical to the float64 forms over the range the
// quantizer produces, which quant_q8k_portable_amd64_test.go asserts directly
// rather than assuming.

// absF32 clears the sign bit. Branchless, and unlike a comparison it leaves
// negative zero as positive zero, matching math.Abs.
func absF32(v float32) float32 {
	return math.Float32frombits(math.Float32bits(v) &^ (1 << 31))
}

// roundToEvenF32 rounds to the nearest integer, ties to even.
//
// Adding 2^23 pushes the value into the range where a float32's ulp is exactly
// 1, so the hardware's default round-to-nearest-even mode does the rounding as
// part of the addition; subtracting 2^23 then recovers it. Valid for
// |v| < 2^22, and this only ever sees |v| <= 127 or so.
//
// It rounds the magnitude and reapplies the sign bit rather than branching on
// the sign. That is exact because round-half-to-even is antisymmetric, it is
// branchless, and it keeps math.RoundToEven's handling of negative zero:
// rounding -0.5 the other way returns +0, which int8() cannot tell apart but a
// future caller of this helper could.
func roundToEvenF32(v float32) float32 {
	const magic = 8388608.0 // 2^23
	r := (absF32(v) + magic) - magic
	return math.Float32frombits(math.Float32bits(r) | (math.Float32bits(v) & (1 << 31)))
}

// q4kDotQ8KRowPortable computes one Q4_K row dot product against Q8K-quantized
// activations. xsums must hold the per-32-element float sums of the ORIGINAL
// (unquantized) activations, because the dmin term stays in exact float.
func q4kDotQ8KRowPortable(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*144 : (b+1)*144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q := block[16:144]
		var blockInt int32
		for s := range 4 {
			sc1, _ := getScaleMinK4(2*s, scales)
			sc2, _ := getScaleMinK4(2*s+1, scales)
			var lo, hi int32
			for l := range 32 {
				qv := q[s*32+l]
				lo += int32(qv&0x0f) * int32(q8[b*256+s*64+l])
				hi += int32(qv>>4) * int32(q8[b*256+s*64+32+l])
			}
			blockInt += int32(sc1)*lo + int32(sc2)*hi
		}
		sum += d * xscales[b] * float32(blockInt)
		var minTerm float32
		for j := range 8 {
			_, m := getScaleMinK4(j, scales)
			minTerm += float32(m) * xsums[b*8+j]
		}
		sum -= dmin * minTerm
	}
	return sum
}

// q5kDotQ8KRowPortable is the Q5_K analogue: the fifth bit of each quant comes
// from the qh bitplane and contributes 16 to the magnitude.
func q5kDotQ8KRowPortable(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*176 : (b+1)*176]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		qh := block[16:48]
		q := block[48:176]
		var blockInt int32
		for s := range 4 {
			sc1, _ := getScaleMinK4(2*s, scales)
			sc2, _ := getScaleMinK4(2*s+1, scales)
			var lo, hi int32
			for l := range 32 {
				qv := q[s*32+l]
				h1 := int32((qh[l] >> (2 * s)) & 1)
				h2 := int32((qh[l] >> (2*s + 1)) & 1)
				lo += (int32(qv&0x0f) + h1*16) * int32(q8[b*256+s*64+l])
				hi += (int32(qv>>4) + h2*16) * int32(q8[b*256+s*64+32+l])
			}
			blockInt += int32(sc1)*lo + int32(sc2)*hi
		}
		sum += d * xscales[b] * float32(blockInt)
		var minTerm float32
		for j := range 8 {
			_, m := getScaleMinK4(j, scales)
			minTerm += float32(m) * xsums[b*8+j]
		}
		sum -= dmin * minTerm
	}
	return sum
}

// q6kDotQ8KRowPortable is the Q6_K analogue. xsums must be the per-16-element
// sums of the original activations pre-scaled by 32, matching how the float
// path folds Q6_K's -32 quant offset into a separate term.
func q6kDotQ8KRowPortable(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*210 : (b+1)*210]
		ql := block[0:128]
		qh := block[128:192]
		sc := block[192:208]
		d := F16ToF32(binaryLE16(block[208:]))
		q8b := q8[b*256 : b*256+256]
		var blockInt int32
		for half := range 2 {
			qlh := ql[half*64 : half*64+64]
			qhh := qh[half*32 : half*32+32]
			sch := sc[half*8 : half*8+8]
			q8h := q8b[half*128 : half*128+128]
			for l := range 32 {
				is := l / 16
				q1 := int32((qlh[l] & 0x0f) | ((qhh[l] & 0x03) << 4))
				q2 := int32((qlh[l+32] & 0x0f) | (((qhh[l] >> 2) & 0x03) << 4))
				q3 := int32((qlh[l] >> 4) | (((qhh[l] >> 4) & 0x03) << 4))
				q4 := int32((qlh[l+32] >> 4) | (((qhh[l] >> 6) & 0x03) << 4))
				blockInt += int32(int8(sch[is]))*q1*int32(q8h[l]) +
					int32(int8(sch[is+2]))*q2*int32(q8h[32+l]) +
					int32(int8(sch[is+4]))*q3*int32(q8h[64+l]) +
					int32(int8(sch[is+6]))*q4*int32(q8h[96+l])
			}
		}
		sum += d * xscales[b] * float32(blockInt)
		var offTerm float32
		for i := range 16 {
			offTerm += float32(int8(sc[i])) * xsums[b*16+i]
		}
		sum -= d * offTerm
	}
	return sum
}

// q8_0DotQ8KRowPortable computes one Q8_0 row dot. Q8_0 stores 32-element
// blocks with their own f16 scale, so eight of them make up one 256-element
// Q8K activation block and no xsums term is needed.
func q8_0DotQ8KRowPortable(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		base := b * 272
		var blockSum float32
		for j := range 8 {
			off := base + j*34
			d := F16ToF32(binaryLE16(row[off:]))
			w := row[off+2 : off+34]
			q8b := q8[b*256+j*32 : b*256+j*32+32]
			var dot int32
			for l := range 32 {
				dot += int32(int8(w[l])) * int32(q8b[l])
			}
			blockSum += d * float32(dot)
		}
		sum += xscales[b] * blockSum
	}
	return sum
}

// q4_0DotQ8KRowPortable computes one Q4_0 row dot. Q4_0's quants are unsigned
// nibbles biased by -8, and that bias is folded out through xsums (the
// per-32-element activation sums) rather than applied per element.
func q4_0DotQ8KRowPortable(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for g := range blocks * 8 {
		block := row[g*18 : (g+1)*18]
		d := F16ToF32(binaryLE16(block[0:]))
		var intdot int32
		for i := range 16 {
			p := block[2+i]
			intdot += int32(p&0x0f) * int32(q8[g*32+i])
			intdot += int32(p>>4) * int32(q8[g*32+16+i])
		}
		sum += d * (xscales[g/8]*float32(intdot) - 8*xsums[g])
	}
	return sum
}

// q4_1DotQ8KRowPortable computes one Q4_1 row dot: per legacy block,
// d*xscale*intdot + m*xsum, where m is the stored per-block minimum.
func q4_1DotQ8KRowPortable(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for g := range blocks * 8 {
		block := row[g*20 : (g+1)*20]
		d := F16ToF32(binaryLE16(block[0:]))
		m := F16ToF32(binaryLE16(block[2:]))
		var intdot int32
		for i := range 16 {
			p := block[4+i]
			intdot += int32(p&0x0f) * int32(q8[g*32+i])
			intdot += int32(p>>4) * int32(q8[g*32+16+i])
		}
		sum += d*xscales[g/8]*float32(intdot) + m*xsums[g]
	}
	return sum
}

// mxfp4DoubledLUT maps an MXFP4 nibble to twice its value, so the whole block
// dot stays in integers and the factor 0.5 is applied once at the end.
var mxfp4DoubledLUT = [16]int32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// mxfp4DotQ8KRowPortable computes one MXFP4 (gpt-oss) row dot. The block scale
// is a raw power-of-two exponent byte; e == 0 means the block is zero.
func mxfp4DotQ8KRowPortable(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	var sum float32
	for g := range blocks * 8 {
		block := row[g*17 : (g+1)*17]
		e := int(block[16])
		var scale float32
		if e > 0 {
			scale = math.Float32frombits(uint32(e) << 23)
		}
		var intdot int32
		for i := range 16 {
			v := block[i]
			intdot += mxfp4DoubledLUT[v&0x0f] * int32(q8[g*32+i*2])
			intdot += mxfp4DoubledLUT[v>>4] * int32(q8[g*32+i*2+1])
		}
		sum += scale * 0.5 * xscales[g/8] * float32(intdot)
	}
	return sum
}

// q2kDotQ8KRowPortable computes one Q2_K row dot product against
// Q8K-quantized activations. Structurally identical to DotQ2KF32
// (quant_extra.go) — same 16-sub-blocks-of-16 walk with a running y/is index
// — except the per-element accumulation is against the int8-quantized
// activations (q8) instead of the original floats, with the per-sub-block
// scale sc&0x0f folded into the row's shared int32 accumulator (like
// q4kDotQ8KRowPortable's sc1*lo+sc2*hi) rather than applied per element, and
// a single d*xscales[b] float multiply closing out each superblock. xsums
// must hold the per-16-element float sums of the ORIGINAL (unquantized)
// activations, because the min term (dmin*(sc>>4)) stays in exact float —
// the same convention fillQ6KXSums16 already serves Q6_K with.
func q2kDotQ8KRowPortable(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*84 : (b+1)*84]
		scales := block[0:16]
		qs := block[16:80]
		d := F16ToF32(binaryLE16(block[80:]))
		dmin := F16ToF32(binaryLE16(block[82:]))
		q8b := q8[b*256 : b*256+256]
		var blockInt int32
		var minTerm float32
		is := 0
		y := 0
		for n := 0; n < 256; n += 128 {
			q := qs[(n/128)*32 : (n/128)*32+32]
			for shift := 0; shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					sc := scales[is]
					var qdot int32
					for l := 0; l < 16; l++ {
						qv := (q[half*16+l] >> shift) & 3
						qdot += int32(qv) * int32(q8b[y+l])
					}
					blockInt += int32(sc&0x0f) * qdot
					minTerm += float32(sc>>4) * xsums[b*16+is]
					is++
					y += 16
				}
			}
		}
		sum += d*xscales[b]*float32(blockInt) - dmin*minTerm
	}
	return sum
}

// q3kDotQ8KRowPortable computes one Q3_K row dot product against
// Q8K-quantized activations. Q3_K is symmetric (no dmin/min term, unlike
// Q2_K/Q4_K/Q5_K), but its high-bit correction is NOT a per-sub-block
// constant offset: DotQ3KF32's "v -= 4" fires per ELEMENT depending on that
// element's own hmask bit, so unlike the other K-quants this cannot be
// factored into "unsigned dot minus a shared xsum term" — it needs a genuine
// signed per-element value (range -4..3, a biased 3-bit code) dotted
// directly against q8. No xsums argument exists because there is no
// separable offset left once that per-element value is computed exactly.
func q3kDotQ8KRowPortable(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*110 : (b+1)*110]
		hmask := block[0:32]
		qs := block[32:96]
		scales := q3KScales(block[96:108])
		d := F16ToF32(binaryLE16(block[108:]))
		q8b := q8[b*256 : b*256+256]
		var blockInt int32
		is := 0
		m := byte(1)
		y := 0
		for n := 0; n < 256; n += 128 {
			q := qs[(n/128)*32 : (n/128)*32+32]
			for shift := 0; shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					dl := int32(scales[is]) - 32
					is++
					var qdot int32
					for l := 0; l < 16; l++ {
						v := int32((q[half*16+l] >> shift) & 3)
						if hmask[half*16+l]&m == 0 {
							v -= 4
						}
						qdot += v * int32(q8b[y+l])
					}
					blockInt += dl * qdot
					y += 16
				}
				m <<= 1
			}
		}
		sum += d * xscales[b] * float32(blockInt)
	}
	return sum
}
