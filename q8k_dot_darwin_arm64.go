//go:build darwin && arm64

package gopherllm

import (
	"fmt"
	"os"
)

// Apple Silicon int8-activation kernels.
//
// SDOT (FEAT_DotProd) multiplies four int8 pairs and accumulates into an int32
// lane, so one instruction retires 16 int8 MACs across a 128-bit vector. The
// float path this replaces does 4 f32 FMAs per VFMLA and has to convert every
// unpacked nibble to float first, so the int8 path removes both the conversion
// and roughly three quarters of the multiply instructions.
//
// The split of work is deliberate: assembly does the nibble unpack and the
// integer dots (see q8k_dot_darwin_arm64.s), Go does the 6-bit packed
// scale/min decode and the float combination. That decode is intricate, it is
// already written and tested once in quant_q8k_portable.go, and moving it into
// hand-encoded assembly would buy little — it runs 8 times per 256 elements,
// against 64 SDOTs — while making the kernel much harder to verify.
//
// hasQ8KDotAsm is true here, which also makes the int8-activation path the
// default on Apple Silicon (see defaultQ8Activations in
// kernels_portable_tunables.go). GOPHERLLM_Q8_ACTIVATIONS=0 forces the exact
// float kernels back on for A/B testing or if a numerical problem is suspected.
const hasQ8KDotAsm = true

// q4kQ8Dots8Asm computes the 8 per-sub-block int32 dot products of one Q4_K
// block: unsigned 4-bit quants times int8 activations, no scales applied. q
// must point at the block's 128 packed nibble bytes, q8 at its 256 int8
// activations, out at 8 int32s.
//
//go:noescape
func q4kQ8Dots8Asm(q *byte, q8 *int8, out *int32)

// dotInt8Asm is a plain int8 dot product over n elements, n a multiple of 16.
//
//go:noescape
func dotInt8Asm(a, b *int8, n int) int32

func q8kQuantize(x []float32, q8 []int8, scales []float32, blocks int) {
	// The quantizer is a float absmax pass plus a round-to-int8 pass; it is
	// bandwidth-bound on the activation vector, which is tiny next to a weight
	// matrix, and it runs once per matvec rather than once per row. Not worth
	// its own assembly.
	q8kQuantizePortable(x, q8, scales, blocks)
}

// q4kDotAsmOK is a startup self-check on the hand-encoded SDOT kernel.
//
// SDOT is emitted as a raw WORD because Go has no mnemonic for it, so nothing
// in the toolchain validates that encoding — not the assembler, and not the
// disassembler, whose instruction table predates FEAT_DotProd. The kernel is
// also written and cross-compiled on machines that cannot execute it. Its
// failure mode is therefore the bad one: not a crash, but subtly wrong logits
// that still look like plausible text.
//
// So the kernel proves itself once, at package init, against the same scalar
// unpack the portable kernel uses. If it disagrees the process keeps running on
// the portable path instead, which is correct everywhere. This costs one
// 256-element block of work at startup and removes the need to trust untested
// assembly. q8k_dot_darwin_arm64_test.go covers the same ground far more
// thoroughly; this is the belt to that suite's braces.
var (
	q4kDotAsmOK = validateQ4KQ8Dots8Asm()
	q6kDotAsmOK = validateQ6KQ8Dots16Asm()
)

// q8kSelfCheckActivations builds a deterministic activation block that
// exercises both signs and the int8 extremes, so a lane-order or signedness
// mistake cannot cancel out.
func q8kSelfCheckActivations() [256]int8 {
	var q8 [256]int8
	for i := range q8 {
		switch i % 4 {
		case 0:
			q8[i] = 127
		case 1:
			q8[i] = -128
		case 2:
			q8[i] = int8(i%251 - 125)
		default:
			q8[i] = 0
		}
	}
	return q8
}

func validateQ6KQ8Dots16Asm() bool {
	var ql [128]byte
	var qh [64]byte
	for i := range ql {
		ql[i] = byte(i*13 + i/8)
	}
	for i := range qh {
		qh[i] = byte(i*29 + 7)
	}
	q8 := q8kSelfCheckActivations()

	var want [16]int32
	for half := range 2 {
		qlh := ql[half*64 : half*64+64]
		qhh := qh[half*32 : half*32+32]
		q8h := q8[half*128 : half*128+128]
		for l := range 32 {
			is := l / 16
			q1 := int32((qlh[l] & 0x0f) | ((qhh[l] & 0x03) << 4))
			q2 := int32((qlh[l+32] & 0x0f) | (((qhh[l] >> 2) & 0x03) << 4))
			q3 := int32((qlh[l] >> 4) | (((qhh[l] >> 4) & 0x03) << 4))
			q4 := int32((qlh[l+32] >> 4) | (((qhh[l] >> 6) & 0x03) << 4))
			want[half*8+0*2+is] += q1 * int32(q8h[l])
			want[half*8+1*2+is] += q2 * int32(q8h[32+l])
			want[half*8+2*2+is] += q3 * int32(q8h[64+l])
			want[half*8+3*2+is] += q4 * int32(q8h[96+l])
		}
	}

	var got [16]int32
	q6kQ8Dots16Asm(&ql[0], &qh[0], &q8[0], &got[0])
	if got != want {
		fmt.Fprintf(os.Stderr,
			"gopherllm: NEON SDOT Q6_K self-check failed (got %v want %v); falling back to the portable Q6_K int8 kernel. Please report this with your CPU model.\n",
			got, want)
		return false
	}
	return true
}

func validateQ4KQ8Dots8Asm() bool {
	// Deterministic inputs that exercise both nibble halves, both signs of the
	// activations, and the int8 extremes.
	var q [128]byte
	for i := range q {
		q[i] = byte(i*7 + i/16) // spreads distinct values across both nibbles
	}
	q8 := q8kSelfCheckActivations()

	var want [8]int32
	for s := range 4 {
		var lo, hi int32
		for l := range 32 {
			qv := q[s*32+l]
			lo += int32(qv&0x0f) * int32(q8[s*64+l])
			hi += int32(qv>>4) * int32(q8[s*64+32+l])
		}
		want[2*s], want[2*s+1] = lo, hi
	}

	var got [8]int32
	q4kQ8Dots8Asm(&q[0], &q8[0], &got[0])
	if got != want {
		// Loud, once, on the way past: a silent downgrade would hide a real bug.
		fmt.Fprintf(os.Stderr,
			"gopherllm: NEON SDOT self-check failed (got %v want %v); falling back to the portable Q4_K int8 kernel. Please report this with your CPU model.\n",
			got, want)
		return false
	}
	return true
}

// q4kDotQ8KRow computes one Q4_K row dot against Q8K-quantized activations,
// with the integer sub-block dots in NEON. xsums holds the per-32-element
// sums of the ORIGINAL activations, so the dmin term stays exact float —
// matching q4kDotQ8KRowPortable, which is this function's test oracle.
func q4kDotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	if !q4kDotAsmOK {
		return q4kDotQ8KRowPortable(row, q8, xscales, xsums, blocks)
	}
	var qdots [8]int32
	var sum float32
	for b := range blocks {
		block := row[b*144 : (b+1)*144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]

		q4kQ8Dots8Asm(&block[16], &q8[b*256], &qdots[0])

		var blockInt int32
		var minTerm float32
		for j := range 8 {
			sc, m := getScaleMinK4(j, scales)
			blockInt += int32(sc) * qdots[j]
			minTerm += float32(m) * xsums[b*8+j]
		}
		sum += d * xscales[b] * float32(blockInt)
		sum -= dmin * minTerm
	}
	return sum
}

// q6kQ8Dots16Asm computes the 16 per-scale-group int32 dot products of one
// Q6_K block. ql must point at 128 bytes, qh at 64, q8 at 256 int8
// activations, out at 16 int32s.
//
//go:noescape
func q6kQ8Dots16Asm(ql *byte, qh *byte, q8 *int8, out *int32)

// q6kDotQ8KRow computes one Q6_K row dot against Q8K-quantized activations.
//
// Q6_K matters more than its share of a model's bytes suggests: a Q4_K_M mix
// stores output.weight as Q6_K, so this kernel runs over the whole vocabulary
// on every single token. Leaving it on the scalar portable kernel while Q4_K
// got NEON would have made the int8 path a net regression on Apple Silicon,
// because the float path it replaces is already vectorised there
// (q6kQDots16 in kernels_arm64.s).
//
// xsums are the per-16-element sums of the original activations pre-scaled by
// 32, carrying Q6_K's -32 quant offset as an exact float term.
func q6kDotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	if !q6kDotAsmOK {
		return q6kDotQ8KRowPortable(row, q8, xscales, xsums, blocks)
	}
	var qdots [16]int32
	var sum float32
	for b := range blocks {
		block := row[b*210 : (b+1)*210]
		sc := block[192:208]
		d := F16ToF32(binaryLE16(block[208:]))

		q6kQ8Dots16Asm(&block[0], &block[128], &q8[b*256], &qdots[0])

		// The assembly indexes its output by half*8 + group*2 + l/16, which is
		// exactly the scale index, so this is a straight walk.
		var blockInt int32
		var offTerm float32
		for i := range 16 {
			s := int32(int8(sc[i]))
			blockInt += s * qdots[i]
			offTerm += float32(s) * xsums[b*16+i]
		}
		sum += d * xscales[b] * float32(blockInt)
		sum -= d * offTerm
	}
	return sum
}
