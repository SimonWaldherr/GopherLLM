//go:build arm64

package gopherllm

import (
	"fmt"
	"math"
	"os"
	"slices"
)

// arm64 int8-activation kernels.
//
// SDOT (FEAT_DotProd) multiplies four int8 pairs and accumulates into an int32
// lane, so one instruction retires 16 int8 MACs across a 128-bit vector. The
// float path this replaces does 4 f32 FMAs per VFMLA and has to convert every
// unpacked nibble to float first, so the int8 path removes both the conversion
// and roughly three quarters of the multiply instructions.
//
// The split of work is deliberate and uniform across all seven formats:
// assembly does the unpack and the integer dots (see q8k_dot_arm64.s), Go
// decodes the scales and does the float combination. Scale decode runs 8 or
// 16 times per 256 elements against 32-64 SDOTs, so keeping it in Go costs
// almost nothing and keeps each kernel small enough to review by eye.
//
// SDOT is optional before ARMv8.4, so these kernels are reachable only after
// hasDotProd (dotprod_arm64.go) confirms the CPU has it. On a part without
// dotprod every *DotAsmOK below is false and each entry point returns its
// portable counterpart, which is the exact behaviour non-Apple arm64 targets
// had when this file was gated to darwin && arm64.
//
// hasQ8KDotAsm therefore tracks the probe rather than the build tag. It also
// decides whether the int8-activation path is the default (see
// defaultQ8Activations in kernels_portable_tunables.go);
// GOPHERLLM_Q8_ACTIVATIONS=0 forces the exact float kernels back on for A/B
// testing or if a numerical problem is suspected.
var hasQ8KDotAsm = hasDotProd

/* ── Assembly entry points ───────────────────────────────────────────────── */

// Each of these computes the raw integer dot products of ONE 256-element
// superchunk, before any scale is applied. q8 always points at that
// superchunk's 256 int8 activations, out at the dots.

// q4kQ8Dots8Asm: 8 sub-block dots. q is the 128 packed nibble bytes.
//
//go:noescape
func q4kQ8Dots8Asm(q *byte, q8 *int8, out *int32)

// q5kQ8Dots8Asm: 8 sub-block dots. q is 128 packed nibble bytes, qh the
// 32-byte fifth-bit plane.
//
//go:noescape
func q5kQ8Dots8Asm(q *byte, qh *byte, q8 *int8, out *int32)

// q6kQ8Dots16Asm: 16 per-scale-group dots. ql is 128 bytes, qh 64.
//
//go:noescape
func q6kQ8Dots16Asm(ql *byte, qh *byte, q8 *int8, out *int32)

// q8_0Q8Dots8Asm: 8 block dots over 8 consecutive 34-byte blocks.
//
//go:noescape
func q8_0Q8Dots8Asm(row *byte, q8 *int8, out *int32)

// q4_0Q8Dots8Asm: 8 block dots over 8 consecutive 18-byte blocks.
//
//go:noescape
func q4_0Q8Dots8Asm(row *byte, q8 *int8, out *int32)

// q4_1Q8Dots8Asm: 8 block dots over 8 consecutive 20-byte blocks.
//
//go:noescape
func q4_1Q8Dots8Asm(row *byte, q8 *int8, out *int32)

// mxfp4Q8Dots8Asm: 8 block dots over 8 consecutive 17-byte blocks. lut is the
// 16-entry table of doubled signed values that VTBL indexes with each nibble.
//
//go:noescape
func mxfp4Q8Dots8Asm(row *byte, q8 *int8, lut *int8, out *int32)

// dotInt8Asm is a plain int8 dot product over n elements, n a multiple of 16.
// No row kernel uses it; it exists so the tests can pin SDOT's behaviour on its
// own, with no block format in the way.
//
//go:noescape
func dotInt8Asm(a, b *int8, n int) int32

// q2kQ8Dots16Asm: 16 sub-block dots. qs is the 64 packed 2-bit-code bytes.
//
//go:noescape
func q2kQ8Dots16Asm(qs *byte, q8 *int8, out *int32)

// q3kQ8Dots16Asm: 16 sub-block dots, with the per-element high-bit bias
// already applied. qs is the 64 packed 2-bit-code bytes, hmask the 32-byte
// bit-plane.
//
//go:noescape
func q3kQ8Dots16Asm(qs *byte, hmask *byte, q8 *int8, out *int32)

// mxfp4DoubledLUT8 narrows the portable kernel's table to the int8 form VTBL
// needs. Every entry is within [-12, 12], so nothing is lost.
var mxfp4DoubledLUT8 = func() [16]int8 {
	var t [16]int8
	for i, v := range mxfp4DoubledLUT {
		t[i] = int8(v)
	}
	return t
}()

/* ── Startup self-checks ─────────────────────────────────────────────────── */

// SDOT is emitted as a raw WORD because Go has no mnemonic for it, so nothing
// in the toolchain validates that encoding — not the assembler, and not the
// disassembler, whose instruction table predates FEAT_DotProd. The kernels were
// also written and cross-compiled on machines that cannot execute them. The
// failure mode is therefore the bad one: not a crash, but subtly wrong logits
// that still look like plausible text.
//
// So every kernel proves itself once, at package init, against the same scalar
// unpack its portable counterpart uses. On mismatch the process keeps running on
// the portable path, which is correct everywhere, and says so on stderr. This
// costs one 256-element block per format at startup and removes the need to
// trust untested assembly. q8k_dot_arm64_test.go covers the same ground far
// more thoroughly; these are the belt to that suite's braces.
//
// The self-checks matter more now that this file is no longer Apple-only: they
// run against whatever arm64 part the binary landed on, so a lane-order or
// signedness bug that only manifests on some other microarchitecture degrades
// to the portable kernel with a diagnostic instead of silently corrupting
// logits.
// Each flag short-circuits on hasDotProd FIRST. That ordering is load-bearing,
// not stylistic: validate* executes SDOT, so calling one on a CPU without
// FEAT_DotProd would SIGILL during package init, before any user code runs. The
// && also means a non-dotprod part pays nothing for these checks.
var (
	q4kDotAsmOK   = hasDotProd && validateQ4KQ8Dots8Asm()
	q5kDotAsmOK   = hasDotProd && validateQ5KQ8Dots8Asm()
	q6kDotAsmOK   = hasDotProd && validateQ6KQ8Dots16Asm()
	q8_0DotAsmOK  = hasDotProd && validateQ8_0Q8Dots8Asm()
	q4_0DotAsmOK  = hasDotProd && validateQ4_0Q8Dots8Asm()
	q4_1DotAsmOK  = hasDotProd && validateQ4_1Q8Dots8Asm()
	mxfp4DotAsmOK = hasDotProd && validateMXFP4Q8Dots8Asm()
	q2kDotAsmOK   = hasDotProd && validateQ2KQ8Dots16Asm()
	q3kDotAsmOK   = hasDotProd && validateQ3KQ8Dots16Asm()
)

// q8kSelfCheck reports a kernel mismatch once and returns whether the kernel is
// usable. One place for the message so seven kernels cannot drift in how they
// describe the same failure.
func q8kSelfCheck(format string, got, want []int32) bool {
	if slices.Equal(got, want) {
		return true
	}
	fmt.Fprintf(os.Stderr,
		"gopherllm: NEON SDOT %s self-check failed (got %v want %v); falling back to the portable %s int8 kernel. Please report this with your CPU model.\n",
		format, got, want, format)
	return false
}

// q8kSelfCheckActivations builds a deterministic activation block that exercises
// both signs and the int8 extremes, so a lane-order or signedness mistake cannot
// cancel out.
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

// q8kSelfCheckBytes fills n bytes deterministically. Callers pass different
// multipliers so no two formats share a pattern that could hide a layout bug in
// one of them, and so every bit position gets exercised.
func q8kSelfCheckBytes(n, mul, add int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*mul + add)
	}
	return b
}

func validateQ4KQ8Dots8Asm() bool {
	q := q8kSelfCheckBytes(128, 7, 0)
	q8 := q8kSelfCheckActivations()
	want := make([]int32, 8)
	for s := range 4 {
		var lo, hi int32
		for l := range 32 {
			qv := q[s*32+l]
			lo += int32(qv&0x0f) * int32(q8[s*64+l])
			hi += int32(qv>>4) * int32(q8[s*64+32+l])
		}
		want[2*s], want[2*s+1] = lo, hi
	}
	got := make([]int32, 8)
	q4kQ8Dots8Asm(&q[0], &q8[0], &got[0])
	return q8kSelfCheck("Q4_K", got, want)
}

func validateQ5KQ8Dots8Asm() bool {
	q := q8kSelfCheckBytes(128, 7, 0)
	qh := q8kSelfCheckBytes(32, 37, 11)
	q8 := q8kSelfCheckActivations()
	want := make([]int32, 8)
	for s := range 4 {
		var lo, hi int32
		for l := range 32 {
			qv := q[s*32+l]
			h1 := int32((qh[l] >> (2 * s)) & 1)
			h2 := int32((qh[l] >> (2*s + 1)) & 1)
			lo += (int32(qv&0x0f) + h1*16) * int32(q8[s*64+l])
			hi += (int32(qv>>4) + h2*16) * int32(q8[s*64+32+l])
		}
		want[2*s], want[2*s+1] = lo, hi
	}
	got := make([]int32, 8)
	q5kQ8Dots8Asm(&q[0], &qh[0], &q8[0], &got[0])
	return q8kSelfCheck("Q5_K", got, want)
}

// validateQ2KQ8Dots16Asm walks the 2-bit codes independently of the kernel's
// chunk/shift/half structure: sub-block is takes its codes from
// qs[(is/8)*32 + (is%2)*16 + l] at shift 2*((is/2)%4). Deriving the indices
// from is (rather than from a nested loop shaped like the assembly) is what
// makes a transposed sub-block order show up as a mismatch.
func validateQ2KQ8Dots16Asm() bool {
	qs := q8kSelfCheckBytes(64, 37, 11)
	q8 := q8kSelfCheckActivations()
	want := make([]int32, 16)
	for is := range 16 {
		chunk := (is / 8) * 32
		shift := uint(2 * ((is / 2) % 4))
		half := (is % 2) * 16
		var dot int32
		for l := range 16 {
			dot += int32((qs[chunk+half+l]>>shift)&3) * int32(q8[is*16+l])
		}
		want[is] = dot
	}
	got := make([]int32, 16)
	q2kQ8Dots16Asm(&qs[0], &q8[0], &got[0])
	return q8kSelfCheck("Q2_K", got, want)
}

// validateQ3KQ8Dots16Asm additionally pins the hmask bit assignment, which is
// the one thing that cannot be checked by inspection: sub-block is takes bit
// is/2 of hmask[(is%2)*16 + l], continuing across the chunk boundary. A kernel
// that reused one bit for all eight groups, or reset the bit per chunk, fails
// here rather than silently corrupting logits.
func validateQ3KQ8Dots16Asm() bool {
	qs := q8kSelfCheckBytes(64, 41, 3)
	hmask := q8kSelfCheckBytes(32, 23, 19)
	q8 := q8kSelfCheckActivations()
	want := make([]int32, 16)
	for is := range 16 {
		chunk := (is / 8) * 32
		shift := uint(2 * ((is / 2) % 4))
		half := (is % 2) * 16
		hbit := uint(is / 2)
		var dot int32
		for l := range 16 {
			v := int32((qs[chunk+half+l] >> shift) & 3)
			if hmask[half+l]&(1<<hbit) == 0 {
				v -= 4
			}
			dot += v * int32(q8[is*16+l])
		}
		want[is] = dot
	}
	got := make([]int32, 16)
	q3kQ8Dots16Asm(&qs[0], &hmask[0], &q8[0], &got[0])
	return q8kSelfCheck("Q3_K", got, want)
}

func validateQ6KQ8Dots16Asm() bool {
	ql := q8kSelfCheckBytes(128, 13, 0)
	qh := q8kSelfCheckBytes(64, 29, 7)
	q8 := q8kSelfCheckActivations()
	want := make([]int32, 16)
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
	got := make([]int32, 16)
	q6kQ8Dots16Asm(&ql[0], &qh[0], &q8[0], &got[0])
	return q8kSelfCheck("Q6_K", got, want)
}

func validateQ8_0Q8Dots8Asm() bool {
	row := q8kSelfCheckBytes(272, 23, 5)
	q8 := q8kSelfCheckActivations()
	want := make([]int32, 8)
	for j := range 8 {
		off := j*34 + 2
		for l := range 32 {
			want[j] += int32(int8(row[off+l])) * int32(q8[j*32+l])
		}
	}
	got := make([]int32, 8)
	q8_0Q8Dots8Asm(&row[0], &q8[0], &got[0])
	return q8kSelfCheck("Q8_0", got, want)
}

// legacyNibbleDots is the shared scalar reference for the 32-element nibble
// formats: low nibbles against the first half of each block's activations, high
// nibbles against the second. header is the byte offset of the packed nibbles
// within a block and stride the block size, which is all that separates Q4_0
// from Q4_1.
func legacyNibbleDots(row []byte, q8 *[256]int8, header, stride int) []int32 {
	out := make([]int32, 8)
	for j := range 8 {
		off := j*stride + header
		for i := range 16 {
			p := row[off+i]
			out[j] += int32(p&0x0f) * int32(q8[j*32+i])
			out[j] += int32(p>>4) * int32(q8[j*32+16+i])
		}
	}
	return out
}

func validateQ4_0Q8Dots8Asm() bool {
	row := q8kSelfCheckBytes(144, 19, 3)
	q8 := q8kSelfCheckActivations()
	want := legacyNibbleDots(row, &q8, 2, 18)
	got := make([]int32, 8)
	q4_0Q8Dots8Asm(&row[0], &q8[0], &got[0])
	return q8kSelfCheck("Q4_0", got, want)
}

func validateQ4_1Q8Dots8Asm() bool {
	row := q8kSelfCheckBytes(160, 31, 9)
	q8 := q8kSelfCheckActivations()
	want := legacyNibbleDots(row, &q8, 4, 20)
	got := make([]int32, 8)
	q4_1Q8Dots8Asm(&row[0], &q8[0], &got[0])
	return q8kSelfCheck("Q4_1", got, want)
}

func validateMXFP4Q8Dots8Asm() bool {
	row := q8kSelfCheckBytes(136, 41, 13)
	q8 := q8kSelfCheckActivations()
	want := make([]int32, 8)
	for j := range 8 {
		off := j * 17
		for i := range 16 {
			v := row[off+i]
			// Interleaved, unlike Q4_0: the low nibble takes activation 2i.
			want[j] += mxfp4DoubledLUT[v&0x0f] * int32(q8[j*32+i*2])
			want[j] += mxfp4DoubledLUT[v>>4] * int32(q8[j*32+i*2+1])
		}
	}
	got := make([]int32, 8)
	mxfp4Q8Dots8Asm(&row[0], &q8[0], &mxfp4DoubledLUT8[0], &got[0])
	return q8kSelfCheck("MXFP4", got, want)
}

/* ── Row kernels ─────────────────────────────────────────────────────────── */

// The per-row loops stay separate per format rather than sharing one function
// with a callback. The repo already measured that a func-value call per row
// costs ~50ns against a ~127ns kernel (see argmaxQ6KRowsQ8 on amd64), so a
// closure here would give back a third of the win these kernels exist for. What
// IS shared is the scale arithmetic, one static call per block.

// combineQ4KStyle folds Q4_K/Q5_K's packed 6-bit scale/min pairs into 8 raw
// unsigned sub-block dots. The two formats differ only in block stride and
// unpack kernel; this arithmetic is identical, so it lives once. xsums must
// start at this block's 8 per-32-element activation sums.
//
// The eight scale/min pairs are deliberately unpacked inline. This is reached
// for every Q4_K/Q5_K row and the generic getScaleMinK4 helper's loop/branch
// showed up prominently in real embedding profiles. Keeping the arithmetic
// spelled out preserves the exact j=0..7 accumulation order while removing
// eight repeated packed-field decodes per 256-element superblock.
func combineQ4KStyle(scales []byte, qdots *[8]int32, xsums []float32, d, dmin, xscale float32) float32 {
	_ = scales[11]
	_ = qdots[7]
	_ = xsums[7]

	s0, m0 := int32(scales[0]&63), int32(scales[4]&63)
	s1, m1 := int32(scales[1]&63), int32(scales[5]&63)
	s2, m2 := int32(scales[2]&63), int32(scales[6]&63)
	s3, m3 := int32(scales[3]&63), int32(scales[7]&63)
	s4 := int32((scales[8] & 0x0f) | ((scales[0] >> 6) << 4))
	m4 := int32((scales[8] >> 4) | ((scales[4] >> 6) << 4))
	s5 := int32((scales[9] & 0x0f) | ((scales[1] >> 6) << 4))
	m5 := int32((scales[9] >> 4) | ((scales[5] >> 6) << 4))
	s6 := int32((scales[10] & 0x0f) | ((scales[2] >> 6) << 4))
	m6 := int32((scales[10] >> 4) | ((scales[6] >> 6) << 4))
	s7 := int32((scales[11] & 0x0f) | ((scales[3] >> 6) << 4))
	m7 := int32((scales[11] >> 4) | ((scales[7] >> 6) << 4))

	var blockInt int32
	blockInt += s0 * qdots[0]
	blockInt += s1 * qdots[1]
	blockInt += s2 * qdots[2]
	blockInt += s3 * qdots[3]
	blockInt += s4 * qdots[4]
	blockInt += s5 * qdots[5]
	blockInt += s6 * qdots[6]
	blockInt += s7 * qdots[7]
	var minTerm float32
	minTerm += float32(m0) * xsums[0]
	minTerm += float32(m1) * xsums[1]
	minTerm += float32(m2) * xsums[2]
	minTerm += float32(m3) * xsums[3]
	minTerm += float32(m4) * xsums[4]
	minTerm += float32(m5) * xsums[5]
	minTerm += float32(m6) * xsums[6]
	minTerm += float32(m7) * xsums[7]
	return d*xscale*float32(blockInt) - dmin*minTerm
}

// q4kDotQ8KRow computes one Q4_K row dot against Q8K-quantized activations.
// xsums holds the per-32-element sums of the ORIGINAL activations, so the dmin
// term stays exact float — matching q4kDotQ8KRowPortable, the test oracle.
func q4kDotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	if !q4kDotAsmOK {
		return q4kDotQ8KRowPortable(row, q8, xscales, xsums, blocks)
	}
	var qdots [8]int32
	var sum float32
	for b := range blocks {
		block := row[b*144 : (b+1)*144]
		q4kQ8Dots8Asm(&block[16], &q8[b*256], &qdots[0])
		sum += combineQ4KStyle(block[4:16], &qdots, xsums[b*8:],
			F16ToF32(binaryLE16(block[0:])), F16ToF32(binaryLE16(block[2:])), xscales[b])
	}
	return sum
}

// q5kDotQ8KRow is the Q5_K analogue: same scale structure, one extra bitplane.
//
// Q5_K_M matters beyond being a common mix in its own right: it is the
// recommended quantization for the hybrid Mamba-2/MoE models this engine
// supports through the nemotron_h path.
func q5kDotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	if !q5kDotAsmOK {
		return q5kDotQ8KRowPortable(row, q8, xscales, xsums, blocks)
	}
	var qdots [8]int32
	var sum float32
	for b := range blocks {
		block := row[b*176 : (b+1)*176]
		q5kQ8Dots8Asm(&block[48], &block[16], &q8[b*256], &qdots[0])
		sum += combineQ4KStyle(block[4:16], &qdots, xsums[b*8:],
			F16ToF32(binaryLE16(block[0:])), F16ToF32(binaryLE16(block[2:])), xscales[b])
	}
	return sum
}

// q6kDotQ8KRow computes one Q6_K row dot.
//
// Q6_K matters more than its share of a model's bytes suggests: a Q4_K_M mix
// stores output.weight as Q6_K, so this runs over the whole vocabulary on every
// token. xsums are the per-16-element sums pre-scaled by 32, carrying Q6_K's -32
// quant offset as an exact float term.
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
		sum += d * (xscales[b]*float32(blockInt) - offTerm)
	}
	return sum
}

// q8_0DotQ8KRow computes one Q8_0 row dot. Q8_0 is symmetric, so unlike the
// K-quants there is no offset term and no xsums argument.
func q8_0DotQ8KRow(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	if !q8_0DotAsmOK {
		return q8_0DotQ8KRowPortable(row, q8, xscales, blocks)
	}
	var qdots [8]int32
	var sum float32
	for b := range blocks {
		base := b * 272
		q8_0Q8Dots8Asm(&row[base], &q8[b*256], &qdots[0])
		var blockSum float32
		for j := range 8 {
			blockSum += F16ToF32(binaryLE16(row[base+j*34:])) * float32(qdots[j])
		}
		sum += xscales[b] * blockSum
	}
	return sum
}

// q4_0DotQ8KRow computes one Q4_0 row dot. Q4_0's quants are unsigned nibbles
// biased by -8, and that bias is folded out through xsums rather than applied
// per element, which is why the kernel returns raw unsigned dots.
func q4_0DotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	if !q4_0DotAsmOK {
		return q4_0DotQ8KRowPortable(row, q8, xscales, xsums, blocks)
	}
	var qdots [8]int32
	var sum float32
	for b := range blocks {
		base := b * 144
		q4_0Q8Dots8Asm(&row[base], &q8[b*256], &qdots[0])
		xscale := xscales[b]
		for j := range 8 {
			d := F16ToF32(binaryLE16(row[base+j*18:]))
			sum += d * (xscale*float32(qdots[j]) - 8*xsums[b*8+j])
		}
	}
	return sum
}

// q4_1DotQ8KRow computes one Q4_1 row dot: per block, d*xscale*dot + m*xsum,
// where m is the stored per-block minimum.
func q4_1DotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	if !q4_1DotAsmOK {
		return q4_1DotQ8KRowPortable(row, q8, xscales, xsums, blocks)
	}
	var qdots [8]int32
	var sum float32
	for b := range blocks {
		base := b * 160
		q4_1Q8Dots8Asm(&row[base], &q8[b*256], &qdots[0])
		xscale := xscales[b]
		for j := range 8 {
			off := base + j*20
			d := F16ToF32(binaryLE16(row[off:]))
			m := F16ToF32(binaryLE16(row[off+2:]))
			sum += d*xscale*float32(qdots[j]) + m*xsums[b*8+j]
		}
	}
	return sum
}

// mxfp4DotQ8KRow computes one MXFP4 (gpt-oss) row dot. The block scale is a raw
// power-of-two exponent byte, where 0 means the block is zero; the 0.5
// compensates for the doubled lookup table.
func mxfp4DotQ8KRow(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	if !mxfp4DotAsmOK {
		return mxfp4DotQ8KRowPortable(row, q8, xscales, blocks)
	}
	var qdots [8]int32
	var sum float32
	for b := range blocks {
		base := b * 136
		mxfp4Q8Dots8Asm(&row[base], &q8[b*256], &mxfp4DoubledLUT8[0], &qdots[0])
		xscale := xscales[b]
		for j := range 8 {
			e := uint32(row[base+j*17+16])
			if e == 0 {
				continue
			}
			sum += math.Float32frombits(e<<23) * 0.5 * xscale * float32(qdots[j])
		}
	}
	return sum
}

// q2kDotQ8KRow computes one Q2_K row dot. Q2_K's 16 sub-blocks of 16 elements
// each carry a packed scale/min byte (low nibble = d-scale, high nibble =
// min-scale), so the structure is Q4_K's split differently: the integer dots
// come from assembly, the nibble decode and the exact-float dmin term stay
// here. xsums are the per-16-element sums of the ORIGINAL activations
// (fillQ6KXSums16 — Q2_K shares Q6_K's 16-element grouping), unscaled.
func q2kDotQ8KRow(row []byte, q8 []int8, xscales, xsums []float32, blocks int) float32 {
	if !q2kDotAsmOK {
		return q2kDotQ8KRowPortable(row, q8, xscales, xsums, blocks)
	}
	var qdots [16]int32
	var sum float32
	for b := range blocks {
		block := row[b*84 : (b+1)*84]
		scales := block[0:16]
		d := F16ToF32(binaryLE16(block[80:]))
		dmin := F16ToF32(binaryLE16(block[82:]))
		q2kQ8Dots16Asm(&block[16], &q8[b*256], &qdots[0])
		var blockInt int32
		var minTerm float32
		for i := range 16 {
			sc := scales[i]
			blockInt += int32(sc&0x0f) * qdots[i]
			minTerm += float32(sc>>4) * xsums[b*16+i]
		}
		sum += d*xscales[b]*float32(blockInt) - dmin*minTerm
	}
	return sum
}

// q3kDotQ8KRow computes one Q3_K row dot. The kernel returns dots that already
// include the per-element -4 high-bit bias (see Q3SUB* in q8k_dot_arm64.s),
// because SDOT is signed×signed and the biased code -4..3 is a valid int8 —
// so, unlike the AVX2 kernel, no activation-sum correction is needed. Go is
// left with the six-bit scale unpack and dl = scale - 32.
//
// Q3_K is symmetric, so there is no xsums argument and no min term.
func q3kDotQ8KRow(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	if !q3kDotAsmOK {
		return q3kDotQ8KRowPortable(row, q8, xscales, blocks)
	}
	var qdots [16]int32
	var sum float32
	for b := range blocks {
		block := row[b*110 : (b+1)*110]
		scales := q3KScales(block[96:108])
		d := F16ToF32(binaryLE16(block[108:]))
		q3kQ8Dots16Asm(&block[32], &block[0], &q8[b*256], &qdots[0])
		var blockInt int32
		for i := range 16 {
			blockInt += (int32(scales[i]) - 32) * qdots[i]
		}
		sum += d * xscales[b] * float32(blockInt)
	}
	return sum
}

func q8kQuantize(x []float32, q8 []int8, scales []float32, blocks int) {
	// The quantizer is a float absmax pass plus a round-to-int8 pass; it is
	// bandwidth-bound on the activation vector, which is tiny next to a weight
	// matrix, and it runs once per matvec rather than once per row. Not worth
	// its own assembly.
	q8kQuantizePortable(x, q8, scales, blocks)
}
