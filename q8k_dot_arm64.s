//go:build arm64

#include "textflag.h"

// NEON int8 dot-product kernels for the Q8K activation path on arm64.
//
// WHY WORD ENCODINGS
//
// Go's arm64 assembler has no SDOT/UDOT/USDOT/SMMLA mnemonic, and in fact no
// integer SIMD multiply of any kind — VPMULL is *polynomial* multiply for
// crypto, not a widening integer multiply. So neither SDOT nor the classic
// pre-dotprod VMULL+VPADAL fallback can be written as mnemonics. SDOT is
// therefore emitted as a raw WORD, the same way this package's existing
// kernels_arm64.s already emits ucvtf, fadd.4s and faddp.4s. Every WORD below
// carries its disassembly in a trailing comment; keep them in sync.
//
//   sdot Vd.4S, Vn.16B, Vm.16B  =  0x4E809400 | Rm<<16 | Rn<<5 | Rd
//
// derived from the ARM ARM encoding for SDOT (vector):
//   bit31=0, bit30=Q=1, bit29=U=0, bits28..24=01110, bits23..22=size=10,
//   bit21=0, bits20..16=Rm, bits15..12=1001, bits11..10=01,
//   bits9..5=Rn, bits4..0=Rd
//
// SDOT is FEAT_DotProd: optional in ARMv8.2, mandatory from ARMv8.4. Executing
// it on a part without it is a SIGILL, so NOTHING in this file may run until
// hasDotProd (dotprod_arm64.go) has confirmed the CPU implements it. That
// contract is enforced in exactly one place: the q4kDotAsmOK-style init
// variables in q8k_dot_arm64.go short-circuit on hasDotProd before calling any
// validate* function, and every kernel entry point is already guarded by its
// *DotAsmOK flag. Do not call a function from this file on any other path.
//
// This file was previously gated darwin && arm64 precisely because every Apple
// Silicon part (M1 and later, A11 and later) has dotprod, making a probe
// unnecessary there. The runtime probe replaces that gate so Graviton, Ampere,
// Snapdragon, Raspberry Pi 5 and Android also get these kernels instead of
// falling back to the portable scalar path.

#define SDOT(Vd, Vn, Vm) WORD $(0x4E809400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// func q4kQ8Dots8Asm(q *byte, q8 *int8, out *int32)
//
// Computes the 8 per-sub-block int32 dot products of ONE Q4_K block: the
// unsigned 4-bit weight quants times the int8 activations, before any scale is
// applied. q points at the block's 128 packed nibble bytes, q8 at the block's
// 256 int8 activations, out at 8 int32s.
//
// Q4_K packs sub-block s (s = 0..3) as 32 bytes at q[s*32:], where the LOW
// nibbles are the 32 quants of sub-block 2s and the HIGH nibbles are the 32
// quants of sub-block 2s+1. The activations for those two sub-blocks are
// q8[s*64:s*64+32] and q8[s*64+32:s*64+64] respectively. That is the same
// layout q4kQDots8 in kernels_arm64.s walks for the float path, so the two stay
// directly comparable.
//
// Splitting the block scales out into Go (rather than folding them in here) is
// deliberate: the 6-bit packed scale/min decode is fiddly, it is already
// written and tested once in quant_q8k_portable.go, and keeping it out of
// assembly is what makes this kernel small enough to review by eye.
TEXT ·q4kQ8Dots8Asm(SB), NOSPLIT|NOFRAME, $0-24
	MOVD q+0(FP), R0    // packed nibbles
	MOVD q8+8(FP), R1   // int8 activations
	MOVD out+16(FP), R2 // 8 x int32

	VMOVI $15, V20.B16 // low-nibble mask

	MOVD $4, R3 // four 32-byte groups, two sub-blocks each

loop:
	// 32 packed bytes -> V0, V1 (16 bytes each).
	VLD1.P 32(R0), [V0.B16, V1.B16]

	// Low nibbles: the quants of the even sub-block.
	VAND V20.B16, V0.B16, V2.B16
	VAND V20.B16, V1.B16, V3.B16

	// High nibbles: the quants of the odd sub-block. Logical shift, so the
	// result stays the unsigned 0..15 value the format defines.
	VUSHR $4, V0.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16

	// 64 activations: V6,V7 for the even sub-block, V8,V9 for the odd one.
	VLD1.P 32(R1), [V6.B16, V7.B16]
	VLD1.P 32(R1), [V8.B16, V9.B16]

	// Accumulate each sub-block into its own 4-lane int32 vector.
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16

	SDOT(16, 2, 6) // sdot v16.4s, v2.16b, v6.16b
	SDOT(16, 3, 7) // sdot v16.4s, v3.16b, v7.16b
	SDOT(17, 4, 8) // sdot v17.4s, v4.16b, v8.16b
	SDOT(17, 5, 9) // sdot v17.4s, v5.16b, v9.16b

	// Horizontal-reduce each accumulator to one int32 and store the pair.
	// The quants are unsigned 0..15 and the activations are int8, so the
	// products fit int16 and 32 of them fit int32 with room to spare; SDOT's
	// own int32 lanes cannot overflow here either.
	VADDV V16.S4, V18
	VADDV V17.S4, V19
	FMOVS F18, R4
	FMOVS F19, R5
	MOVW R4, (R2)
	MOVW R5, 4(R2)
	ADD  $8, R2

	SUB  $1, R3
	CBNZ R3, loop
	RET

// Q5MERGE_* fold Q5_K's fifth bit into a 4-bit quant. The bit sits at position
// k of the qh byte and has to land at position 4 (value 16), so it is moved
// straight there and masked with 0x10 — two instructions instead of the
// shift-right, mask-bit-0, shift-left-4 sequence. V21 holds 0x10.
//
// Vtmp is clobbered.
#define Q5MERGE_SHL(Vdst, Vsrc, N, Vtmp) \
	VSHL $N, Vsrc.B16, Vtmp.B16     \
	VAND V21.B16, Vtmp.B16, Vtmp.B16 \
	VORR Vtmp.B16, Vdst.B16, Vdst.B16

#define Q5MERGE_SHR(Vdst, Vsrc, N, Vtmp) \
	VUSHR $N, Vsrc.B16, Vtmp.B16    \
	VAND V21.B16, Vtmp.B16, Vtmp.B16 \
	VORR Vtmp.B16, Vdst.B16, Vdst.B16

#define Q5MERGE_NONE(Vdst, Vsrc, Vtmp) \
	VAND V21.B16, Vsrc.B16, Vtmp.B16 \
	VORR Vtmp.B16, Vdst.B16, Vdst.B16

// Q5DOTS reduces the two accumulators for one group and stores the pair.
#define Q5DOTS(OFF) \
	VEOR V26.B16, V26.B16, V26.B16 \
	VEOR V27.B16, V27.B16, V27.B16 \
	SDOT(26, 2, 6)                 \
	SDOT(26, 3, 7)                 \
	SDOT(27, 4, 8)                 \
	SDOT(27, 5, 9)                 \
	VADDV V26.S4, V28              \
	VADDV V27.S4, V29              \
	FMOVS F28, R4                  \
	FMOVS F29, R5                  \
	MOVW R4, OFF(R3)               \
	MOVW R5, (OFF+4)(R3)

// func q5kQ8Dots8Asm(q *byte, qh *byte, q8 *int8, out *int32)
//
// Computes the 8 per-sub-block int32 dot products of ONE Q5_K block: the
// unsigned 5-bit weight quants times the int8 activations, before any scale is
// applied. q points at the block's 128 packed nibble bytes, qh at its 32-byte
// fifth-bit plane, q8 at 256 int8 activations, out at 8 int32s.
//
// Q5_K is Q4_K plus one bitplane. The nibble layout is identical: 32 bytes at
// q[s*32:] hold sub-block 2s in the low nibbles and 2s+1 in the high nibbles.
// The fifth bit of each quant lives in qh, which is indexed by the element
// index l ALONE and reused across all four groups — sub-block 2s takes bit 2s of
// qh[l] and sub-block 2s+1 takes bit 2s+1. That is why qh is loaded once,
// outside the group sequence.
//
// The groups are unrolled rather than looped because the qh bit position is a
// different immediate each time, and a vector shift by a register would cost
// more than the unroll saves. The bit is 0..15 in the nibble, so OR-ing 16 into
// it cannot carry, and the resulting 0..31 quant is still positive as an int8,
// which is what lets SDOT's signed interpretation be correct here.
TEXT ·q5kQ8Dots8Asm(SB), NOSPLIT|NOFRAME, $0-32
	MOVD q+0(FP), R0
	MOVD qh+8(FP), R1
	MOVD q8+16(FP), R2
	MOVD out+24(FP), R3

	VMOVI $15, V20.B16 // low-nibble mask
	VMOVI $16, V21.B16 // fifth-bit destination mask (0x10)

	// The whole fifth-bit plane, reused by all four groups.
	VLD1 (R1), [V24.B16, V25.B16]

	// ---- group 0: qh bits 0 (low) and 1 (high) ----
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VAND  V20.B16, V0.B16, V2.B16
	VAND  V20.B16, V1.B16, V3.B16
	VUSHR $4, V0.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16
	Q5MERGE_SHL(V2, V24, 4, V16)
	Q5MERGE_SHL(V3, V25, 4, V17)
	Q5MERGE_SHL(V4, V24, 3, V18)
	Q5MERGE_SHL(V5, V25, 3, V19)
	VLD1.P 32(R2), [V6.B16, V7.B16]
	VLD1.P 32(R2), [V8.B16, V9.B16]
	Q5DOTS(0)

	// ---- group 1: qh bits 2 and 3 ----
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VAND  V20.B16, V0.B16, V2.B16
	VAND  V20.B16, V1.B16, V3.B16
	VUSHR $4, V0.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16
	Q5MERGE_SHL(V2, V24, 2, V16)
	Q5MERGE_SHL(V3, V25, 2, V17)
	Q5MERGE_SHL(V4, V24, 1, V18)
	Q5MERGE_SHL(V5, V25, 1, V19)
	VLD1.P 32(R2), [V6.B16, V7.B16]
	VLD1.P 32(R2), [V8.B16, V9.B16]
	Q5DOTS(8)

	// ---- group 2: qh bits 4 (already in place) and 5 ----
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VAND  V20.B16, V0.B16, V2.B16
	VAND  V20.B16, V1.B16, V3.B16
	VUSHR $4, V0.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16
	Q5MERGE_NONE(V2, V24, V16)
	Q5MERGE_NONE(V3, V25, V17)
	Q5MERGE_SHR(V4, V24, 1, V18)
	Q5MERGE_SHR(V5, V25, 1, V19)
	VLD1.P 32(R2), [V6.B16, V7.B16]
	VLD1.P 32(R2), [V8.B16, V9.B16]
	Q5DOTS(16)

	// ---- group 3: qh bits 6 and 7 ----
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VAND  V20.B16, V0.B16, V2.B16
	VAND  V20.B16, V1.B16, V3.B16
	VUSHR $4, V0.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16
	Q5MERGE_SHR(V2, V24, 2, V16)
	Q5MERGE_SHR(V3, V25, 2, V17)
	Q5MERGE_SHR(V4, V24, 3, V18)
	Q5MERGE_SHR(V5, V25, 3, V19)
	VLD1.P 32(R2), [V6.B16, V7.B16]
	VLD1.P 32(R2), [V8.B16, V9.B16]
	Q5DOTS(24)
	RET

// func q6kQ8Dots16Asm(ql *byte, qh *byte, q8 *int8, out *int32)
//
// Computes the 16 per-scale-group int32 dot products of ONE Q6_K block: the
// unsigned 6-bit weight quants times the int8 activations, before any scale or
// the -32 offset is applied. ql points at 128 bytes, qh at 64, q8 at the
// block's 256 int8 activations, out at 16 int32s.
//
// Q6_K splits each 6-bit quant across two planes: the low 4 bits live in ql and
// the high 2 bits are one of four 2-bit fields in the corresponding qh byte.
// Per half (128 activations), for l in 0..31:
//
//	q1 = (ql[l]    & 0x0f) | ((qh[l] >> 0 & 3) << 4)  against q8[l]
//	q2 = (ql[l+32] & 0x0f) | ((qh[l] >> 2 & 3) << 4)  against q8[32+l]
//	q3 = (ql[l]    >>   4) | ((qh[l] >> 4 & 3) << 4)  against q8[64+l]
//	q4 = (ql[l+32] >>   4) | ((qh[l] >> 6 & 3) << 4)  against q8[96+l]
//
// and the scale that applies is sc[half*8 + group*2 + l/16], where group is
// 0..3 for q1..q4. So each output is a dot over exactly 16 elements — one
// 128-bit SDOT — and out is indexed half*8 + group*2 + (l/16), which is exactly
// the scale index. That is what lets the Go side just walk sc[0..15] against
// out[0..15] in one loop.
//
// The quants are unsigned 0..63, so they are non-negative as int8 and SDOT's
// signed interpretation is the same value. The -32 offset is NOT applied here;
// it is folded into the separate float offTerm over xsums, matching the
// portable kernel.

// The built quant always lands in V16 and the scratch is always V17, so the
// macros hardcode them: SDOT needs its operands as bare register NUMBERS (it is
// a WORD expression), while VAND/VORR need them as register NAMES, and a macro
// parameter cannot be both.
//
// Q6LO builds the low-nibble quant: V16 = (Vql & 0x0f) | ((Vqh & 0x03) << 4).
// Vqh must already be shifted right so the wanted 2-bit field sits at bit 0.
#define Q6LO(Vql, Vqh) \
	VAND V20.B16, Vql.B16, V16.B16 \
	VAND V21.B16, Vqh.B16, V17.B16 \
	VSHL $4, V17.B16, V17.B16      \
	VORR V17.B16, V16.B16, V16.B16

// Q6HI is the same for the high nibble: V16 = (Vql >> 4) | ((Vqh & 0x03) << 4).
#define Q6HI(Vql, Vqh) \
	VUSHR $4, Vql.B16, V16.B16     \
	VAND V21.B16, Vqh.B16, V17.B16 \
	VSHL $4, V17.B16, V17.B16      \
	VORR V17.B16, V16.B16, V16.B16

// Q6EMIT dots the 16 quants in V16 against the 16 activations in vector number
// ACT and stores the reduced int32 at offset OFF from R3.
#define Q6EMIT(ACT, OFF) \
	VEOR V18.B16, V18.B16, V18.B16 \
	SDOT(18, 16, ACT)              \
	VADDV V18.S4, V19              \
	FMOVS F19, R5                  \
	MOVW R5, OFF(R3)

TEXT ·q6kQ8Dots16Asm(SB), NOSPLIT|NOFRAME, $0-32
	MOVD ql+0(FP), R0
	MOVD qh+8(FP), R1
	MOVD q8+16(FP), R2
	MOVD out+24(FP), R3

	VMOVI $15, V20.B16 // low-nibble mask
	VMOVI $3, V21.B16  // 2-bit high-plane mask

	MOVD $2, R4 // two halves

half_loop:
	// ql[0:32] -> V0,V1 (the l and l+16 positions); ql[32:64] -> V2,V3.
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VLD1.P 32(R0), [V2.B16, V3.B16]
	// qh[0:32] -> V4,V5.
	VLD1.P 32(R1), [V4.B16, V5.B16]
	// 128 activations -> V6..V13, in the order q1,q2,q3,q4 consume them.
	VLD1.P 32(R2), [V6.B16, V7.B16]
	VLD1.P 32(R2), [V8.B16, V9.B16]
	VLD1.P 32(R2), [V10.B16, V11.B16]
	VLD1.P 32(R2), [V12.B16, V13.B16]

	// group 0 (q1): low nibble of ql[l], qh field at bit 0.
	Q6LO(V0, V4)
	Q6EMIT(6, 0)
	Q6LO(V1, V5)
	Q6EMIT(7, 4)

	// group 1 (q2): low nibble of ql[l+32], qh field at bit 2.
	VUSHR $2, V4.B16, V22.B16
	VUSHR $2, V5.B16, V23.B16
	Q6LO(V2, V22)
	Q6EMIT(8, 8)
	Q6LO(V3, V23)
	Q6EMIT(9, 12)

	// group 2 (q3): high nibble of ql[l], qh field at bit 4.
	VUSHR $4, V4.B16, V22.B16
	VUSHR $4, V5.B16, V23.B16
	Q6HI(V0, V22)
	Q6EMIT(10, 16)
	Q6HI(V1, V23)
	Q6EMIT(11, 20)

	// group 3 (q4): high nibble of ql[l+32], qh field at bit 6.
	VUSHR $6, V4.B16, V22.B16
	VUSHR $6, V5.B16, V23.B16
	Q6HI(V2, V22)
	Q6EMIT(12, 24)
	Q6HI(V3, V23)
	Q6EMIT(13, 28)

	ADD  $32, R3 // 8 int32s written
	SUB  $1, R4
	CBNZ R4, half_loop
	RET

// LEGDOTS reduces the two nibble-half accumulators into one int32 and stores it.
// The legacy 32-element formats produce a single dot per block, unlike the
// K-quants' two, because they carry one scale rather than packed sub-scales.
#define LEGDOTS \
	VEOR V16.B16, V16.B16, V16.B16 \
	SDOT(16, 2, 6)                 \
	SDOT(16, 3, 7)                 \
	VADDV V16.S4, V18              \
	FMOVS F18, R4                  \
	MOVW R4, (R2)                  \
	ADD $4, R2

// func q4_0Q8Dots8Asm(row *byte, q8 *int8, out *int32)
//
// Computes the 8 per-block int32 dot products of ONE 256-element Q4_0
// superchunk. row points at 8 consecutive 18-byte blocks, each an f16 scale
// followed by 16 packed nibble bytes; q8 at 256 int8 activations; out at 8
// int32s.
//
// Q4_0 splits its 32 quants the same way Q4_K splits a sub-block pair: the low
// nibbles of the 16 bytes are quants 0..15 and the high nibbles are quants
// 16..31, against the first and second half of the block's 32 activations. The
// quants are unsigned 0..15 here; Q4_0's -8 bias is not applied per element but
// carried exactly in the float xsums term, which is why this kernel returns raw
// unsigned dots.
TEXT ·q4_0Q8Dots8Asm(SB), NOSPLIT|NOFRAME, $0-24
	MOVD row+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD out+16(FP), R2

	VMOVI $15, V20.B16
	MOVD  $8, R3

q4_0_loop:
	// Step over the f16 scale; the post-indexed load takes the 16 nibble bytes
	// and leaves R0 exactly 18 further on.
	ADD    $2, R0
	VLD1.P 16(R0), [V0.B16]
	VAND   V20.B16, V0.B16, V2.B16
	VUSHR  $4, V0.B16, V3.B16
	VLD1.P 32(R1), [V6.B16, V7.B16]
	LEGDOTS

	SUB  $1, R3
	CBNZ R3, q4_0_loop
	RET

// func q4_1Q8Dots8Asm(row *byte, q8 *int8, out *int32)
//
// The Q4_1 analogue: identical nibble layout, but each block carries both a
// scale and a minimum, so the header is 4 bytes and the stride is 20.
TEXT ·q4_1Q8Dots8Asm(SB), NOSPLIT|NOFRAME, $0-24
	MOVD row+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD out+16(FP), R2

	VMOVI $15, V20.B16
	MOVD  $8, R3

q4_1_loop:
	ADD    $4, R0 // f16 scale plus f16 minimum
	VLD1.P 16(R0), [V0.B16]
	VAND   V20.B16, V0.B16, V2.B16
	VUSHR  $4, V0.B16, V3.B16
	VLD1.P 32(R1), [V6.B16, V7.B16]
	LEGDOTS

	SUB  $1, R3
	CBNZ R3, q4_1_loop
	RET

// func mxfp4Q8Dots8Asm(row *byte, q8 *int8, lut *int8, out *int32)
//
// Computes the 8 per-block int32 dot products of ONE 256-element MXFP4
// (gpt-oss) superchunk. row points at 8 consecutive 17-byte blocks, each 16
// packed nibble bytes followed by one raw exponent byte; q8 at 256 int8
// activations; lut at the 16-entry table of DOUBLED signed values; out at 8
// int32s.
//
// Two things make MXFP4 different from the other nibble formats:
//
// First, the nibble is not a magnitude but an index into a table of signed
// values {0,1,2,3,4,6,8,12} and their negations. VTBL does that lookup directly
// — a 16-byte table indexed by 16 nibbles in one instruction — which is why the
// table is passed in rather than the values being reconstructed arithmetically.
// The table holds twice each value so the lookup stays integral; the factor 0.5
// is applied once per block in Go.
//
// Second, the pairing is interleaved, not split in halves: the low nibble of
// byte i multiplies activation 2i and the high nibble multiplies 2i+1. So the
// activations are de-interleaved with UZP1/UZP2 (evens and odds) to line them up
// against the low- and high-nibble vectors, rather than loaded as two halves.
TEXT ·mxfp4Q8Dots8Asm(SB), NOSPLIT|NOFRAME, $0-32
	MOVD row+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD lut+16(FP), R5
	MOVD out+24(FP), R2

	VMOVI $15, V20.B16
	VLD1  (R5), [V21.B16] // doubled-value table, reused by every block
	MOVD  $8, R3

mxfp4_loop:
	// 16 packed bytes; the trailing exponent byte is skipped after the load.
	VLD1.P 16(R0), [V0.B16]
	ADD    $1, R0

	VAND  V20.B16, V0.B16, V4.B16 // low nibbles  -> indices
	VUSHR $4, V0.B16, V5.B16      // high nibbles -> indices

	// Table lookup turns each index into its doubled signed value.
	VTBL V4.B16, [V21.B16], V2.B16
	VTBL V5.B16, [V21.B16], V3.B16

	// 32 activations, then split into evens (for the low nibbles) and odds.
	VLD1.P 32(R1), [V8.B16, V9.B16]
	VUZP1  V9.B16, V8.B16, V6.B16
	VUZP2  V9.B16, V8.B16, V7.B16

	LEGDOTS

	SUB  $1, R3
	CBNZ R3, mxfp4_loop
	RET

// func q8_0Q8Dots8Asm(row *byte, q8 *int8, out *int32)
//
// Computes the 8 per-block int32 dot products of ONE 256-element Q8_0
// superchunk. row points at the first of 8 consecutive 34-byte blocks, each an
// f16 scale followed by 32 int8 weights; q8 points at the 256 int8 activations;
// out at 8 int32s.
//
// Q8_0 is the cheapest of these kernels because the weights are already int8:
// there is no nibble unpack and no bitplane, just the stride-34 walk that steps
// over each block's scale. The scales stay in Go, which combines them with the
// per-superchunk activation scale.
//
// The float path this replaces has no NEON kernel on arm64 at all, so unlike
// Q4_K and Q6_K this is a straight win rather than a defence of one.
TEXT ·q8_0Q8Dots8Asm(SB), NOSPLIT|NOFRAME, $0-24
	MOVD row+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD out+16(FP), R2

	MOVD $8, R3

q8_0_loop:
	// Step over this block's f16 scale; the post-indexed load then advances R0
	// by the 32 weight bytes, so each iteration consumes exactly 34.
	ADD    $2, R0
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VLD1.P 32(R1), [V2.B16, V3.B16]

	VEOR V16.B16, V16.B16, V16.B16
	SDOT(16, 0, 2) // sdot v16.4s, v0.16b, v2.16b
	SDOT(16, 1, 3) // sdot v16.4s, v1.16b, v3.16b

	VADDV V16.S4, V18
	FMOVS F18, R4
	MOVW  R4, (R2)
	ADD   $4, R2

	SUB  $1, R3
	CBNZ R3, q8_0_loop
	RET

// func dotInt8Asm(a, b *int8, n int) int32
//
// Straight int8 dot product of two n-element vectors, n a multiple of 16. This
// is the Q8_0 shape (weights already int8) and the building block the tests use
// to pin SDOT's behaviour on its own, independent of any block format.
TEXT ·dotInt8Asm(SB), NOSPLIT|NOFRAME, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2

	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16

	CMP  $32, R2
	BLT  tail
loop32:
	VLD1.P 32(R0), [V0.B16, V1.B16]
	VLD1.P 32(R1), [V2.B16, V3.B16]
	SDOT(16, 0, 2) // sdot v16.4s, v0.16b, v2.16b
	SDOT(17, 1, 3) // sdot v17.4s, v1.16b, v3.16b
	SUB  $32, R2
	CMP  $32, R2
	BGE  loop32

tail:
	CBZ  R2, reduce
loop16:
	VLD1.P 16(R0), [V0.B16]
	VLD1.P 16(R1), [V2.B16]
	SDOT(16, 0, 2) // sdot v16.4s, v0.16b, v2.16b
	SUB  $16, R2
	CBNZ R2, loop16

reduce:
	VADD  V17.S4, V16.S4, V16.S4
	VADDV V16.S4, V18
	FMOVS F18, R4
	MOVW  R4, ret+24(FP)
	RET

// Q2SUB0/Q2SUB build one 16-element Q2_K sub-block into V16 as unsigned
// codes 0..3. VUSHR cannot encode a zero shift, so shift 0 gets its own
// macro rather than a degenerate VUSHR $0.
#define Q2SUB0(Vq) \
	VAND V20.B16, Vq.B16, V16.B16

#define Q2SUB(Vq, SHIFT) \
	VUSHR $SHIFT, Vq.B16, V16.B16 \
	VAND V20.B16, V16.B16, V16.B16

// Q2EMIT dots the sub-block in V16 against the next 16 activations and
// appends the int32 result. SDOT accumulates, so V18 is cleared first.
#define Q2EMIT \
	VLD1.P 16(R1), [V6.B16]        \
	VEOR V18.B16, V18.B16, V18.B16 \
	SDOT(18, 16, 6)                \
	VADDV V18.S4, V19              \
	FMOVS F19, R5                  \
	MOVW.P R5, 4(R2)

// func q2kQ8Dots16Asm(qs *byte, q8 *int8, out *int32)
//
// Computes the 16 per-sub-block int32 dot products of ONE Q2_K block: the
// unsigned 2-bit codes against the int8 activations, before any scale or min
// is applied. qs points at the block's 64 packed quant bytes (block[16:80]),
// q8 at the block's 256 activations, out at 16 int32s.
//
// Q2_K packs 16 sub-blocks of 16 elements as two 32-byte chunks; within a
// chunk, element l of sub-block (shift/2)*2 + half is
// (qs[chunk*32 + half*16 + l] >> shift) & 3 for shift 0/2/4/6. Sub-block
// index therefore advances 0..15 in exactly the order emitted below, which is
// why both the activation pointer and the output pointer can simply
// post-increment.
//
// The codes are non-negative, so SDOT's signed interpretation is safe without
// any bias fixup — unlike Q3_K below. The scale/min nibbles and the dmin term
// stay in Go (see q2kDotQ8KRow), matching every other kernel in this file.
TEXT ·q2kQ8Dots16Asm(SB), NOSPLIT|NOFRAME, $0-24
	MOVD qs+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD out+16(FP), R2

	VMOVI $3, V20.B16 // 2-bit code mask

	MOVD $2, R4 // two 32-byte chunks

q2k_chunk:
	VLD1.P 32(R0), [V0.B16, V1.B16] // V0 = half 0, V1 = half 1

	Q2SUB0(V0)
	Q2EMIT
	Q2SUB0(V1)
	Q2EMIT
	Q2SUB(V0, 2)
	Q2EMIT
	Q2SUB(V1, 2)
	Q2EMIT
	Q2SUB(V0, 4)
	Q2EMIT
	Q2SUB(V1, 4)
	Q2EMIT
	Q2SUB(V0, 6)
	Q2EMIT
	Q2SUB(V1, 6)
	Q2EMIT

	SUB  $1, R4
	CBNZ R4, q2k_chunk
	RET

// Q3EMIT is Q2EMIT with Q3_K's register assignment: this kernel needs a
// fourth pointer for the hmask plane, so the activations are in R2 and the
// output in R3 rather than R1/R2.
#define Q3EMIT \
	VLD1.P 16(R2), [V6.B16]        \
	VEOR V18.B16, V18.B16, V18.B16 \
	SDOT(18, 16, 6)                \
	VADDV V18.S4, V19              \
	FMOVS F19, R5                  \
	MOVW.P R5, 4(R3)

// Q3SUB* build one 16-element Q3_K sub-block into V16 as SIGNED codes -4..3.
//
// The quant is v = ((q >> shift) & 3) - 4*(1 - h), h being this element's bit
// HBIT of the hmask plane. Rearranged, v = (u | 4h) - 4: u occupies bits 0-1
// and 4h bit 2, so they are disjoint and VORR does the work of an add. The
// result is a signed int8, which is exactly what SDOT wants — so on NEON the
// bias needs no activation-sum term at all, unlike the AVX2 kernel where
// VPMADDUBSW's unsigned operand forces the Σ(u+4h)·a − 4·Σa factorisation.
//
// Two degenerate-shift variants exist because VUSHR cannot encode $0: HBIT is
// 0 only for chunk 0 shift 0, and SHIFT is 0 once per chunk.
#define Q3SUB_S0H0(Vq, Vh) \
	VAND V20.B16, Vq.B16, V16.B16  \
	VAND V21.B16, Vh.B16, V17.B16  \
	VSHL $2, V17.B16, V17.B16      \
	VORR V17.B16, V16.B16, V16.B16 \
	VSUB V22.B16, V16.B16, V16.B16

#define Q3SUB_S0(Vq, Vh, HBIT) \
	VAND V20.B16, Vq.B16, V16.B16  \
	VUSHR $HBIT, Vh.B16, V17.B16   \
	VAND V21.B16, V17.B16, V17.B16 \
	VSHL $2, V17.B16, V17.B16      \
	VORR V17.B16, V16.B16, V16.B16 \
	VSUB V22.B16, V16.B16, V16.B16

#define Q3SUB(Vq, Vh, SHIFT, HBIT) \
	VUSHR $SHIFT, Vq.B16, V16.B16  \
	VAND V20.B16, V16.B16, V16.B16 \
	VUSHR $HBIT, Vh.B16, V17.B16   \
	VAND V21.B16, V17.B16, V17.B16 \
	VSHL $2, V17.B16, V17.B16      \
	VORR V17.B16, V16.B16, V16.B16 \
	VSUB V22.B16, V16.B16, V16.B16

// func q3kQ8Dots16Asm(qs *byte, hmask *byte, q8 *int8, out *int32)
//
// Computes the 16 per-sub-block int32 dot products of ONE Q3_K block, with
// the per-element high-bit bias already applied, so Go only has to apply the
// six-bit scales (dl = scale - 32) and the block scale. qs points at the 64
// packed quant bytes (block[32:96]), hmask at the 32-byte bit-plane
// (block[0:32]), q8 at the 256 activations, out at 16 int32s.
//
// The quant packing is Q2_K's, so the sub-block walk is identical; what
// differs is that each of the eight (chunk, shift) pairs takes a different
// BIT of the same 32-byte hmask plane — bit chunk*4 + shift/2, continuing
// across the chunk boundary exactly as the portable kernel's m <<= 1 does.
// The plane is therefore loaded once, without post-increment, and reused.
TEXT ·q3kQ8Dots16Asm(SB), NOSPLIT|NOFRAME, $0-32
	MOVD qs+0(FP), R0
	MOVD hmask+8(FP), R1
	MOVD q8+16(FP), R2
	MOVD out+24(FP), R3

	VMOVI $3, V20.B16 // 2-bit code mask
	VMOVI $1, V21.B16 // hmask bit mask
	VMOVI $4, V22.B16 // the -4 bias

	VLD1 (R1), [V4.B16, V5.B16]     // hmask: V4 = half 0, V5 = half 1
	VLD1.P 32(R0), [V0.B16, V1.B16] // qs chunk 0
	VLD1.P 32(R0), [V2.B16, V3.B16] // qs chunk 1

	Q3SUB_S0H0(V0, V4)
	Q3EMIT
	Q3SUB_S0H0(V1, V5)
	Q3EMIT
	Q3SUB(V0, V4, 2, 1)
	Q3EMIT
	Q3SUB(V1, V5, 2, 1)
	Q3EMIT
	Q3SUB(V0, V4, 4, 2)
	Q3EMIT
	Q3SUB(V1, V5, 4, 2)
	Q3EMIT
	Q3SUB(V0, V4, 6, 3)
	Q3EMIT
	Q3SUB(V1, V5, 6, 3)
	Q3EMIT

	Q3SUB_S0(V2, V4, 4)
	Q3EMIT
	Q3SUB_S0(V3, V5, 4)
	Q3EMIT
	Q3SUB(V2, V4, 2, 5)
	Q3EMIT
	Q3SUB(V3, V5, 2, 5)
	Q3EMIT
	Q3SUB(V2, V4, 4, 6)
	Q3EMIT
	Q3SUB(V3, V5, 4, 6)
	Q3EMIT
	Q3SUB(V2, V4, 6, 7)
	Q3EMIT
	Q3SUB(V3, V5, 6, 7)
	Q3EMIT

	RET
