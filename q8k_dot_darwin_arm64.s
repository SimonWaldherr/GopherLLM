//go:build darwin && arm64

#include "textflag.h"

// NEON int8 dot-product kernels for the Q8K activation path on Apple Silicon.
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
// SDOT is FEAT_DotProd, mandatory from ARMv8.4. Every Apple Silicon part (M1
// and later, and A11 and later) implements it, which is why this file is gated
// on darwin && arm64 and needs no runtime feature probe. The same gate is
// already used by kv_f16_arm64.go for the same reason. Do NOT relax the tag to
// bare arm64: ARMv8.0/8.2 Linux and Android parts exist without dotprod and
// would take SIGILL.

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
