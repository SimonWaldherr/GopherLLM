//go:build arm64

#include "textflag.h"
#include "arm64_fp_compat.h"

#define SDOT(D,N,M) WORD $(0x4E809400 | ((M)<<16) | ((N)<<5) | (D))

#define BATCHQ6LO(LO, HI, SHIFT) \
	VAND V20.B16, LO.B16, V16.B16 \
	VSHL $SHIFT, HI.B16, V17.B16 \
	VAND V21.B16, V17.B16, V17.B16 \
	VORR V17.B16, V16.B16, V16.B16

#define BATCHQ6HI(LO, HI) \
	VUSHR $4, LO.B16, V16.B16 \
	VAND V21.B16, HI.B16, V17.B16 \
	VORR V17.B16, V16.B16, V16.B16

#define BATCHQ6HILAST(LO, HI) \
	VUSHR $4, LO.B16, V16.B16 \
	VUSHR $2, HI.B16, V17.B16 \
	VAND V21.B16, V17.B16, V17.B16 \
	VORR V17.B16, V16.B16, V16.B16

// The same 16 quants in V16 feed four independent token dots. Pack their
// pairwise integer reductions into four lanes, then apply the shared signed
// scale. Neither the int32 dots nor their signed scale products overflow int32.
// V26 holds four offsets; each lane follows the original 16-step FMA order.
#define BATCHQ6DOTS() \
	VLD1.P 16(R1), [V6.B16] \
	VLD1.P 16(R2), [V7.B16] \
	VLD1.P 16(R3), [V8.B16] \
	VLD1.P 16(R4), [V9.B16] \
	VEOR V22.B16, V22.B16, V22.B16 \
	VEOR V23.B16, V23.B16, V23.B16 \
	VEOR V24.B16, V24.B16, V24.B16 \
	VEOR V25.B16, V25.B16, V25.B16 \
	SDOT(22, 16, 6) \
	SDOT(23, 16, 7) \
	SDOT(24, 16, 8) \
	SDOT(25, 16, 9) \
	VADDP V23.S4, V22.S4, V22.S4 \
	VADDP V25.S4, V24.S4, V24.S4 \
	VADDP V24.S4, V22.S4, V22.S4 \
	MOVB.P 1(R16), R17 \
	VDUP R17, V27.S4 \
	MLA4S(30, 22, 27) \
	WORD $(0x4e21d800 | (27<<5) | 27) \
	VLD1.P 4(R5), V6.S[0] \
	VLD1.P 4(R6), V6.S[1] \
	VLD1.P 4(R7), V6.S[2] \
	VLD1.P 4(R8), V6.S[3] \
	VFMLA V27.S4, V6.S4, V26.S4

// Negate the offset before the first FMA, matching the scalar FNMSUBS.
// The second FMA accumulates d*blockValue into the row sum independently in
// each token lane; neither reduction is reassociated.
#define BATCHQ6FINISH() \
	VLD1.P 4(R9), V22.S[0] \
	VLD1.P 4(R10), V22.S[1] \
	VLD1.P 4(R11), V22.S[2] \
	VLD1.P 4(R12), V22.S[3] \
	FNEG4S(23, 26) \
	VFMLA V30.S4, V22.S4, V23.S4 \
	VDUP V25.S[0], V25.S4 \
	VFMLA V23.S4, V25.S4, V31.S4

// func q6kRows4Asm(row *byte, q8 *int8, scales, sums, lut *float32, cols, blocks int, out *float32)
TEXT ·q6kRows4Asm(SB), NOSPLIT|NOFRAME, $0-64
	MOVD row+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD scales+16(FP), R9
	MOVD sums+24(FP), R5
	MOVD lut+32(FP), R20
	MOVD cols+40(FP), R21
	MOVD blocks+48(FP), R19
	MOVD out+56(FP), R22
	ADD R21, R1, R2
	ADD R21, R2, R3
	ADD R21, R3, R4
	LSL $6, R19, R23
	ADD R23, R5, R6
	ADD R23, R6, R7
	ADD R23, R7, R8
	LSL $2, R19, R23
	ADD R23, R9, R10
	ADD R23, R10, R11
	ADD R23, R11, R12
	VMOVI $15, V20.B16
	VMOVI $48, V21.B16
	VEOR V31.B16, V31.B16, V31.B16
	CBZ R19, batchdone
batchblock:
	MOVD R0, R14
	ADD $128, R0, R15
	ADD $192, R0, R16
	MOVD $2, R13
	VEOR V30.B16, V30.B16, V30.B16
	VEOR V26.B16, V26.B16, V26.B16
batchhalf:
	VLD1.P 32(R14), [V0.B16, V1.B16]
	VLD1.P 32(R14), [V2.B16, V3.B16]
	VLD1.P 32(R15), [V4.B16, V5.B16]
	BATCHQ6LO(V0, V4, 4)
	BATCHQ6DOTS()
	BATCHQ6LO(V1, V5, 4)
	BATCHQ6DOTS()
	BATCHQ6LO(V2, V4, 2)
	BATCHQ6DOTS()
	BATCHQ6LO(V3, V5, 2)
	BATCHQ6DOTS()
	BATCHQ6HI(V0, V4)
	BATCHQ6DOTS()
	BATCHQ6HI(V1, V5)
	BATCHQ6DOTS()
	BATCHQ6HILAST(V2, V4)
	BATCHQ6DOTS()
	BATCHQ6HILAST(V3, V5)
	BATCHQ6DOTS()
	SUB $1, R13
	CBNZ R13, batchhalf
	MOVHU 208(R0), R17
	FMOVS (R20)(R17<<2), F25
	WORD $(0x4e21d800 | (30<<5) | 30) // scvtf v30.4s,v30.4s
	BATCHQ6FINISH()
	ADD $210, R0
	SUB $1, R19
	CBNZ R19, batchblock
batchdone:
	VST1 [V31.S4], (R22)
	RET
