//go:build arm64

#include "textflag.h"

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
// reductions into four lanes, then apply the shared signed scale. Neither
// the int32 dots nor their signed scale products can overflow int32.
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
	VADDV V22.S4, V22 \
	VADDV V23.S4, V23 \
	VADDV V24.S4, V24 \
	VADDV V25.S4, V25 \
	VZIP1 V23.S4, V22.S4, V22.S4 \
	VZIP1 V25.S4, V24.S4, V24.S4 \
	VZIP1 V24.D2, V22.D2, V22.D2 \
	MOVB.P 1(R16), R17 \
	FMOVS R17, F27 \
	VDUP V27.S[0], V27.S4 \
	WORD $(0x4ea09c00 | (27<<16) | (22<<5) | 22) \
	VADD V22.S4, V30.S4, V30.S4 \
	WORD $(0x4e21d800 | (27<<5) | 27) \
	FMOVS.P 4(R5), F6 \
	FMOVS.P 4(R6), F7 \
	FMOVS.P 4(R7), F8 \
	FMOVS.P 4(R8), F9 \
	VZIP1 V7.S4, V6.S4, V6.S4 \
	VZIP1 V9.S4, V8.S4, V8.S4 \
	VZIP1 V8.D2, V6.D2, V6.D2 \
	VFMLA V27.S4, V6.S4, V26.S4

// Preserve the existing scalar FNMSUBS + FMADDS combination, including
// signed-zero behavior, independently for each token lane.
#define BATCHQ6FINISH(SCALES, LANE) \
	FMOVS.P 4(SCALES), F22 \
	VMOV V30.S[LANE], V23.S[0] \
	VMOV V26.S[LANE], V24.S[0] \
	FNMSUBS F23, F24, F22, F22 \
	VMOV V31.S[LANE], V27.S[0] \
	FMADDS F22, F27, F25, F27 \
	VMOV V27.S[0], V31.S[LANE]

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
	BATCHQ6FINISH(R9, 0)
	BATCHQ6FINISH(R10, 1)
	BATCHQ6FINISH(R11, 2)
	BATCHQ6FINISH(R12, 3)
	ADD $210, R0
	SUB $1, R19
	CBNZ R19, batchblock
batchdone:
	VST1 [V31.S4], (R22)
	RET
