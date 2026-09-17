//go:build arm64
#include "textflag.h"
#define SDOT(D,N,M) WORD $(0x4E809400 | ((M)<<16) | ((N)<<5) | (D))

// Q6_K x Q8_K: keep the unpack, signed scales, offset and row sum in
// registers. The scalar FMAs follow q6kDotQ8KRowBlock exactly.
// Move the high plane directly into bits 4..5 before masking, avoiding
// separate field extraction and left shifts for each of the four groups.
#define ROWQ6LO(Vql, Vqh, SHIFT) \
	VAND V20.B16, Vql.B16, V16.B16 \
	VSHL $SHIFT, Vqh.B16, V17.B16 \
	VAND V21.B16, V17.B16, V17.B16 \
	VORR V17.B16, V16.B16, V16.B16

#define ROWQ6HI(Vql, Vqh) \
	VUSHR $4, Vql.B16, V16.B16 \
	VAND V21.B16, Vqh.B16, V17.B16 \
	VORR V17.B16, V16.B16, V16.B16

#define ROWQ6HILAST(Vql, Vqh) \
	VUSHR $4, Vql.B16, V16.B16 \
	VUSHR $2, Vqh.B16, V17.B16 \
	VAND V21.B16, V17.B16, V17.B16 \
	VORR V17.B16, V16.B16, V16.B16

// Pack four exact integer dots per vector; apply signed scales together.
#define ROWQ6EMIT(ACT, D, DST) \
	VEOR DST.B16, DST.B16, DST.B16 \
	SDOT(D, 16, ACT)

// Keep the offset's float reduction in the same scale-index FMA order.
#define OFFSET(SCALES, SUMS, LANE) \
	VMOV SCALES.S[LANE], V27.S[0] \
	VMOV SUMS.S[LANE], V28.S[0] \
	FMADDS F27, F26, F28, F26

// func q6kRowDecodeAsm(row *byte, q8 *int8, scales, sums, lut *float32, blocks int) float32
TEXT ·q6kRowDecodeAsm(SB), NOSPLIT|NOFRAME, $0-52
	MOVD row+0(FP), R8
	MOVD q8+8(FP), R2
	MOVD scales+16(FP), R6
	MOVD sums+24(FP), R5
	MOVD lut+32(FP), R7
	MOVD blocks+40(FP), R9
	FMOVS ZR, F31
	VMOVI $15, V20.B16
	VMOVI $48, V21.B16
	CBZ R9, rowdone
rowblock:
	MOVD R8, R0
	ADD $128, R8, R1
	ADD $192, R8, R3
	MOVD $2, R4
	VEOR V30.B16, V30.B16, V30.B16
	FMOVS ZR, F26
rowhalf:
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
	ROWQ6LO(V0, V4, 4)
	ROWQ6EMIT(6, 14, V14)
	ROWQ6LO(V1, V5, 4)
	ROWQ6EMIT(7, 15, V15)

	// group 1 (q2): low nibble of ql[l+32], qh field at bit 2.
	ROWQ6LO(V2, V4, 2)
	ROWQ6EMIT(8, 18, V18)
	ROWQ6LO(V3, V5, 2)
	ROWQ6EMIT(9, 19, V19)

	// group 2 (q3): high nibble of ql[l], qh field at bit 4.
	ROWQ6HI(V0, V4)
	ROWQ6EMIT(10, 24, V24)
	ROWQ6HI(V1, V5)
	ROWQ6EMIT(11, 25, V25)

	// group 3 (q4): high nibble of ql[l+32], qh field at bit 6.
	ROWQ6HILAST(V2, V4)
	ROWQ6EMIT(12, 27, V27)
	ROWQ6HILAST(V3, V5)
	ROWQ6EMIT(13, 28, V28)

	VADDV V14.S4, V14
	VADDV V15.S4, V15
	VADDV V18.S4, V18
	VADDV V19.S4, V19
	VADDV V24.S4, V24
	VADDV V25.S4, V25
	VADDV V27.S4, V27
	VADDV V28.S4, V28
	VMOV V15.S[0], V14.S[1]
	VMOV V18.S[0], V14.S[2]
	VMOV V19.S[0], V14.S[3]
	VMOV V24.S[0], V15.S[0]
	VMOV V25.S[0], V15.S[1]
	VMOV V27.S[0], V15.S[2]
	VMOV V28.S[0], V15.S[3]
	// Eight signed sub-scales, widened to int32 and multiplied by the dots.
	VLD1.P 8(R3), [V24.B8]
	VSXTL V24.B8, V24.H8
	VSXTL2 V24.H8, V25.S4
	VSXTL V24.H4, V24.S4
	WORD $0x4eb89dce // mul v14.4s,v14.4s,v24.4s
	WORD $0x4eb99def // mul v15.4s,v15.4s,v25.4s
	VADD V14.S4, V15.S4, V14.S4
	VADD V14.S4, V30.S4, V30.S4
	WORD $0x4e21db18 // scvtf v24.4s,v24.4s
	WORD $0x4e21db39 // scvtf v25.4s,v25.4s
	VLD1.P 32(R5), [V0.S4, V1.S4]
	OFFSET(V24, V0, 0)
	OFFSET(V24, V0, 1)
	OFFSET(V24, V0, 2)
	OFFSET(V24, V0, 3)
	OFFSET(V25, V1, 0)
	OFFSET(V25, V1, 1)
	OFFSET(V25, V1, 2)
	OFFSET(V25, V1, 3)
	SUB $1, R4
	CBNZ R4, rowhalf
	MOVHU 208(R8), R12
	FMOVS (R7)(R12<<2), F27
	FMOVS.P 4(R6), F28
	VADDV V30.S4, V29
	WORD $0x4e21dbbd // scvtf v29.4s,v29.4s
	FNMSUBS F29, F26, F28, F28
	FMADDS F28, F31, F27, F31
	ADD $210, R8
	SUB $1, R9
	CBNZ R9, rowblock
rowdone:
	FMOVS F31, ret+48(FP)
	RET
