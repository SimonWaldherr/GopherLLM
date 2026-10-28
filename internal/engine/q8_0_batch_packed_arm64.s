//go:build arm64

#include "textflag.h"

// SDOT by indexed four-byte element: each lane of N contains a different
// token's four activations and uses the same four weights from M[index].
#define SDOTEL(D,N,M,INDEX) WORD $(0x4F80E000 | (D) | ((N)<<5) | ((M)<<16) | (((INDEX)&2)<<10) | (((INDEX)&1)<<21))

// Pack a four-by-four matrix of 32-bit words. Each word contains four int8
// activations; the four output vectors put the token index in the SIMD lane.
// func q8_0Pack4Asm(src, dst *int8, cols int)
TEXT ·q8_0Pack4Asm(SB), NOSPLIT|NOFRAME, $0-24
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R4
	MOVD cols+16(FP), R5
	ADD R5, R0, R1
	ADD R5, R1, R2
	ADD R5, R2, R3
	LSR $4, R5
	CBZ R5, pack_done
pack_loop:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R2), [V2.S4]
	VLD1.P 16(R3), [V3.S4]
	VZIP1 V1.S4, V0.S4, V4.S4
	VZIP2 V1.S4, V0.S4, V5.S4
	VZIP1 V3.S4, V2.S4, V6.S4
	VZIP2 V3.S4, V2.S4, V7.S4
	VZIP1 V6.D2, V4.D2, V0.D2
	VZIP2 V6.D2, V4.D2, V1.D2
	VZIP1 V7.D2, V5.D2, V2.D2
	VZIP2 V7.D2, V5.D2, V3.D2
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R4)
	SUB $1, R5
	CBNZ R5, pack_loop
pack_done:
	RET

// Four partial sums hide SDOT latency. Integer addition can be reassociated
// exactly: a full 32-element int8 dot fits comfortably inside int32. D keeps
// one complete dot per token lane, ready for the unchanged float conversion.
#define Q8PACKDOT(D,V,F,SCALEV) \
	MOVHU (R0), R6 \
	FMOVS (R3)(R6<<2), F \
	VDUP SCALEV.S[0], SCALEV.S4 \
	ADD $2, R0 \
	VLD1.P 32(R0), [V0.B16, V1.B16] \
	VLD1.P 64(R1), [V2.B16, V3.B16, V4.B16, V5.B16] \
	VLD1.P 64(R1), [V6.B16, V7.B16, V8.B16, V9.B16] \
	VEOR V10.B16, V10.B16, V10.B16 \
	VEOR V11.B16, V11.B16, V11.B16 \
	VEOR V12.B16, V12.B16, V12.B16 \
	VEOR V13.B16, V13.B16, V13.B16 \
	SDOTEL(10,2,0,0) \
	SDOTEL(11,3,0,1) \
	SDOTEL(12,4,0,2) \
	SDOTEL(13,5,0,3) \
	SDOTEL(10,6,1,0) \
	SDOTEL(11,7,1,1) \
	SDOTEL(12,8,1,2) \
	SDOTEL(13,9,1,3) \
	VADD V11.S4, V10.S4, V10.S4 \
	VADD V13.S4, V12.S4, V12.S4 \
	VADD V12.S4, V10.S4, V.S4 \
	WORD $(0x4e21d800 | ((D)<<5) | (D))

// func q8_0Rows4PackedAsm(row *byte, packed *int8, scales, lut *float32, blocks int, out *float32)
TEXT ·q8_0Rows4PackedAsm(SB), NOSPLIT|NOFRAME, $0-48
	MOVD row+0(FP), R0
	MOVD packed+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD lut+24(FP), R3
	MOVD blocks+32(FP), R4
	MOVD out+40(FP), R5
	LSL $2, R4, R8
	VEOR V15.B16, V15.B16, V15.B16
	CBZ R4, packed_done
packed_block:
	Q8PACKDOT(16,V16,F24,V24)
	Q8PACKDOT(17,V17,F25,V25)
	Q8PACKDOT(18,V18,F26,V26)
	Q8PACKDOT(19,V19,F27,V27)
	Q8PACKDOT(20,V20,F28,V28)
	Q8PACKDOT(21,V21,F29,V29)
	Q8PACKDOT(22,V22,F30,V30)
	Q8PACKDOT(23,V23,F31,V31)
	VEOR V14.B16, V14.B16, V14.B16
	VFMLA V16.S4, V24.S4, V14.S4
	VFMLA V17.S4, V25.S4, V14.S4
	VFMLA V18.S4, V26.S4, V14.S4
	VFMLA V19.S4, V27.S4, V14.S4
	VFMLA V20.S4, V28.S4, V14.S4
	VFMLA V21.S4, V29.S4, V14.S4
	VFMLA V22.S4, V30.S4, V14.S4
	VFMLA V23.S4, V31.S4, V14.S4
	FMOVS (R2), F24
	ADD R8, R2, R7
	FMOVS (R7), F25
	ADD R8, R7, R7
	FMOVS (R7), F26
	ADD R8, R7, R7
	FMOVS (R7), F27
	VZIP1 V25.S4, V24.S4, V24.S4
	VZIP1 V27.S4, V26.S4, V26.S4
	VZIP1 V26.D2, V24.D2, V24.D2
	VFMLA V14.S4, V24.S4, V15.S4
	ADD $4, R2
	SUB $1, R4
	CBNZ R4, packed_block
packed_done:
	VST1 [V15.S4], (R5)
	RET
