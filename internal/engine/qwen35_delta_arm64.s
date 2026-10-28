//go:build arm64

#include "textflag.h"
#include "arm64_fp_compat.h"

// A head owns an n*n state matrix; n is a positive multiple of 16.
// Fuse Scale+Dot, then Axpy+Dot for each row while it is still hot. FMUL
// rounds decay separately and the four dot accumulators/reduction match
// dotF32 exactly. The delta subtraction and beta multiplication also stay
// separate. Only the existing Axpy and dot products use fused arithmetic.
// func qwen35DeltaHeadNEON(state, q, k, v, out *float32, decay, beta float32, n int)
TEXT ·qwen35DeltaHeadNEON(SB), NOSPLIT|NOFRAME, $0-56
	MOVD state+0(FP), R0
	MOVD q+8(FP), R1
	MOVD k+16(FP), R2
	MOVD v+24(FP), R3
	MOVD out+32(FP), R4
	FMOVS decay+40(FP), F30
	VDUP V30.S[0], V30.S4
	FMOVS beta+44(FP), F29
	MOVD n+48(FP), R5
	MOVD R5, R6
	FMOVS $(1.0), F31
	VDUP V31.S[0], V31.S4

delta_head_row:
	MOVD R0, R7
	MOVD R2, R8
	MOVD R0, R9
	MOVD R5, R10
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
delta_head_recall:
	VLD1.P 64(R7), [V1.S4, V2.S4, V3.S4, V4.S4]
	VLD1.P 64(R8), [V5.S4, V6.S4, V7.S4, V8.S4]
	FMUL4S(1, 1, 30)
	FMUL4S(2, 2, 30)
	FMUL4S(3, 3, 30)
	FMUL4S(4, 4, 30)
	VST1.P [V1.S4, V2.S4, V3.S4, V4.S4], 64(R9)
	VFMLA V1.S4, V5.S4, V0.S4
	VFMLA V2.S4, V6.S4, V9.S4
	VFMLA V3.S4, V7.S4, V10.S4
	VFMLA V4.S4, V8.S4, V11.S4
	SUB $16, R10, R10
	CBNZ R10, delta_head_recall
	VFMLA V9.S4, V31.S4, V0.S4
	VFMLA V10.S4, V31.S4, V0.S4
	VFMLA V11.S4, V31.S4, V0.S4
	WORD $0x6e20d400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x6e20d400
	FMOVS.P 4(R3), F27
	FSUBS F0, F27, F27
	FMULS F29, F27, F27
	VDUP V27.S[0], V28.S4

	MOVD R0, R7
	MOVD R1, R8
	MOVD R0, R9
	MOVD R2, R11
	MOVD R5, R10
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
delta_head_update:
#define DELTA_UPDATE(ACC) \
	VLD1.P 16(R7), [V1.S4] \
	VLD1.P 16(R11), [V2.S4] \
	VFMLA V2.S4, V28.S4, V1.S4 \
	VST1.P [V1.S4], 16(R9) \
	VLD1.P 16(R8), [V3.S4] \
	VFMLA V1.S4, V3.S4, ACC.S4
	DELTA_UPDATE(V0)
	DELTA_UPDATE(V9)
	DELTA_UPDATE(V10)
	DELTA_UPDATE(V11)
	SUB $16, R10, R10
	CBNZ R10, delta_head_update
	VFMLA V9.S4, V31.S4, V0.S4
	VFMLA V10.S4, V31.S4, V0.S4
	VFMLA V11.S4, V31.S4, V0.S4
	WORD $0x6e20d400
	WORD $0x6e20d400
	FMOVS.P F0, 4(R4)
	MOVD R9, R0
	SUB $1, R6, R6
	CBNZ R6, delta_head_row
	RET
