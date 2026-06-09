#include "textflag.h"

// func axpyF32(out []float32, alpha float32, x []float32)
TEXT ·axpyF32(SB), NOSPLIT|NOFRAME, $0-56
	MOVD out_base+0(FP), R0
	MOVD out_len+8(FP), R2
	FMOVS alpha+24(FP), F31
	MOVD x_base+32(FP), R1
	MOVD x_len+40(FP), R3
	CMP R3, R2
	BLS axpy_min_done
	MOVD R3, R2

axpy_min_done:
	VDUP V31.S[0], V31.S4
	CMP $16, R2
	BLT axpy_loop4_start

axpy_loop16:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R0), [V2.S4]
	VLD1.P 16(R1), [V3.S4]
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V5.S4]
	VLD1.P 16(R0), [V6.S4]
	VLD1.P 16(R1), [V7.S4]
	VFMLA V1.S4, V31.S4, V0.S4
	VFMLA V3.S4, V31.S4, V2.S4
	VFMLA V5.S4, V31.S4, V4.S4
	VFMLA V7.S4, V31.S4, V6.S4
	SUB $64, R0, R0
	VST1.P [V0.S4], 16(R0)
	VST1.P [V2.S4], 16(R0)
	VST1.P [V4.S4], 16(R0)
	VST1.P [V6.S4], 16(R0)
	SUB $16, R2, R2
	CMP $16, R2
	BGE axpy_loop16

axpy_loop4_start:
	CMP $4, R2
	BLT axpy_tail

axpy_loop4:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VFMLA V1.S4, V31.S4, V0.S4
	SUB $16, R0, R0
	VST1.P [V0.S4], 16(R0)
	SUB $4, R2, R2
	CMP $4, R2
	BGE axpy_loop4

axpy_tail:
	CBZ R2, axpy_done
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FMULS F31, F1, F1
	FADDS F1, F0
	SUB $4, R0, R0
	FMOVS.P F0, 4(R0)
	SUB $1, R2, R2
	B axpy_tail

axpy_done:
	RET

// func scaleF32(out []float32, alpha float32)
TEXT ·scaleF32(SB), NOSPLIT|NOFRAME, $0-28
	MOVD out_base+0(FP), R0
	MOVD out_len+8(FP), R2
	FMOVS alpha+24(FP), F31
	VDUP V31.S[0], V31.S4
	VEOR V30.B16, V30.B16, V30.B16
	CMP $16, R2
	BLT scale_loop4_start

scale_loop16:
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R0), [V5.S4]
	VLD1.P 16(R0), [V6.S4]
	VLD1.P 16(R0), [V7.S4]
	VMOV V30.B16, V0.B16
	VMOV V30.B16, V1.B16
	VMOV V30.B16, V2.B16
	VMOV V30.B16, V3.B16
	VFMLA V4.S4, V31.S4, V0.S4
	VFMLA V5.S4, V31.S4, V1.S4
	VFMLA V6.S4, V31.S4, V2.S4
	VFMLA V7.S4, V31.S4, V3.S4
	SUB $64, R0, R0
	VST1.P [V0.S4], 16(R0)
	VST1.P [V1.S4], 16(R0)
	VST1.P [V2.S4], 16(R0)
	VST1.P [V3.S4], 16(R0)
	SUB $16, R2, R2
	CMP $16, R2
	BGE scale_loop16

scale_loop4_start:
	CMP $4, R2
	BLT scale_tail

scale_loop4:
	VLD1.P 16(R0), [V0.S4]
	VMOV V30.B16, V1.B16
	VFMLA V0.S4, V31.S4, V1.S4
	SUB $16, R0, R0
	VST1.P [V1.S4], 16(R0)
	SUB $4, R2, R2
	CMP $4, R2
	BGE scale_loop4

scale_tail:
	CBZ R2, scale_done
	FMOVS.P 4(R0), F0
	FMULS F31, F0, F0
	SUB $4, R0, R0
	FMOVS.P F0, 4(R0)
	SUB $1, R2, R2
	B scale_tail

scale_done:
	RET

// func scaleAddF32(out []float32, alpha float32, x []float32)
TEXT ·scaleAddF32(SB), NOSPLIT|NOFRAME, $0-56
	MOVD out_base+0(FP), R0
	MOVD out_len+8(FP), R2
	FMOVS alpha+24(FP), F31
	MOVD x_base+32(FP), R1
	MOVD x_len+40(FP), R3
	CMP R3, R2
	BLS scaleadd_min_done
	MOVD R3, R2

scaleadd_min_done:
	VDUP V31.S[0], V31.S4
	CMP $16, R2
	BLT scaleadd_loop4_start

scaleadd_loop16:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R0), [V2.S4]
	VLD1.P 16(R1), [V3.S4]
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V5.S4]
	VLD1.P 16(R0), [V6.S4]
	VLD1.P 16(R1), [V7.S4]
	VFMLA V0.S4, V31.S4, V1.S4
	VFMLA V2.S4, V31.S4, V3.S4
	VFMLA V4.S4, V31.S4, V5.S4
	VFMLA V6.S4, V31.S4, V7.S4
	SUB $64, R0, R0
	VST1.P [V1.S4], 16(R0)
	VST1.P [V3.S4], 16(R0)
	VST1.P [V5.S4], 16(R0)
	VST1.P [V7.S4], 16(R0)
	SUB $16, R2, R2
	CMP $16, R2
	BGE scaleadd_loop16

scaleadd_loop4_start:
	CMP $4, R2
	BLT scaleadd_tail

scaleadd_loop4:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VFMLA V0.S4, V31.S4, V1.S4
	SUB $16, R0, R0
	VST1.P [V1.S4], 16(R0)
	SUB $4, R2, R2
	CMP $4, R2
	BGE scaleadd_loop4

scaleadd_tail:
	CBZ R2, scaleadd_done
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FMULS F31, F0, F0
	FADDS F1, F0
	SUB $4, R0, R0
	FMOVS.P F0, 4(R0)
	SUB $1, R2, R2
	B scaleadd_tail

scaleadd_done:
	RET
