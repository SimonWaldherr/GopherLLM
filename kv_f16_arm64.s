//go:build darwin && arm64

#include "textflag.h"

// Apple Silicon f16 KV-cache kernels. Go's arm64 assembler does not yet
// expose FCVTL/FCVTL2/FCVTN/FCVTN2 mnemonics, so their architectural
// encodings are emitted with WORD. All functions process eight elements per
// iteration; the Go wrappers handle tails.

// func dotF32F16NEON(a []float32, b []uint16) float32
TEXT ·dotF32F16NEON(SB), NOSPLIT|NOFRAME, $0-52
	MOVD a_base+0(FP), R0
	MOVD a_len+8(FP), R2
	MOVD b_base+24(FP), R1
	MOVD b_len+32(FP), R3
	CMP R3, R2
	BLS dot_min_done
	MOVD R3, R2

dot_min_done:
	LSR $3, R2, R2
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V9.B16, V9.B16, V9.B16
	CBZ R2, dot_reduce

dot_loop8:
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R0), [V2.S4]
	VLD1.P 16(R1), [V3.H8]
	WORD $0x0e217864 // fcvtl  v4.4s, v3.4h
	WORD $0x4e217865 // fcvtl2 v5.4s, v3.8h
	VFMLA V1.S4, V4.S4, V0.S4
	VFMLA V2.S4, V5.S4, V9.S4
	SUB $1, R2, R2
	CBNZ R2, dot_loop8

dot_reduce:
	FMOVS $(1.0), F31
	VDUP V31.S[0], V31.S4
	VFMLA V9.S4, V31.S4, V0.S4
	VMOV V0.S[0], R4
	VMOV V0.S[1], R5
	VMOV V0.S[2], R6
	VMOV V0.S[3], R7
	FMOVS R4, F0
	FMOVS R5, F1
	FMOVS R6, F2
	FMOVS R7, F3
	FADDS F1, F0
	FADDS F3, F2
	FADDS F2, F0
	FMOVS F0, ret+48(FP)
	RET

// func axpyF16NEON(out []float32, alpha float32, x []uint16)
TEXT ·axpyF16NEON(SB), NOSPLIT|NOFRAME, $0-56
	MOVD out_base+0(FP), R0
	MOVD R0, R4
	MOVD out_len+8(FP), R2
	FMOVS alpha+24(FP), F31
	VDUP V31.S[0], V31.S4
	MOVD x_base+32(FP), R1
	MOVD x_len+40(FP), R3
	CMP R3, R2
	BLS axpy_min_done
	MOVD R3, R2

axpy_min_done:
	LSR $3, R2, R2
	CBZ R2, axpy_done

axpy_loop8:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R1), [V2.H8]
	WORD $0x0e217843 // fcvtl  v3.4s, v2.4h
	WORD $0x4e217844 // fcvtl2 v4.4s, v2.8h
	VFMLA V3.S4, V31.S4, V0.S4
	VFMLA V4.S4, V31.S4, V1.S4
	VST1.P [V0.S4], 16(R4)
	VST1.P [V1.S4], 16(R4)
	SUB $1, R2, R2
	CBNZ R2, axpy_loop8

axpy_done:
	RET

// func scaleAddF16NEON(out []float32, alpha float32, x []uint16)
TEXT ·scaleAddF16NEON(SB), NOSPLIT|NOFRAME, $0-56
	MOVD out_base+0(FP), R0
	MOVD R0, R4
	MOVD out_len+8(FP), R2
	FMOVS alpha+24(FP), F31
	VDUP V31.S[0], V31.S4
	MOVD x_base+32(FP), R1
	MOVD x_len+40(FP), R3
	CMP R3, R2
	BLS scaleadd_min_done
	MOVD R3, R2

scaleadd_min_done:
	LSR $3, R2, R2
	CBZ R2, scaleadd_done

scaleadd_loop8:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R1), [V2.H8]
	WORD $0x0e217843 // fcvtl  v3.4s, v2.4h
	WORD $0x4e217844 // fcvtl2 v4.4s, v2.8h
	VFMLA V0.S4, V31.S4, V3.S4
	VFMLA V1.S4, V31.S4, V4.S4
	VST1.P [V3.S4], 16(R4)
	VST1.P [V4.S4], 16(R4)
	SUB $1, R2, R2
	CBNZ R2, scaleadd_loop8

scaleadd_done:
	RET

// func f32ToF16RowNEON(dst []uint16, src []float32)
TEXT ·f32ToF16RowNEON(SB), NOSPLIT|NOFRAME, $0-48
	MOVD dst_base+0(FP), R0
	MOVD dst_len+8(FP), R2
	MOVD src_base+24(FP), R1
	MOVD src_len+32(FP), R3
	CMP R3, R2
	BLS cvt_min_done
	MOVD R3, R2

cvt_min_done:
	LSR $3, R2, R2
	CBZ R2, cvt_done

cvt_loop8:
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	WORD $0x0e216823 // fcvtn  v3.4h, v1.4s
	WORD $0x4e216843 // fcvtn2 v3.8h, v2.4s
	VST1.P [V3.H8], 16(R0)
	SUB $1, R2, R2
	CBNZ R2, cvt_loop8

cvt_done:
	RET
