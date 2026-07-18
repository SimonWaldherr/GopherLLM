//go:build amd64

#include "textflag.h"

// f16 KV-cache kernels (AVX2 + F16C + FMA): the cache stores K/V rows as
// IEEE half floats and attention converts on the fly with VCVTPH2PS, halving
// the bytes streamed per attended position versus f32 rows. All four kernels
// process 8 elements per iteration; callers handle the (in practice absent)
// non-multiple-of-8 tails in Go.

// func dotF32F16AVX2(a []float32, b []uint16) float32
// Dot product of an f32 vector with an f16 vector over min(len) &^ 7 elems.
TEXT ·dotF32F16AVX2(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), DI
	MOVQ a_len+8(FP), AX
	MOVQ b_base+24(FP), SI
	MOVQ b_len+32(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	VXORPS Y0, Y0, Y0
	XORQ CX, CX
	ANDQ $-8, AX
df16_loop:
	CMPQ CX, AX
	JGE  df16_done
	VCVTPH2PS (SI)(CX*2), Y1
	VFMADD231PS (DI)(CX*4), Y1, Y0
	ADDQ $8, CX
	JMP  df16_loop
df16_done:
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+48(FP)
	VZEROUPPER
	RET

// func axpyF16AVX2(out []float32, alpha float32, x []uint16)
// out[i] += alpha * f16(x[i]) over min(len) &^ 7 elems.
TEXT ·axpyF16AVX2(SB), NOSPLIT, $0-56
	MOVQ out_base+0(FP), DI
	MOVQ out_len+8(FP), AX
	MOVQ x_base+32(FP), SI
	MOVQ x_len+40(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	MOVSS alpha+24(FP), X0
	VBROADCASTSS X0, Y0
	XORQ CX, CX
	ANDQ $-8, AX
axf16_loop:
	CMPQ CX, AX
	JGE  axf16_done
	VCVTPH2PS (SI)(CX*2), Y1
	VMOVUPS (DI)(CX*4), Y2
	VFMADD231PS Y0, Y1, Y2
	VMOVUPS Y2, (DI)(CX*4)
	ADDQ $8, CX
	JMP  axf16_loop
axf16_done:
	VZEROUPPER
	RET

// func scaleAddF16AVX2(out []float32, alpha float32, x []uint16)
// out[i] = out[i]*alpha + f16(x[i]) over min(len) &^ 7 elems.
TEXT ·scaleAddF16AVX2(SB), NOSPLIT, $0-56
	MOVQ out_base+0(FP), DI
	MOVQ out_len+8(FP), AX
	MOVQ x_base+32(FP), SI
	MOVQ x_len+40(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	MOVSS alpha+24(FP), X0
	VBROADCASTSS X0, Y0
	XORQ CX, CX
	ANDQ $-8, AX
saf16_loop:
	CMPQ CX, AX
	JGE  saf16_done
	VMOVUPS (DI)(CX*4), Y1
	VMULPS  Y0, Y1, Y1
	VCVTPH2PS (SI)(CX*2), Y2
	VADDPS  Y2, Y1, Y1
	VMOVUPS Y1, (DI)(CX*4)
	ADDQ $8, CX
	JMP  saf16_loop
saf16_done:
	VZEROUPPER
	RET

// func f32ToF16RowAVX2(dst []uint16, src []float32)
// dst[i] = f16(src[i]) (round to nearest even) over min(len) &^ 7 elems.
TEXT ·f32ToF16RowAVX2(SB), NOSPLIT, $0-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), AX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	XORQ CX, CX
	ANDQ $-8, AX
cvt16_loop:
	CMPQ CX, AX
	JGE  cvt16_done
	VMOVUPS (SI)(CX*4), Y1
	VCVTPS2PH $0, Y1, (DI)(CX*2)
	ADDQ $8, CX
	JMP  cvt16_loop
cvt16_done:
	VZEROUPPER
	RET
