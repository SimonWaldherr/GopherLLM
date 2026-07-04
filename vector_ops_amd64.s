//go:build amd64

#include "textflag.h"

// func axpyF32AVX2(out []float32, alpha float32, x []float32)
// out[i] += alpha*x[i] for i < min(len(out), len(x)).
TEXT ·axpyF32AVX2(SB), NOSPLIT, $0-56
	MOVQ out_base+0(FP), DI
	MOVQ out_len+8(FP), AX
	MOVQ x_base+32(FP), SI
	MOVQ x_len+40(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	MOVSS alpha+24(FP), X0
	VBROADCASTSS X0, Y0
	XORQ CX, CX
	MOVQ AX, DX
	ANDQ $-8, DX
ax_loop:
	CMPQ CX, DX
	JGE  ax_tail
	VMOVUPS (DI)(CX*4), Y1
	VFMADD231PS (SI)(CX*4), Y0, Y1
	VMOVUPS Y1, (DI)(CX*4)
	ADDQ $8, CX
	JMP  ax_loop
ax_tail:
	CMPQ CX, AX
	JGE  ax_done
	VMOVSS (DI)(CX*4), X1
	VFMADD231SS (SI)(CX*4), X0, X1
	VMOVSS X1, (DI)(CX*4)
	INCQ CX
	JMP  ax_tail
ax_done:
	VZEROUPPER
	RET

// func scaleF32AVX2(out []float32, alpha float32)
// out[i] *= alpha.
TEXT ·scaleF32AVX2(SB), NOSPLIT, $0-28
	MOVQ out_base+0(FP), DI
	MOVQ out_len+8(FP), AX
	MOVSS alpha+24(FP), X0
	VBROADCASTSS X0, Y0
	XORQ CX, CX
	MOVQ AX, DX
	ANDQ $-8, DX
sc_loop:
	CMPQ CX, DX
	JGE  sc_tail
	VMULPS (DI)(CX*4), Y0, Y1
	VMOVUPS Y1, (DI)(CX*4)
	ADDQ $8, CX
	JMP  sc_loop
sc_tail:
	CMPQ CX, AX
	JGE  sc_done
	VMULSS (DI)(CX*4), X0, X1
	VMOVSS X1, (DI)(CX*4)
	INCQ CX
	JMP  sc_tail
sc_done:
	VZEROUPPER
	RET

// func scaleAddF32AVX2(out []float32, alpha float32, x []float32)
// out[i] = out[i]*alpha + x[i] for i < min(len(out), len(x)).
TEXT ·scaleAddF32AVX2(SB), NOSPLIT, $0-56
	MOVQ out_base+0(FP), DI
	MOVQ out_len+8(FP), AX
	MOVQ x_base+32(FP), SI
	MOVQ x_len+40(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	MOVSS alpha+24(FP), X0
	VBROADCASTSS X0, Y0
	XORQ CX, CX
	MOVQ AX, DX
	ANDQ $-8, DX
sa_loop:
	CMPQ CX, DX
	JGE  sa_tail
	VMOVUPS (DI)(CX*4), Y1
	VMULPS  Y0, Y1, Y1
	VADDPS  (SI)(CX*4), Y1, Y1
	VMOVUPS Y1, (DI)(CX*4)
	ADDQ $8, CX
	JMP  sa_loop
sa_tail:
	CMPQ CX, AX
	JGE  sa_done
	VMOVSS (DI)(CX*4), X1
	VMULSS X0, X1, X1
	VADDSS (SI)(CX*4), X1, X1
	VMOVSS X1, (DI)(CX*4)
	INCQ CX
	JMP  sa_tail
sa_done:
	VZEROUPPER
	RET

// func mulScaleF32AVX2(x []float32, weight []float32, scale float32, out []float32)
// out[i] = x[i]*weight[i]*scale for i < min(len(x), len(weight), len(out)).
TEXT ·mulScaleF32AVX2(SB), NOSPLIT, $0-80
	MOVQ x_base+0(FP), SI
	MOVQ x_len+8(FP), AX
	MOVQ weight_base+24(FP), DI
	MOVQ weight_len+32(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	MOVQ out_base+56(FP), R8
	MOVQ out_len+64(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX
	MOVSS scale+48(FP), X0
	VBROADCASTSS X0, Y0
	XORQ CX, CX
	MOVQ AX, DX
	ANDQ $-8, DX
ms_loop:
	CMPQ CX, DX
	JGE  ms_tail
	VMOVUPS (SI)(CX*4), Y1
	VMULPS  (DI)(CX*4), Y1, Y1
	VMULPS  Y0, Y1, Y1
	VMOVUPS Y1, (R8)(CX*4)
	ADDQ $8, CX
	JMP  ms_loop
ms_tail:
	CMPQ CX, AX
	JGE  ms_done
	VMOVSS (SI)(CX*4), X1
	VMULSS (DI)(CX*4), X1, X1
	VMULSS X0, X1, X1
	VMOVSS X1, (R8)(CX*4)
	INCQ CX
	JMP  ms_tail
ms_done:
	VZEROUPPER
	RET
