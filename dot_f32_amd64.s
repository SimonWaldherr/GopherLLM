//go:build amd64

#include "textflag.h"

// func dotF32AVX2(a, b []float32) float32
// Computes sum(a[i]*b[i]) over i in [0, min(len(a), len(b))).
TEXT ·dotF32AVX2(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), SI
	MOVQ a_len+8(FP), AX
	MOVQ b_base+24(FP), DI
	MOVQ b_len+32(FP), BX
	CMPQ BX, AX
	CMOVQLT BX, AX          // AX = n = min(len(a), len(b))

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	XORQ CX, CX             // i = 0

	MOVQ AX, DX
	ANDQ $-32, DX           // DX = n rounded down to a multiple of 32

loop32:
	CMPQ CX, DX
	JGE  after32
	VMOVUPS (SI)(CX*4), Y4
	VMOVUPS 32(SI)(CX*4), Y5
	VMOVUPS 64(SI)(CX*4), Y6
	VMOVUPS 96(SI)(CX*4), Y7
	VFMADD231PS (DI)(CX*4), Y4, Y0
	VFMADD231PS 32(DI)(CX*4), Y5, Y1
	VFMADD231PS 64(DI)(CX*4), Y6, Y2
	VFMADD231PS 96(DI)(CX*4), Y7, Y3
	ADDQ $32, CX
	JMP  loop32

after32:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

	MOVQ AX, DX
	ANDQ $-8, DX            // DX = n rounded down to a multiple of 8

loop8:
	CMPQ CX, DX
	JGE  reduce
	VMOVUPS (SI)(CX*4), Y4
	VFMADD231PS (DI)(CX*4), Y4, Y0
	ADDQ $8, CX
	JMP  loop8

reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0       // X0 = 4 partial sums
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0      // X0[0] = sum of the 4 lanes

scalar:
	CMPQ CX, AX
	JGE  done
	VMOVSS (SI)(CX*4), X2
	VMULSS (DI)(CX*4), X2, X2
	VADDSS X2, X0, X0
	INCQ CX
	JMP  scalar

done:
	VZEROUPPER
	MOVSS X0, ret+48(FP)
	RET
