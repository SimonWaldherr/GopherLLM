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

// Broadcast polynomial constants for the Cephes-style expf in siluMulF32AVX2.
DATA siluP0<>+0(SB)/4, $0x39506967
DATA siluP0<>+4(SB)/4, $0x39506967
DATA siluP0<>+8(SB)/4, $0x39506967
DATA siluP0<>+12(SB)/4, $0x39506967
DATA siluP0<>+16(SB)/4, $0x39506967
DATA siluP0<>+20(SB)/4, $0x39506967
DATA siluP0<>+24(SB)/4, $0x39506967
DATA siluP0<>+28(SB)/4, $0x39506967
GLOBL siluP0<>(SB), RODATA, $32
DATA siluP1<>+0(SB)/4, $0x3ab743ce
DATA siluP1<>+4(SB)/4, $0x3ab743ce
DATA siluP1<>+8(SB)/4, $0x3ab743ce
DATA siluP1<>+12(SB)/4, $0x3ab743ce
DATA siluP1<>+16(SB)/4, $0x3ab743ce
DATA siluP1<>+20(SB)/4, $0x3ab743ce
DATA siluP1<>+24(SB)/4, $0x3ab743ce
DATA siluP1<>+28(SB)/4, $0x3ab743ce
GLOBL siluP1<>(SB), RODATA, $32
DATA siluP2<>+0(SB)/4, $0x3c088908
DATA siluP2<>+4(SB)/4, $0x3c088908
DATA siluP2<>+8(SB)/4, $0x3c088908
DATA siluP2<>+12(SB)/4, $0x3c088908
DATA siluP2<>+16(SB)/4, $0x3c088908
DATA siluP2<>+20(SB)/4, $0x3c088908
DATA siluP2<>+24(SB)/4, $0x3c088908
DATA siluP2<>+28(SB)/4, $0x3c088908
GLOBL siluP2<>(SB), RODATA, $32
DATA siluP3<>+0(SB)/4, $0x3d2aa9c1
DATA siluP3<>+4(SB)/4, $0x3d2aa9c1
DATA siluP3<>+8(SB)/4, $0x3d2aa9c1
DATA siluP3<>+12(SB)/4, $0x3d2aa9c1
DATA siluP3<>+16(SB)/4, $0x3d2aa9c1
DATA siluP3<>+20(SB)/4, $0x3d2aa9c1
DATA siluP3<>+24(SB)/4, $0x3d2aa9c1
DATA siluP3<>+28(SB)/4, $0x3d2aa9c1
GLOBL siluP3<>(SB), RODATA, $32
DATA siluP4<>+0(SB)/4, $0x3e2aaaaa
DATA siluP4<>+4(SB)/4, $0x3e2aaaaa
DATA siluP4<>+8(SB)/4, $0x3e2aaaaa
DATA siluP4<>+12(SB)/4, $0x3e2aaaaa
DATA siluP4<>+16(SB)/4, $0x3e2aaaaa
DATA siluP4<>+20(SB)/4, $0x3e2aaaaa
DATA siluP4<>+24(SB)/4, $0x3e2aaaaa
DATA siluP4<>+28(SB)/4, $0x3e2aaaaa
GLOBL siluP4<>(SB), RODATA, $32
DATA siluP5<>+0(SB)/4, $0x3f000000
DATA siluP5<>+4(SB)/4, $0x3f000000
DATA siluP5<>+8(SB)/4, $0x3f000000
DATA siluP5<>+12(SB)/4, $0x3f000000
DATA siluP5<>+16(SB)/4, $0x3f000000
DATA siluP5<>+20(SB)/4, $0x3f000000
DATA siluP5<>+24(SB)/4, $0x3f000000
DATA siluP5<>+28(SB)/4, $0x3f000000
GLOBL siluP5<>(SB), RODATA, $32

// func siluMulF32AVX2(gate, up, out []float32)
// out[i] = gate[i] * sigmoid(gate[i]) * up[i] (SwiGLU inner product), 8 lanes
// per iteration. exp(-gate) uses the Cephes expf polynomial (range-reduced
// degree-5, ~1e-7 relative error). Lengths must be equal multiples of 8
// (the Go wrapper slices accordingly and handles the tail).
TEXT ·siluMulF32AVX2(SB), NOSPLIT, $0-72
	MOVQ gate_base+0(FP), DI
	MOVQ gate_len+8(FP), AX
	MOVQ up_base+24(FP), SI
	MOVQ out_base+48(FP), R8
	MOVL $0x80000000, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y8           // sign mask
	MOVL $0x0000007f, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y9           // exponent bias 127
	MOVL $0xc2b0c0a5, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y10          // lo clamp -88.376
	MOVL $0x42b0c0a5, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y11          // hi clamp +88.376
	MOVL $0xb95e8083, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y12          // c2 = -2.12194440e-4
	MOVL $0x3f800000, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y13          // 1.0
	MOVL $0x3f318000, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y14          // c1 = 0.693359375
	MOVL $0x3fb8aa3b, BX
	MOVQ BX, X0
	VPBROADCASTD X0, Y15          // log2(e)
	XORQ CX, CX
si_loop:
	CMPQ CX, AX
	JGE  si_done
	VMOVUPS (DI)(CX*4), Y0        // gate
	VXORPS Y8, Y0, Y1             // x = -gate
	VMINPS Y11, Y1, Y1
	VMAXPS Y10, Y1, Y1
	VMULPS Y15, Y1, Y2            // fx = x * log2e
	VROUNDPS $0, Y2, Y2           // round to nearest even
	VMOVAPS Y1, Y3
	VFNMADD231PS Y14, Y2, Y3      // r = x - fx*c1
	VFNMADD231PS Y12, Y2, Y3      // r -= fx*c2
	VMOVUPS siluP0<>(SB), Y4
	VFMADD213PS siluP1<>(SB), Y3, Y4
	VFMADD213PS siluP2<>(SB), Y3, Y4
	VFMADD213PS siluP3<>(SB), Y3, Y4
	VFMADD213PS siluP4<>(SB), Y3, Y4
	VFMADD213PS siluP5<>(SB), Y3, Y4
	VMULPS Y3, Y3, Y5             // r^2
	VFMADD213PS Y3, Y5, Y4        // y = y*r^2 + r
	VADDPS Y13, Y4, Y4            // y += 1 -> exp(r)
	VCVTPS2DQ Y2, Y6
	VPADDD Y9, Y6, Y6
	VPSLLD $23, Y6, Y6            // 2^n
	VMULPS Y6, Y4, Y4             // e = exp(-gate)
	VADDPS Y13, Y4, Y4            // 1 + e
	VDIVPS Y4, Y13, Y5            // s = 1 / (1 + e)
	VMULPS Y0, Y5, Y5             // gate * s
	VMULPS (SI)(CX*4), Y5, Y5     // * up
	VMOVUPS Y5, (R8)(CX*4)
	ADDQ $8, CX
	JMP  si_loop
si_done:
	VZEROUPPER
	RET
