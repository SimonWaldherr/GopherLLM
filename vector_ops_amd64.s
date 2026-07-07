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

// siluFxLo clamps the rounded exponent fx to >= -126 so the biased exponent
// (fx+127) constructed below never underflows below the valid finite-normal
// range [1, 254]. No upper clamp is needed: fx = a*log2e where a = -|gate|
// is always <= 0, so fx is always <= 0 by construction and can never
// overflow the exponent field on the positive side (see the comment on
// siluMulF32AVX2 for why the exponent argument is restructured this way).
DATA siluFxLo<>+0(SB)/4, $0xc2fc0000
DATA siluFxLo<>+4(SB)/4, $0xc2fc0000
DATA siluFxLo<>+8(SB)/4, $0xc2fc0000
DATA siluFxLo<>+12(SB)/4, $0xc2fc0000
DATA siluFxLo<>+16(SB)/4, $0xc2fc0000
DATA siluFxLo<>+20(SB)/4, $0xc2fc0000
DATA siluFxLo<>+24(SB)/4, $0xc2fc0000
DATA siluFxLo<>+28(SB)/4, $0xc2fc0000
GLOBL siluFxLo<>(SB), RODATA, $32

// func siluMulF32AVX2(gate, up, out []float32)
// out[i] = gate[i] * sigmoid(gate[i]) * up[i] (SwiGLU inner product), 8 lanes
// per iteration.
//
// sigmoid is evaluated via the numerically-stable split used by scipy's
// expit / PyTorch: for gate>=0, sigmoid(gate) = 1/(1+exp(-gate)); for
// gate<0, sigmoid(gate) = exp(gate)/(1+exp(gate)). Both branches only ever
// evaluate exp() at a NON-POSITIVE argument (a = -|gate|), so exp(a) is
// always in (0,1] and can never overflow — unlike clamping the exp() input
// symmetrically to +-88.376, which keeps exp() itself finite but silently
// turns "conceptually infinite" into "merely ~3.4e38", and that large-but-
// finite value then multiplied against an equally extreme original `gate`
// in the final step produces numerical garbage instead of the correct
// near-zero result. (A prior version of this kernel did exactly that and
// returned -1.4142231 instead of -0 for gate = -MaxFloat32.)
//
// exp(a) is computed from a CLAMPED copy of a (a >= -88.376, needed so the
// 2^fx bit-trick reconstruction stays within the valid finite-exponent
// range); that clamp always produces the same ~4e-39 result regardless of
// how much more negative the true a is, so it is only safe to use directly
// when the clamp didn't engage. When it did (the true |gate| is large
// enough that the correct exp(a) is negligible relative to any float32
// value it could plausibly be multiplied against), the result is forced to
// exactly 0 instead — a standard flush-to-zero, matching what the true
// unclamped exp(a) would round to for any realistic use. Both this and the
// gate-sign branch above are combined via branchless bitwise blends.
// exp(a) itself uses the Cephes expf polynomial (range-reduced degree-5,
// ~1e-7 relative error). Lengths must be equal multiples of 8 (the Go
// wrapper slices accordingly and handles the tail).
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
	VPBROADCASTD X0, Y10          // lo clamp -88.376 (a >= this; exp underflows to ~0 well before it)
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
	VPANDN Y0, Y8, Y1             // absg = gate & ~signbit
	VPOR   Y8, Y1, Y1             // a = absg | signbit = -|gate|  (a <= 0 always)
	VMOVAPS Y1, Y11               // keep the unclamped a for the flush-to-zero test below
	VMAXPS Y10, Y1, Y1            // clamp a >= -88.376
	VMULPS Y15, Y1, Y2            // fx = a * log2e  (fx <= 0 always)
	VROUNDPS $0, Y2, Y2           // round to nearest even
	VMOVUPS siluFxLo<>(SB), Y7
	VMAXPS Y7, Y2, Y2             // clamp fx >= -126 (defensive underflow floor)
	VMOVAPS Y1, Y3
	VFNMADD231PS Y14, Y2, Y3      // r = a - fx*c1
	VFNMADD231PS Y12, Y2, Y3      // r -= fx*c2
	VMOVUPS siluP0<>(SB), Y4
	VFMADD213PS siluP1<>(SB), Y3, Y4
	VFMADD213PS siluP2<>(SB), Y3, Y4
	VFMADD213PS siluP3<>(SB), Y3, Y4
	VFMADD213PS siluP4<>(SB), Y3, Y4
	VFMADD213PS siluP5<>(SB), Y3, Y4
	VMULPS Y3, Y3, Y5             // r^2
	VFMADD213PS Y3, Y5, Y4        // y = y*r^2 + r
	VADDPS Y13, Y4, Y4            // y += 1 -> poly(r) ~= exp(r)
	VCVTPS2DQ Y2, Y6
	VPADDD Y9, Y6, Y6
	VPSLLD $23, Y6, Y6            // 2^fx (fx <= 0, biased exponent always in [1,127])
	VMULPS Y6, Y4, Y4             // e = exp(a_clamped)
	VCMPPS $1, Y10, Y11, Y6       // Y6 = (a_unclamped < -88.376) ? allones : 0
	VPANDN Y4, Y6, Y4             // e = clamp engaged ? 0 : e  (flush-to-zero, see comment above)
	VPSRAD $31, Y0, Y7            // signmask = all-1 if gate<0 else all-0
	VPAND  Y7, Y4, Y5             // t1 = signmask & e
	VPANDN Y13, Y7, Y6            // t2 = ~signmask & 1.0
	VPOR   Y5, Y6, Y5             // numerator = gate<0 ? e : 1.0
	VADDPS Y13, Y4, Y4            // denom = 1 + e
	VDIVPS Y4, Y5, Y5             // s = numerator / denom = sigmoid(gate)
	VMULPS Y0, Y5, Y5             // gate * s
	VMULPS (SI)(CX*4), Y5, Y5     // * up
	VMOVUPS Y5, (R8)(CX*4)
	ADDQ $8, CX
	JMP  si_loop
si_done:
	VZEROUPPER
	RET
