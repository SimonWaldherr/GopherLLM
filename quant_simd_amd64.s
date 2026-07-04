//go:build amd64

#include "textflag.h"

// func sumF32Groups32(x *float32, out *float32, groups int)
// out[g] = sum of x[g*32 : g*32+32].
TEXT ·sumF32Groups32(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ groups+16(FP), CX
	TESTQ CX, CX
	JLE  s32_ret
s32_loop:
	VMOVUPS (SI), Y0
	VADDPS  32(SI), Y0, Y0
	VADDPS  64(SI), Y0, Y0
	VADDPS  96(SI), Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS  X0, (DI)
	ADDQ    $128, SI
	ADDQ    $4, DI
	DECQ    CX
	JNZ     s32_loop
s32_ret:
	VZEROUPPER
	RET

// func sumF32Groups16(x *float32, out *float32, groups int)
// out[g] = sum of x[g*16 : g*16+16].
TEXT ·sumF32Groups16(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ groups+16(FP), CX
	TESTQ CX, CX
	JLE  s16_ret
s16_loop:
	VMOVUPS (SI), Y0
	VADDPS  32(SI), Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS  X0, (DI)
	ADDQ    $64, SI
	ADDQ    $4, DI
	DECQ    CX
	JNZ     s16_loop
s16_ret:
	VZEROUPPER
	RET

// func q4kQDots8(q *byte, x *float32, qdots *float32)
// For each of the 4 32-byte chunks s of the 128 packed-nibble bytes:
//   qdots[2s]   = sum_{l<32} (q[32s+l] & 0x0f) * x[64s + l]
//   qdots[2s+1] = sum_{l<32} (q[32s+l] >> 4)   * x[64s + 32 + l]
TEXT ·q4kQDots8(SB), NOSPLIT, $0-24
	MOVQ q+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ qdots+16(FP), DX
	MOVL $0x0000000f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15 = per-dword 0x0f mask
	XORQ CX, CX                // step s = 0
q4k_step:
	CMPQ CX, $4
	JGE  q4k_ret
	MOVQ CX, R8
	SHLQ $5, R8
	ADDQ SI, R8                // R8 = q + 32*s
	MOVQ CX, R9
	SHLQ $8, R9
	ADDQ DI, R9                // R9 = x + 256*s bytes (= x + 64*s floats)
	VXORPS Y0, Y0, Y0          // accLo
	VXORPS Y1, Y1, Y1          // accHi

	VPMOVZXBD (R8), Y2
	VPAND     Y15, Y2, Y3
	VPSRLD    $4, Y2, Y4
	VCVTDQ2PS Y3, Y3
	VCVTDQ2PS Y4, Y4
	VFMADD231PS (R9), Y3, Y0
	VFMADD231PS 128(R9), Y4, Y1

	VPMOVZXBD 8(R8), Y2
	VPAND     Y15, Y2, Y3
	VPSRLD    $4, Y2, Y4
	VCVTDQ2PS Y3, Y3
	VCVTDQ2PS Y4, Y4
	VFMADD231PS 32(R9), Y3, Y0
	VFMADD231PS 160(R9), Y4, Y1

	VPMOVZXBD 16(R8), Y2
	VPAND     Y15, Y2, Y3
	VPSRLD    $4, Y2, Y4
	VCVTDQ2PS Y3, Y3
	VCVTDQ2PS Y4, Y4
	VFMADD231PS 64(R9), Y3, Y0
	VFMADD231PS 192(R9), Y4, Y1

	VPMOVZXBD 24(R8), Y2
	VPAND     Y15, Y2, Y3
	VPSRLD    $4, Y2, Y4
	VCVTDQ2PS Y3, Y3
	VCVTDQ2PS Y4, Y4
	VFMADD231PS 96(R9), Y3, Y0
	VFMADD231PS 224(R9), Y4, Y1

	MOVQ CX, R10
	SHLQ $3, R10
	ADDQ DX, R10               // R10 = qdots + 8*s bytes
	VEXTRACTF128 $1, Y0, X5
	VADDPS  X5, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS  X0, (R10)
	VEXTRACTF128 $1, Y1, X6
	VADDPS  X6, X1, X1
	VHADDPS X1, X1, X1
	VHADDPS X1, X1, X1
	VMOVSS  X1, 4(R10)
	INCQ CX
	JMP  q4k_step
q4k_ret:
	VZEROUPPER
	RET

// func q4kQDots8Q8(q *byte, q8 *int8, intqdots *int32)
// Like q4kQDots8 but dots the unsigned 4-bit weights against int8-quantized
// activations using VPMADDUBSW, producing 8 integer sub-block dot products:
//   intqdots[2s]   = sum_{l<32} (q[32s+l] & 0x0f) * q8[64s + l]
//   intqdots[2s+1] = sum_{l<32} (q[32s+l] >> 4)   * q8[64s + 32 + l]
TEXT ·q4kQDots8Q8(SB), NOSPLIT, $0-24
	MOVQ q+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ intqdots+16(FP), DX
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15 = per-byte 0x0f mask
	MOVL $0x00010001, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14      // Y14 = per-int16 1 (for VPMADDWD widening)
	XORQ CX, CX                // step s = 0
q8_step:
	CMPQ CX, $4
	JGE  q8_ret
	MOVQ CX, R8
	SHLQ $5, R8
	ADDQ SI, R8                // R8 = q + 32*s
	MOVQ CX, R9
	SHLQ $6, R9
	ADDQ DI, R9                // R9 = q8 + 64*s

	VMOVDQU (R8), Y0           // 32 packed nibble bytes
	VPAND   Y15, Y0, Y1        // low nibbles
	VPSRLW  $4, Y0, Y2
	VPAND   Y15, Y2, Y2        // high nibbles

	VPMADDUBSW (R9), Y1, Y3    // Y3 = lowNibbles(u) * q8lo(s) -> 16 int16
	VPMADDWD   Y14, Y3, Y3     // -> 8 int32
	VPMADDUBSW 32(R9), Y2, Y4  // Y4 = highNibbles(u) * q8hi(s) -> 16 int16
	VPMADDWD   Y14, Y4, Y4     // -> 8 int32

	MOVQ CX, R10
	SHLQ $3, R10
	ADDQ DX, R10               // R10 = intqdots + 8*s bytes (2 int32)
	VEXTRACTI128 $1, Y3, X5
	VPADDD  X5, X3, X3
	VPHADDD X3, X3, X3
	VPHADDD X3, X3, X3         // X3[0] = intqdots[2s]
	MOVL    X3, (R10)
	VEXTRACTI128 $1, Y4, X6
	VPADDD  X6, X4, X4
	VPHADDD X4, X4, X4
	VPHADDD X4, X4, X4         // X4[0] = intqdots[2s+1]
	MOVL    X4, 4(R10)
	INCQ CX
	JMP  q8_step
q8_ret:
	VZEROUPPER
	RET

// SUBQ6K accumulates 8 unsigned 6-bit quants times activations into ACC:
//   v = ((ql[QLOFF..] >> QLSR) & 0x0f) | (((qh[QHOFF..] >> QHSR) & 3) << 4)
//   ACC += v * x[XOFF..]
#define SUBQ6K(QLOFF, QHOFF, XOFF, QLSR, QHSR, ACC) \
	VPMOVZXBD QLOFF(R8), Y2 \
	VPSRLD    $QLSR, Y2, Y2 \
	VPAND     Y13, Y2, Y2 \
	VPMOVZXBD QHOFF(R9), Y3 \
	VPSRLD    $QHSR, Y3, Y3 \
	VPAND     Y14, Y3, Y3 \
	VPSLLD    $4, Y3, Y3 \
	VPOR      Y3, Y2, Y2 \
	VCVTDQ2PS Y2, Y2 \
	VFMADD231PS XOFF(R10), Y2, ACC

// REDUCE horizontally sums the 8 lanes of YS (low half XS) into DST.
#define REDUCE(YS, XS, XT, DST) \
	VEXTRACTF128 $1, YS, XT \
	VADDPS  XT, XS, XS \
	VHADDPS XS, XS, XS \
	VHADDPS XS, XS, XS \
	VMOVSS  XS, DST

// func q6kQDots16(ql *byte, qh *byte, x *float32, qdots *float32)
// qdots[g] = sum over the g-th natural group of 16 elements of
// (unsigned 6-bit quant) * x, matching the Q6_K layout in DequantRowQ6K.
TEXT ·q6kQDots16(SB), NOSPLIT, $0-32
	MOVQ ql+0(FP), AX
	MOVQ qh+8(FP), BX
	MOVQ x+16(FP), DX
	MOVQ qdots+24(FP), DI
	MOVL $0x0000000f, R11
	MOVQ R11, X13
	VPBROADCASTD X13, Y13      // Y13 = 0x0f
	MOVL $0x00000003, R11
	MOVQ R11, X14
	VPBROADCASTD X14, Y14      // Y14 = 0x03
	XORQ CX, CX                // half = 0
q6k_half:
	CMPQ CX, $2
	JGE  q6k_ret
	MOVQ CX, R8
	SHLQ $6, R8
	ADDQ AX, R8                // R8 = ql + 64*half
	MOVQ CX, R9
	SHLQ $5, R9
	ADDQ BX, R9                // R9 = qh + 32*half
	MOVQ CX, R10
	SHLQ $9, R10
	ADDQ DX, R10               // R10 = x + 512*half bytes (= x + 128*half floats)
	MOVQ CX, R12
	SHLQ $5, R12
	ADDQ DI, R12               // R12 = qdots + 32*half bytes (= qdots + 8*half floats)

	// quadrant 0: ql low nibble, qh bits 0-1, x cols 0-31
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	SUBQ6K(0, 0, 0, 0, 0, Y0)
	SUBQ6K(8, 8, 32, 0, 0, Y0)
	SUBQ6K(16, 16, 64, 0, 0, Y1)
	SUBQ6K(24, 24, 96, 0, 0, Y1)
	REDUCE(Y0, X0, X5, (R12))
	REDUCE(Y1, X1, X6, 4(R12))

	// quadrant 1: ql[l+32] low nibble, qh bits 2-3, x cols 32-63
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	SUBQ6K(32, 0, 128, 0, 2, Y0)
	SUBQ6K(40, 8, 160, 0, 2, Y0)
	SUBQ6K(48, 16, 192, 0, 2, Y1)
	SUBQ6K(56, 24, 224, 0, 2, Y1)
	REDUCE(Y0, X0, X5, 8(R12))
	REDUCE(Y1, X1, X6, 12(R12))

	// quadrant 2: ql high nibble, qh bits 4-5, x cols 64-95
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	SUBQ6K(0, 0, 256, 4, 4, Y0)
	SUBQ6K(8, 8, 288, 4, 4, Y0)
	SUBQ6K(16, 16, 320, 4, 4, Y1)
	SUBQ6K(24, 24, 352, 4, 4, Y1)
	REDUCE(Y0, X0, X5, 16(R12))
	REDUCE(Y1, X1, X6, 20(R12))

	// quadrant 3: ql[l+32] high nibble, qh bits 6-7, x cols 96-127
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	SUBQ6K(32, 0, 384, 4, 6, Y0)
	SUBQ6K(40, 8, 416, 4, 6, Y0)
	SUBQ6K(48, 16, 448, 4, 6, Y1)
	SUBQ6K(56, 24, 480, 4, 6, Y1)
	REDUCE(Y0, X0, X5, 24(R12))
	REDUCE(Y1, X1, X6, 28(R12))

	INCQ CX
	JMP  q6k_half
q6k_ret:
	VZEROUPPER
	RET
