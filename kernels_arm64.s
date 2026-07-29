//go:build arm64

#include "textflag.h"

// Float32 dot product
// func dotF32(a, b []float32) float32
TEXT ·dotF32(SB), NOSPLIT|NOFRAME, $0-52
	MOVD a_base+0(FP), R0
	MOVD a_len+8(FP), R2
	MOVD b_base+24(FP), R1
	MOVD b_len+32(FP), R3
	CMP R3, R2
	BLS min_done
	MOVD R3, R2

min_done:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	FMOVS $(1.0), F31
	VDUP V31.S[0], V31.S4
	CBZ R2, reduce
	CMP $16, R2
	BLT reduce

loop16:
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	VFMLA V1.S4, V2.S4, V0.S4
	VLD1.P 16(R0), [V3.S4]
	VLD1.P 16(R1), [V4.S4]
	VFMLA V3.S4, V4.S4, V9.S4
	VLD1.P 16(R0), [V5.S4]
	VLD1.P 16(R1), [V6.S4]
	VFMLA V5.S4, V6.S4, V10.S4
	VLD1.P 16(R0), [V7.S4]
	VLD1.P 16(R1), [V8.S4]
	VFMLA V7.S4, V8.S4, V11.S4
	SUB $16, R2, R2
	CMP $16, R2
	BGE loop16

reduce:
	VFMLA V9.S4, V31.S4, V0.S4
	VFMLA V10.S4, V31.S4, V0.S4
	VFMLA V11.S4, V31.S4, V0.S4
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
	CBZ R2, done

tail:
	FMOVS.P 4(R0), F4
	FMOVS.P 4(R1), F5
	FMULS F5, F4, F4
	FADDS F4, F0
	SUB $1, R2, R2
	CBNZ R2, tail

done:
	FMOVS F0, ret+48(FP)
	RET

// Quantized dot products
// NEON kernels for K-quant dot products.
//
// WORD-encoded instructions (mnemonics unsupported by the Go assembler):
//   ucvtf vN.4s, vN.4s = 0x6E21D800 | N<<5 | N
//   fadd  vd.4s, vn.4s, vm.4s = 0x4E20D400 | m<<16 | n<<5 | d
//   faddp vd.4s, vn.4s, vm.4s = 0x6E20D400 | m<<16 | n<<5 | d

// QGROUP dot-multiplies one register of 16 quantized byte values (Vq)
// against 16 sequential floats streamed from R1, accumulating into the
// float accumulators Va and Vb. Temps: V22-V29.
#define QGROUP(Vq, Va, Vb) \
	VUXTL  Vq.B8, V22.H8     \
	VUXTL2 Vq.B16, V23.H8    \
	VUXTL  V22.H4, V24.S4    \
	VUXTL2 V22.H8, V25.S4    \
	VUXTL  V23.H4, V26.S4    \
	VUXTL2 V23.H8, V27.S4    \
	WORD   $0x6E21DB18       \ // ucvtf v24.4s, v24.4s
	WORD   $0x6E21DB39       \ // ucvtf v25.4s, v25.4s
	WORD   $0x6E21DB5A       \ // ucvtf v26.4s, v26.4s
	WORD   $0x6E21DB7B       \ // ucvtf v27.4s, v27.4s
	VLD1.P 32(R1), [V22.S4, V23.S4] \
	VLD1.P 32(R1), [V28.S4, V29.S4] \
	VFMLA  V22.S4, V24.S4, Va.S4    \
	VFMLA  V23.S4, V25.S4, Vb.S4    \
	VFMLA  V28.S4, V26.S4, Va.S4    \
	VFMLA  V29.S4, V27.S4, Vb.S4

// func q4kQDots8(q *byte, x *float32, qdots *float32)
//
// q points at the 128 packed nibble bytes of one Q4_K block, x at the 256
// matching activations. Writes 8 sub-block dot products (sum of q*x over 32
// elements, q unsigned 0..15) to qdots, in sub-block order.
TEXT ·q4kQDots8(SB), NOSPLIT|NOFRAME, $0-24
	MOVD  q+0(FP), R0
	MOVD  x+8(FP), R1
	MOVD  qdots+16(FP), R2
	VMOVI $15, V31.B16
	MOVD  $4, R3

q4k_step:
	VLD1.P 32(R0), [V16.B16, V17.B16]
	VAND   V31.B16, V16.B16, V18.B16
	VAND   V31.B16, V17.B16, V19.B16
	VUSHR  $4, V16.B16, V20.B16
	VUSHR  $4, V17.B16, V21.B16
	VEOR   V0.B16, V0.B16, V0.B16
	VEOR   V1.B16, V1.B16, V1.B16
	VEOR   V2.B16, V2.B16, V2.B16
	VEOR   V3.B16, V3.B16, V3.B16

	QGROUP(V18, V0, V1) // low nibbles, x[0:16]
	QGROUP(V19, V0, V1) // low nibbles, x[16:32]
	QGROUP(V20, V2, V3) // high nibbles, x[32:48]
	QGROUP(V21, V2, V3) // high nibbles, x[48:64]

	WORD  $0x4E21D400 // fadd  v0.4s, v0.4s, v1.4s
	WORD  $0x4E23D442 // fadd  v2.4s, v2.4s, v3.4s
	WORD  $0x6E22D400 // faddp v0.4s, v0.4s, v2.4s
	WORD  $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	FMOVD F0, (R2)
	ADD   $8, R2
	SUB   $1, R3, R3
	CBNZ  R3, q4k_step
	RET

// func q4kDotPrepared(q *byte, x *float32, scales *float32, mins *float32, xsums *float32, blocks int) float32
//
// q points at the first block's 128 packed nibble bytes (row block + 16).
// scales/mins/xsums contain 8 floats per 256-value block.
TEXT ·q4kDotPrepared(SB), NOSPLIT|NOFRAME, $0-52
	MOVD  q+0(FP), R0
	MOVD  x+8(FP), R1
	MOVD  scales+16(FP), R2
	MOVD  mins+24(FP), R3
	MOVD  xsums+32(FP), R4
	MOVD  blocks+40(FP), R5
	VEOR  V30.B16, V30.B16, V30.B16
	VMOVI $15, V31.B16
	CBZ   R5, q4kprep_done

q4kprep_block:
	MOVD $4, R8

q4kprep_step:
	VLD1.P 32(R0), [V16.B16, V17.B16]
	VAND   V31.B16, V16.B16, V18.B16
	VAND   V31.B16, V17.B16, V19.B16
	VUSHR  $4, V16.B16, V20.B16
	VUSHR  $4, V17.B16, V21.B16
	VEOR   V0.B16, V0.B16, V0.B16
	VEOR   V1.B16, V1.B16, V1.B16
	VEOR   V2.B16, V2.B16, V2.B16
	VEOR   V3.B16, V3.B16, V3.B16

	QGROUP(V18, V0, V1)
	QGROUP(V19, V0, V1)
	QGROUP(V20, V2, V3)
	QGROUP(V21, V2, V3)

	WORD $0x4E21D400 // fadd  v0.4s, v0.4s, v1.4s
	WORD $0x4E23D442 // fadd  v2.4s, v2.4s, v3.4s
	WORD $0x6E22D400 // faddp v0.4s, v0.4s, v2.4s
	WORD $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s

	VMOV      V0.S[0], R6
	VMOV      V0.S[1], R7
	FMOVS     R6, F4
	FMOVS     R7, F5
	FMOVS.P   4(R2), F6
	FMOVS.P   4(R2), F7
	FMOVS.P   4(R3), F8
	FMOVS.P   4(R3), F9
	FMOVS.P   4(R4), F10
	FMOVS.P   4(R4), F11
	FMULS     F6, F4, F4
	FMULS     F7, F5, F5
	FMULS     F10, F8, F8
	FMULS     F11, F9, F9
	FSUBS     F8, F4, F4
	FSUBS     F9, F5, F5
	FADDS     F4, F30
	FADDS     F5, F30
	SUB       $1, R8, R8
	CBNZ      R8, q4kprep_step

	ADD $16, R0
	SUB $1, R5, R5
	CBNZ R5, q4kprep_block

q4kprep_done:
	FMOVS F30, ret+48(FP)
	RET

// func q6kQDots16(ql *byte, qh *byte, x *float32, qdots *float32)
//
// ql points at the 128 low-nibble bytes of one Q6_K block, qh at the 64
// high-bit bytes, x at the 256 matching activations. Writes 16 sub-block dot
// products (sum of q*x over 16 elements, q unsigned 0..63 before the -32
// offset) to qdots, in scale order sc[0..15].
TEXT ·q6kQDots16(SB), NOSPLIT|NOFRAME, $0-32
	MOVD  ql+0(FP), R0
	MOVD  qh+8(FP), R4
	MOVD  x+16(FP), R1
	MOVD  qdots+24(FP), R2
	VMOVI $15, V31.B16
	VMOVI $48, V30.B16
	MOVD  $2, R3

q6k_step:
	VLD1.P 64(R0), [V16.B16, V17.B16, V18.B16, V19.B16]
	VLD1.P 32(R4), [V20.B16, V21.B16]
	VEOR   V0.B16, V0.B16, V0.B16
	VEOR   V1.B16, V1.B16, V1.B16
	VEOR   V2.B16, V2.B16, V2.B16
	VEOR   V3.B16, V3.B16, V3.B16
	VEOR   V4.B16, V4.B16, V4.B16
	VEOR   V5.B16, V5.B16, V5.B16
	VEOR   V6.B16, V6.B16, V6.B16
	VEOR   V7.B16, V7.B16, V7.B16
	VEOR   V8.B16, V8.B16, V8.B16
	VEOR   V9.B16, V9.B16, V9.B16
	VEOR   V10.B16, V10.B16, V10.B16
	VEOR   V11.B16, V11.B16, V11.B16
	VEOR   V12.B16, V12.B16, V12.B16
	VEOR   V13.B16, V13.B16, V13.B16
	VEOR   V14.B16, V14.B16, V14.B16
	VEOR   V15.B16, V15.B16, V15.B16

	// q1 = (ql[0:32]&0x0f) | ((qh<<4)&0x30), x[0:32], sc 0/1
	VSHL $4, V20.B16, V28.B16
	VAND V30.B16, V28.B16, V28.B16
	VAND V31.B16, V16.B16, V29.B16
	VORR V29.B16, V28.B16, V29.B16
	QGROUP(V29, V0, V1)
	VSHL $4, V21.B16, V28.B16
	VAND V30.B16, V28.B16, V28.B16
	VAND V31.B16, V17.B16, V29.B16
	VORR V29.B16, V28.B16, V29.B16
	QGROUP(V29, V2, V3)

	// q2 = (ql[32:64]&0x0f) | ((qh<<2)&0x30), x[32:64], sc 2/3
	VSHL $2, V20.B16, V28.B16
	VAND V30.B16, V28.B16, V28.B16
	VAND V31.B16, V18.B16, V29.B16
	VORR V29.B16, V28.B16, V29.B16
	QGROUP(V29, V4, V5)
	VSHL $2, V21.B16, V28.B16
	VAND V30.B16, V28.B16, V28.B16
	VAND V31.B16, V19.B16, V29.B16
	VORR V29.B16, V28.B16, V29.B16
	QGROUP(V29, V6, V7)

	// q3 = (ql[0:32]>>4) | (qh&0x30), x[64:96], sc 4/5
	VAND  V30.B16, V20.B16, V28.B16
	VUSHR $4, V16.B16, V29.B16
	VORR  V29.B16, V28.B16, V29.B16
	QGROUP(V29, V8, V9)
	VAND  V30.B16, V21.B16, V28.B16
	VUSHR $4, V17.B16, V29.B16
	VORR  V29.B16, V28.B16, V29.B16
	QGROUP(V29, V10, V11)

	// q4 = (ql[32:64]>>4) | ((qh>>2)&0x30), x[96:128], sc 6/7
	VUSHR $2, V20.B16, V28.B16
	VAND  V30.B16, V28.B16, V28.B16
	VUSHR $4, V18.B16, V29.B16
	VORR  V29.B16, V28.B16, V29.B16
	QGROUP(V29, V12, V13)
	VUSHR $2, V21.B16, V28.B16
	VAND  V30.B16, V28.B16, V28.B16
	VUSHR $4, V19.B16, V29.B16
	VORR  V29.B16, V28.B16, V29.B16
	QGROUP(V29, V14, V15)

	WORD  $0x4E21D400 // fadd  v0.4s, v0.4s, v1.4s
	WORD  $0x4E23D442 // fadd  v2.4s, v2.4s, v3.4s
	WORD  $0x6E22D400 // faddp v0.4s, v0.4s, v2.4s
	WORD  $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	FMOVD F0, (R2)
	WORD  $0x4E25D484 // fadd  v4.4s, v4.4s, v5.4s
	WORD  $0x4E27D4C6 // fadd  v6.4s, v6.4s, v7.4s
	WORD  $0x6E26D484 // faddp v4.4s, v4.4s, v6.4s
	WORD  $0x6E24D484 // faddp v4.4s, v4.4s, v4.4s
	FMOVD F4, 8(R2)
	WORD  $0x4E29D508 // fadd  v8.4s, v8.4s, v9.4s
	WORD  $0x4E2BD54A // fadd  v10.4s, v10.4s, v11.4s
	WORD  $0x6E2AD508 // faddp v8.4s, v8.4s, v10.4s
	WORD  $0x6E28D508 // faddp v8.4s, v8.4s, v8.4s
	FMOVD F8, 16(R2)
	WORD  $0x4E2DD58C // fadd  v12.4s, v12.4s, v13.4s
	WORD  $0x4E2FD5CE // fadd  v14.4s, v14.4s, v15.4s
	WORD  $0x6E2ED58C // faddp v12.4s, v12.4s, v14.4s
	WORD  $0x6E2CD58C // faddp v12.4s, v12.4s, v12.4s
	FMOVD F12, 24(R2)
	ADD   $32, R2
	SUB   $1, R3, R3
	CBNZ  R3, q6k_step
	RET

// func sumF32Groups32(x *float32, out *float32, groups int)
//
// Writes out[g] = sum(x[g*32:(g+1)*32]).
TEXT ·sumF32Groups32(SB), NOSPLIT|NOFRAME, $0-24
	MOVD  x+0(FP), R0
	MOVD  out+8(FP), R1
	MOVD  groups+16(FP), R2
	CBZ   R2, sum32_done

sum32_loop:
	VEOR   V0.B16, V0.B16, V0.B16
	VLD1.P 16(R0), [V1.S4]
	WORD   $0x4E21D400 // fadd v0.4s, v0.4s, v1.4s
	VLD1.P 16(R0), [V2.S4]
	WORD   $0x4E22D400 // fadd v0.4s, v0.4s, v2.4s
	VLD1.P 16(R0), [V3.S4]
	WORD   $0x4E23D400 // fadd v0.4s, v0.4s, v3.4s
	VLD1.P 16(R0), [V4.S4]
	WORD   $0x4E24D400 // fadd v0.4s, v0.4s, v4.4s
	VLD1.P 16(R0), [V5.S4]
	WORD   $0x4E25D400 // fadd v0.4s, v0.4s, v5.4s
	VLD1.P 16(R0), [V6.S4]
	WORD   $0x4E26D400 // fadd v0.4s, v0.4s, v6.4s
	VLD1.P 16(R0), [V7.S4]
	WORD   $0x4E27D400 // fadd v0.4s, v0.4s, v7.4s
	VLD1.P 16(R0), [V8.S4]
	WORD   $0x4E28D400 // fadd v0.4s, v0.4s, v8.4s
	WORD   $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	WORD   $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	FMOVS  F0, (R1)
	ADD    $4, R1
	SUB    $1, R2, R2
	CBNZ   R2, sum32_loop

sum32_done:
	RET

// func sumF32Groups16(x *float32, out *float32, groups int)
//
// Writes out[g] = sum(x[g*16:(g+1)*16]).
TEXT ·sumF32Groups16(SB), NOSPLIT|NOFRAME, $0-24
	MOVD  x+0(FP), R0
	MOVD  out+8(FP), R1
	MOVD  groups+16(FP), R2
	CBZ   R2, sum16_done

sum16_loop:
	VEOR   V0.B16, V0.B16, V0.B16
	VLD1.P 16(R0), [V1.S4]
	WORD   $0x4E21D400 // fadd v0.4s, v0.4s, v1.4s
	VLD1.P 16(R0), [V2.S4]
	WORD   $0x4E22D400 // fadd v0.4s, v0.4s, v2.4s
	VLD1.P 16(R0), [V3.S4]
	WORD   $0x4E23D400 // fadd v0.4s, v0.4s, v3.4s
	VLD1.P 16(R0), [V4.S4]
	WORD   $0x4E24D400 // fadd v0.4s, v0.4s, v4.4s
	WORD   $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	WORD   $0x6E20D400 // faddp v0.4s, v0.4s, v0.4s
	FMOVS  F0, (R1)
	ADD    $4, R1
	SUB    $1, R2, R2
	CBNZ   R2, sum16_loop

sum16_done:
	RET

// Vector operations
// func axpyF32(out []float32, alpha float32, x []float32)
TEXT ·axpyF32(SB), NOSPLIT|NOFRAME, $0-56
	MOVD out_base+0(FP), R0
	MOVD R0, R4
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
	VST1.P [V0.S4], 16(R4)
	VST1.P [V2.S4], 16(R4)
	VST1.P [V4.S4], 16(R4)
	VST1.P [V6.S4], 16(R4)
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
	VST1.P [V0.S4], 16(R4)
	SUB $4, R2, R2
	CMP $4, R2
	BGE axpy_loop4

axpy_tail:
	CBZ R2, axpy_done
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FMULS F31, F1, F1
	FADDS F1, F0
	FMOVS.P F0, 4(R4)
	SUB $1, R2, R2
	B axpy_tail

axpy_done:
	RET

// func scaleF32(out []float32, alpha float32)
TEXT ·scaleF32(SB), NOSPLIT|NOFRAME, $0-28
	MOVD out_base+0(FP), R0
	MOVD R0, R4
	MOVD out_len+8(FP), R2
	FMOVS alpha+24(FP), F31
	VDUP V31.S[0], V31.S4
	CMP $16, R2
	BLT scale_loop4_start

scale_loop16:
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R0), [V5.S4]
	VLD1.P 16(R0), [V6.S4]
	VLD1.P 16(R0), [V7.S4]
	WORD   $0x6E3FDC84 // fmul v4.4s, v4.4s, v31.4s
	WORD   $0x6E3FDCA5 // fmul v5.4s, v5.4s, v31.4s
	WORD   $0x6E3FDCC6 // fmul v6.4s, v6.4s, v31.4s
	WORD   $0x6E3FDCE7 // fmul v7.4s, v7.4s, v31.4s
	VST1.P [V4.S4], 16(R4)
	VST1.P [V5.S4], 16(R4)
	VST1.P [V6.S4], 16(R4)
	VST1.P [V7.S4], 16(R4)
	SUB $16, R2, R2
	CMP $16, R2
	BGE scale_loop16

scale_loop4_start:
	CMP $4, R2
	BLT scale_tail

scale_loop4:
	VLD1.P 16(R0), [V0.S4]
	WORD   $0x6E3FDC00 // fmul v0.4s, v0.4s, v31.4s
	VST1.P [V0.S4], 16(R4)
	SUB $4, R2, R2
	CMP $4, R2
	BGE scale_loop4

scale_tail:
	CBZ R2, scale_done
	FMOVS.P 4(R0), F0
	FMULS F31, F0, F0
	FMOVS.P F0, 4(R4)
	SUB $1, R2, R2
	B scale_tail

scale_done:
	RET

// func scaleAddF32(out []float32, alpha float32, x []float32)
TEXT ·scaleAddF32(SB), NOSPLIT|NOFRAME, $0-56
	MOVD out_base+0(FP), R0
	MOVD R0, R4
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
	VST1.P [V1.S4], 16(R4)
	VST1.P [V3.S4], 16(R4)
	VST1.P [V5.S4], 16(R4)
	VST1.P [V7.S4], 16(R4)
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
	VST1.P [V1.S4], 16(R4)
	SUB $4, R2, R2
	CMP $4, R2
	BGE scaleadd_loop4

scaleadd_tail:
	CBZ R2, scaleadd_done
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FMULS F31, F0, F0
	FADDS F1, F0
	FMOVS.P F0, 4(R4)
	SUB $1, R2, R2
	B scaleadd_tail

scaleadd_done:
	RET

// func mulScaleF32(x []float32, weight []float32, scale float32, out []float32)
TEXT ·mulScaleF32(SB), NOSPLIT|NOFRAME, $0-80
	MOVD x_base+0(FP), R0
	MOVD x_len+8(FP), R2
	MOVD weight_base+24(FP), R1
	MOVD weight_len+32(FP), R3
	FMOVS scale+48(FP), F31
	MOVD out_base+56(FP), R4
	MOVD out_len+64(FP), R5
	CMP R3, R2
	BLS mulscale_weight_min_done
	MOVD R3, R2

mulscale_weight_min_done:
	CMP R5, R2
	BLS mulscale_out_min_done
	MOVD R5, R2

mulscale_out_min_done:
	VDUP V31.S[0], V31.S4
	VEOR V30.B16, V30.B16, V30.B16
	CMP $16, R2
	BLT mulscale_loop4_start

mulscale_loop16:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VMOV V30.B16, V2.B16
	VFMLA V1.S4, V0.S4, V2.S4
	VMOV V30.B16, V3.B16
	VFMLA V2.S4, V31.S4, V3.S4
	VST1.P [V3.S4], 16(R4)

	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V5.S4]
	VMOV V30.B16, V6.B16
	VFMLA V5.S4, V4.S4, V6.S4
	VMOV V30.B16, V7.B16
	VFMLA V6.S4, V31.S4, V7.S4
	VST1.P [V7.S4], 16(R4)

	VLD1.P 16(R0), [V8.S4]
	VLD1.P 16(R1), [V9.S4]
	VMOV V30.B16, V10.B16
	VFMLA V9.S4, V8.S4, V10.S4
	VMOV V30.B16, V11.B16
	VFMLA V10.S4, V31.S4, V11.S4
	VST1.P [V11.S4], 16(R4)

	VLD1.P 16(R0), [V12.S4]
	VLD1.P 16(R1), [V13.S4]
	VMOV V30.B16, V14.B16
	VFMLA V13.S4, V12.S4, V14.S4
	VMOV V30.B16, V15.B16
	VFMLA V14.S4, V31.S4, V15.S4
	VST1.P [V15.S4], 16(R4)

	SUB $16, R2, R2
	CMP $16, R2
	BGE mulscale_loop16

mulscale_loop4_start:
	CMP $4, R2
	BLT mulscale_tail

mulscale_loop4:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VMOV V30.B16, V2.B16
	VFMLA V1.S4, V0.S4, V2.S4
	VMOV V30.B16, V3.B16
	VFMLA V2.S4, V31.S4, V3.S4
	VST1.P [V3.S4], 16(R4)
	SUB $4, R2, R2
	CMP $4, R2
	BGE mulscale_loop4

mulscale_tail:
	CBZ R2, mulscale_done
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FMULS F1, F0, F0
	FMULS F31, F0, F0
	FMOVS.P F0, 4(R4)
	SUB $1, R2, R2
	B mulscale_tail

mulscale_done:
	RET
