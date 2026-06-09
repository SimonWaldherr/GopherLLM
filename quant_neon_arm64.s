#include "textflag.h"

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
