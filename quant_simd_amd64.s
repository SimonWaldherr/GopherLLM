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

// q8kPermIdx reorders the dword groups produced by the
// VPACKSSDW/VPACKSSWB lane-interleaved packs back into sequential order.
DATA q8kPermIdx<>+0(SB)/4, $0
DATA q8kPermIdx<>+4(SB)/4, $4
DATA q8kPermIdx<>+8(SB)/4, $1
DATA q8kPermIdx<>+12(SB)/4, $5
DATA q8kPermIdx<>+16(SB)/4, $2
DATA q8kPermIdx<>+20(SB)/4, $6
DATA q8kPermIdx<>+24(SB)/4, $3
DATA q8kPermIdx<>+28(SB)/4, $7
GLOBL q8kPermIdx<>(SB), RODATA, $32

// func q8kQuantize(x *float32, q8 *int8, scales *float32, blocks int)
// Quantizes x to int8 per 256-element block (symmetric absmax, llama.cpp's
// Q8_K convention): scales[b] = absmax/127, q8[i] = round(x[i]*127/absmax)
// with round-to-nearest-even (VCVTPS2DQ under the default MXCSR mode). An
// all-zero (or NaN-max) block stores scale 0 and zero quants.
TEXT ·q8kQuantize(SB), NOSPLIT, $0-32
	MOVQ x+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ scales+16(FP), DX
	MOVQ blocks+24(FP), CX
	TESTQ CX, CX
	JLE q8kq_ret
	MOVL $0x7fffffff, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15 = abs mask
	MOVL $0x42fe0000, AX       // 127.0f
	MOVQ AX, X14
	VMOVDQU q8kPermIdx<>(SB), Y13
q8kq_block:
	// absmax over the block's 256 floats
	VXORPS Y1, Y1, Y1
	MOVQ SI, R8
	MOVQ $8, R10
q8kq_amax:
	VMOVUPS (R8), Y0
	VANDPS Y15, Y0, Y0
	VMAXPS Y0, Y1, Y1
	VMOVUPS 32(R8), Y0
	VANDPS Y15, Y0, Y0
	VMAXPS Y0, Y1, Y1
	VMOVUPS 64(R8), Y0
	VANDPS Y15, Y0, Y0
	VMAXPS Y0, Y1, Y1
	VMOVUPS 96(R8), Y0
	VANDPS Y15, Y0, Y0
	VMAXPS Y0, Y1, Y1
	ADDQ $128, R8
	DECQ R10
	JNZ q8kq_amax
	VEXTRACTF128 $1, Y1, X2
	VMAXPS X2, X1, X1
	VSHUFPS $0x0e, X1, X1, X2
	VMAXPS X2, X1, X1
	VSHUFPS $0x01, X1, X1, X2
	VMAXSS X2, X1, X1          // X1[0] = absmax
	VXORPS X0, X0, X0
	VUCOMISS X0, X1
	JNE q8kq_scale
	// zero (or NaN) block: scale 0, quants 0
	VMOVSS X0, (DX)
	VPXOR Y0, Y0, Y0
	VMOVDQU Y0, (DI)
	VMOVDQU Y0, 32(DI)
	VMOVDQU Y0, 64(DI)
	VMOVDQU Y0, 96(DI)
	VMOVDQU Y0, 128(DI)
	VMOVDQU Y0, 160(DI)
	VMOVDQU Y0, 192(DI)
	VMOVDQU Y0, 224(DI)
	ADDQ $1024, SI
	ADDQ $256, DI
	JMP q8kq_next
q8kq_scale:
	VDIVSS X14, X1, X3         // scale = absmax / 127
	VMOVSS X3, (DX)
	VDIVSS X1, X14, X4         // inv = 127 / absmax
	VBROADCASTSS X4, Y4
	MOVQ $8, R10
q8kq_quant:
	VMULPS (SI), Y4, Y0
	VCVTPS2DQ Y0, Y0
	VMULPS 32(SI), Y4, Y1
	VCVTPS2DQ Y1, Y1
	VMULPS 64(SI), Y4, Y2
	VCVTPS2DQ Y2, Y2
	VMULPS 96(SI), Y4, Y3
	VCVTPS2DQ Y3, Y3
	VPACKSSDW Y1, Y0, Y0
	VPACKSSDW Y3, Y2, Y2
	VPACKSSWB Y2, Y0, Y0
	VPERMD Y0, Y13, Y0
	VMOVDQU Y0, (DI)
	ADDQ $128, SI
	ADDQ $32, DI
	DECQ R10
	JNZ q8kq_quant
q8kq_next:
	ADDQ $4, DX
	DECQ CX
	JNZ q8kq_block
q8kq_ret:
	VZEROUPPER
	RET

// Q4KQ8CHUNK processes one 32-byte nibble chunk s of a Q4_K block: the low
// nibbles dot sub-block 2s of the int8 activations, the high nibbles
// sub-block 2s+1, each VPMADDWD-scaled by its 6-bit scale (as int16 at
// SCOFF(SP)) and accumulated vertically into the block's int32 accumulator
// Y12. QOFF = 16+32s (weight bytes), Q8OFF = 64s (activation bytes),
// SCOFF/SCOFF2 = 16+4s / 18+4s (int16 scale slots on the stack).
#define Q4KQ8CHUNK(QOFF, Q8OFF, SCOFF, SCOFF2) \
	VMOVDQU QOFF(SI), Y0 \
	VPAND Y15, Y0, Y1 \
	VPSRLW $4, Y0, Y2 \
	VPAND Y15, Y2, Y2 \
	VPMADDUBSW Q8OFF(DI), Y1, Y3 \
	VPBROADCASTW SCOFF(SP), Y5 \
	VPMADDWD Y5, Y3, Y3 \
	VPADDD Y3, Y12, Y12 \
	VPMADDUBSW (Q8OFF+32)(DI), Y2, Y4 \
	VPBROADCASTW SCOFF2(SP), Y6 \
	VPMADDWD Y6, Y4, Y4 \
	VPADDD Y4, Y12, Y12

// func q4kDotQ8KRow(q *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32
// Full-row Q4_K x int8-activation dot product: one call per weight row.
// q points at the row's 144-byte superblocks, q8 at the Q8K-quantized
// activations (256 int8 per block), xscales at the per-block activation
// scales, and xsums at the per-32-element float sums of the ORIGINAL
// activations (fillQ4KXSums) for the exact dmin term. Scale/min decoding,
// integer dots (VPMADDUBSW/VPMADDWD) and scale application all stay
// in-register; a single horizontal reduction happens at the end of the row.
TEXT ·q4kDotQ8KRow(SB), NOSPLIT, $32-44
	MOVQ q+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ xsums+24(FP), R9
	MOVQ blocks+32(FP), CX
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15 = per-byte 0x0f mask
	VXORPS Y11, Y11, Y11       // Y11 = accF (d*sc integer-dot term)
	VXORPS Y10, Y10, Y10       // Y10 = accM (dmin*min*xsum term)
	TESTQ CX, CX
	JLE q4kq8k_done
q4kq8k_block:
	// d (f16 at +0) and dmin (f16 at +2)
	MOVL (SI), AX
	MOVQ AX, X8
	VCVTPH2PS X8, X8           // X8 = [d, dmin, ., .]
	// decode the 12 packed scale/min bytes into 8 scales + 8 mins
	// (getScaleMinK4 for all j at once, llama.cpp's kmask dance)
	MOVL 4(SI), AX             // A = scale bytes 0..3
	MOVL 8(SI), BX             // B = scale bytes 4..7
	MOVL 12(SI), R10           // C = scale bytes 8..11
	MOVL AX, R11
	ANDL $0x3f3f3f3f, R11      // sc[0..3] = A & 63
	MOVL R10, R12
	ANDL $0x0f0f0f0f, R12
	MOVL AX, R13
	SHRL $2, R13
	ANDL $0x30303030, R13
	ORL R13, R12               // sc[4..7] = (C & 0x0f) | ((A >> 6) << 4)
	MOVL BX, R13
	ANDL $0x3f3f3f3f, R13      // m[0..3] = B & 63
	SHRL $4, R10
	ANDL $0x0f0f0f0f, R10
	MOVL BX, AX
	SHRL $2, AX
	ANDL $0x30303030, AX
	ORL AX, R10                // m[4..7] = (C >> 4) | ((B >> 6) << 4)
	MOVL R11, 0(SP)
	MOVL R12, 4(SP)
	MOVL R13, 8(SP)
	MOVL R10, 12(SP)
	VPMOVZXBW 0(SP), X1        // 8 scales -> int16
	VMOVDQU X1, 16(SP)
	// min term: accM += dmin * mins_f32 * xsums
	VPMOVZXBD 8(SP), Y2
	VCVTDQ2PS Y2, Y2
	VMULPS (R9), Y2, Y2
	VPSHUFD $0x55, X8, X4      // X4[0] = dmin
	VBROADCASTSS X4, Y4
	VFMADD231PS Y4, Y2, Y10
	// integer dots: accI (Y12) collects sc-weighted sub-block dots
	VPXOR Y12, Y12, Y12
	Q4KQ8CHUNK(16, 0, 16, 18)
	Q4KQ8CHUNK(48, 64, 20, 22)
	Q4KQ8CHUNK(80, 128, 24, 26)
	Q4KQ8CHUNK(112, 192, 28, 30)
	// accF += float(accI) * (d * activation scale)
	VCVTDQ2PS Y12, Y12
	VMULSS (R8), X8, X6
	VBROADCASTSS X6, Y6
	VFMADD231PS Y6, Y12, Y11
	ADDQ $144, SI
	ADDQ $256, DI
	ADDQ $4, R8
	ADDQ $32, R9
	DECQ CX
	JNZ q4kq8k_block
q4kq8k_done:
	VSUBPS Y10, Y11, Y11
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VMOVSS X11, ret+40(FP)
	VZEROUPPER
	RET

// Q6KQ8QUAD processes one 32-value quadrant of a Q6_K half: rebuilds the
// unsigned 6-bit quants from the low-nibble/high-bit planes already loaded
// in Y0 (ql lo), Y1 (ql hi) and Y2 (qh), dots them against 32 int8
// activations at Q8OFF(DI), applies the two signed per-16 scales (int16 at
// SCOFF/SCOFF2 on the stack) via VPMADDWD, and accumulates into Y12.
// QLREG selects ql plane (Y0 or Y1), QLSR its nibble shift (0 or 4),
// QHSR the qh bit shift (0/2/4/6).
#define Q6KQ8QUAD(QLREG, QLSR, QHSR, Q8OFF, SCOFF, SCOFF2) \
	VPSRLW $QLSR, QLREG, Y3 \
	VPAND Y15, Y3, Y3 \
	VPSRLW $QHSR, Y2, Y4 \
	VPAND Y14, Y4, Y4 \
	VPSLLW $4, Y4, Y4 \
	VPOR Y4, Y3, Y3 \
	VPMADDUBSW Q8OFF(DI), Y3, Y5 \
	VPBROADCASTW SCOFF(SP), X6 \
	VPBROADCASTW SCOFF2(SP), X7 \
	VINSERTI128 $1, X7, Y6, Y6 \
	VPMADDWD Y6, Y5, Y5 \
	VPADDD Y5, Y12, Y12

// func q6kDotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32
// Full-row Q6_K x int8-activation dot product. row points at the row's
// 210-byte superblocks, q8/xscales as in q4kDotQ8KRow, and xsums at the
// per-16-element sums of the ORIGINAL activations pre-scaled by 32
// (fillQ6KXSums16 + ScaleF32(xs, 32)), folding the constant -32 offset of
// Q6_K quants out of the integer dots.
TEXT ·q6kDotQ8KRow(SB), NOSPLIT, $32-44
	MOVQ row+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ xsums+24(FP), R9
	MOVQ blocks+32(FP), CX
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15 = per-byte 0x0f mask
	MOVL $0x03030303, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14      // Y14 = per-byte 0x03 mask
	VXORPS Y11, Y11, Y11       // Y11 = accF
	VXORPS Y10, Y10, Y10       // Y10 = accM (-32 offset term)
	TESTQ CX, CX
	JLE q6kq8k_done
q6kq8k_block:
	// d (f16 at +208)
	MOVWLZX 208(SI), AX
	MOVQ AX, X8
	VCVTPH2PS X8, X8           // X8[0] = d
	// 16 signed per-16 scales -> int16 on the stack
	VPMOVSXBW 192(SI), Y1
	VMOVDQU Y1, 0(SP)
	// offset term: accM += d * (sc_lo*xsums_lo + sc_hi*xsums_hi)
	VPMOVSXBD 192(SI), Y2
	VCVTDQ2PS Y2, Y2
	VMULPS (R9), Y2, Y2
	VPMOVSXBD 200(SI), Y3
	VCVTDQ2PS Y3, Y3
	VFMADD231PS 32(R9), Y3, Y2
	VBROADCASTSS X8, Y4
	VFMADD231PS Y4, Y2, Y10
	VPXOR Y12, Y12, Y12
	// half 0: ql plane bytes 0..63, qh bytes 128..159, x/q8 values 0..127
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VMOVDQU 128(SI), Y2
	Q6KQ8QUAD(Y0, 0, 0, 0, 0, 2)
	Q6KQ8QUAD(Y1, 0, 2, 32, 4, 6)
	Q6KQ8QUAD(Y0, 4, 4, 64, 8, 10)
	Q6KQ8QUAD(Y1, 4, 6, 96, 12, 14)
	// half 1: ql bytes 64..127, qh bytes 160..191, values 128..255
	VMOVDQU 64(SI), Y0
	VMOVDQU 96(SI), Y1
	VMOVDQU 160(SI), Y2
	Q6KQ8QUAD(Y0, 0, 0, 128, 16, 18)
	Q6KQ8QUAD(Y1, 0, 2, 160, 20, 22)
	Q6KQ8QUAD(Y0, 4, 4, 192, 24, 26)
	Q6KQ8QUAD(Y1, 4, 6, 224, 28, 30)
	// accF += float(accI) * (d * activation scale)
	VCVTDQ2PS Y12, Y12
	VMULSS (R8), X8, X6
	VBROADCASTSS X6, Y6
	VFMADD231PS Y6, Y12, Y11
	ADDQ $210, SI
	ADDQ $256, DI
	ADDQ $4, R8
	ADDQ $64, R9
	DECQ CX
	JNZ q6kq8k_block
q6kq8k_done:
	VSUBPS Y10, Y11, Y11
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VMOVSS X11, ret+40(FP)
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
