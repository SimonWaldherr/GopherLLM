//go:build amd64

#include "textflag.h"

// CPUID feature detection
// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() uint32
// Reads the low 32 bits of XCR0 (extended control register 0).
TEXT ·xgetbv(SB), NOSPLIT, $0-4
	XORL CX, CX
	XGETBV
	MOVL AX, ret+0(FP)
	RET

// Float32 dot product
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

// Float16 KV-cache operations
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

// Quantized dot products
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

// Q5KQ8CHUNK processes one 32-byte nibble chunk s of a Q5_K block. Identical
// to Q4KQ8CHUNK except each nibble gains a fifth bit from the qh plane held
// in Y13: sub-block 2s takes bit QHSR1 (= 2s) of qh[l], sub-block 2s+1 takes
// bit QHSR2 (= 2s+1), each shifted up to bit 4 and OR'd onto the nibble
// before the VPMADDUBSW dot (quants stay <= 31, safely unsigned-byte range).
// Y7 is the qh scratch; Y14 holds the per-byte 0x01 mask.
#define Q5KQ8CHUNK(QOFF, Q8OFF, SCOFF, SCOFF2, QHSR1, QHSR2) \
	VMOVDQU QOFF(SI), Y0 \
	VPAND Y15, Y0, Y1 \
	VPSRLW $4, Y0, Y2 \
	VPAND Y15, Y2, Y2 \
	VPSRLW $QHSR1, Y13, Y7 \
	VPAND Y14, Y7, Y7 \
	VPSLLW $4, Y7, Y7 \
	VPOR Y7, Y1, Y1 \
	VPSRLW $QHSR2, Y13, Y7 \
	VPAND Y14, Y7, Y7 \
	VPSLLW $4, Y7, Y7 \
	VPOR Y7, Y2, Y2 \
	VPMADDUBSW Q8OFF(DI), Y1, Y3 \
	VPBROADCASTW SCOFF(SP), Y5 \
	VPMADDWD Y5, Y3, Y3 \
	VPADDD Y3, Y12, Y12 \
	VPMADDUBSW (Q8OFF+32)(DI), Y2, Y4 \
	VPBROADCASTW SCOFF2(SP), Y6 \
	VPMADDWD Y6, Y4, Y4 \
	VPADDD Y4, Y12, Y12

// func q5kDotQ8KRow(q *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32
// Full-row Q5_K x int8-activation dot product. The Q5_K superblock is the
// Q4_K layout (f16 d/dmin at +0/+2, 12 packed scale/min bytes at +4) plus a
// 32-byte fifth-bit plane at +16, with the 128 nibble bytes at +48 and a
// 176-byte stride. Scale/min decoding and the dmin/xsums term are exactly
// q4kDotQ8KRow's; only the quant rebuild differs (see Q5KQ8CHUNK).
TEXT ·q5kDotQ8KRow(SB), NOSPLIT, $32-44
	MOVQ q+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ xsums+24(FP), R9
	MOVQ blocks+32(FP), CX
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15 = per-byte 0x0f mask
	MOVL $0x01010101, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14      // Y14 = per-byte 0x01 mask
	VXORPS Y11, Y11, Y11       // Y11 = accF (d*sc integer-dot term)
	VXORPS Y10, Y10, Y10       // Y10 = accM (dmin*min*xsum term)
	TESTQ CX, CX
	JLE q5kq8k_done
q5kq8k_block:
	// d (f16 at +0) and dmin (f16 at +2)
	MOVL (SI), AX
	MOVQ AX, X8
	VCVTPH2PS X8, X8           // X8 = [d, dmin, ., .]
	// decode the 12 packed scale/min bytes (identical to q4kDotQ8KRow)
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
	// fifth-bit plane, shared by all four chunks of this block
	VMOVDQU 16(SI), Y13
	// integer dots: accI (Y12) collects sc-weighted sub-block dots
	VPXOR Y12, Y12, Y12
	Q5KQ8CHUNK(48, 0, 16, 18, 0, 1)
	Q5KQ8CHUNK(80, 64, 20, 22, 2, 3)
	Q5KQ8CHUNK(112, 128, 24, 26, 4, 5)
	Q5KQ8CHUNK(144, 192, 28, 30, 6, 7)
	// accF += float(accI) * (d * activation scale)
	VCVTDQ2PS Y12, Y12
	VMULSS (R8), X8, X6
	VBROADCASTSS X6, Y6
	VFMADD231PS Y6, Y12, Y11
	ADDQ $176, SI
	ADDQ $256, DI
	ADDQ $4, R8
	ADDQ $32, R9
	DECQ CX
	JNZ q5kq8k_block
q5kq8k_done:
	VSUBPS Y10, Y11, Y11
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VMOVSS X11, ret+40(FP)
	VZEROUPPER
	RET

// Q40Q41EXPAND loads one legacy 32-element block's 16 packed nibble bytes
// and expands them to 32 unsigned int8 quants in Y2, in memory order:
// low nibbles are elements 0..15 (lower lane), high nibbles elements 16..31
// (upper lane). X15 must hold the per-byte 0x0f mask.
#define Q40Q41EXPAND(NIBOFF) \
	VMOVDQU NIBOFF(SI), X1 \
	VPAND X15, X1, X2 \
	VPSRLW $4, X1, X3 \
	VPAND X15, X3, X3 \
	VINSERTI128 $1, X3, Y2, Y2

// func q4_0DotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32
// Full-row Q4_0 x int8-activation dot product. Q4_0 packs 32 elements into
// 18 bytes (f16 d + 16 nibble bytes), value = d*(q-8). blocks counts
// 256-element superchunks (8 legacy blocks each) sharing one Q8K activation
// scale. The integer dot uses the raw unsigned quants 0..15; the -8 offset
// is applied exactly afterwards as -8*d*xsum with the per-32-element float
// activation sums (xsums), mirroring the Q4_K dmin term.
TEXT ·q4_0DotQ8KRow(SB), NOSPLIT, $0-44
	MOVQ row+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ xsums+24(FP), R9
	MOVQ blocks+32(FP), CX
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15/X15 = per-byte 0x0f mask
	MOVL $0x00010001, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14      // Y14 = int16 ones for VPMADDWD
	MOVL $0xc1000000, AX       // -8.0f
	MOVQ AX, X13
	VXORPS Y11, Y11, Y11       // Y11 = vector accumulator (d*xscale*intdot)
	VXORPS X10, X10, X10       // X10 = scalar accumulator (-8*d*xsum terms)
	TESTQ CX, CX
	JLE q40q8k_done
q40q8k_super:
	VBROADCASTSS (R8), Y9      // activation scale for this 256-superchunk
	MOVQ $8, DX
q40q8k_block:
	MOVWLZX (SI), AX           // f16 d
	VMOVD AX, X0
	VCVTPH2PS X0, X0           // X0[0] = d
	Q40Q41EXPAND(2)
	VPMADDUBSW (DI), Y2, Y4
	VPMADDWD Y14, Y4, Y4       // 8 int32 partial dots of this block
	VCVTDQ2PS Y4, Y4
	VMULSS X9, X0, X5          // d * xscale
	VBROADCASTSS X5, Y5
	VFMADD231PS Y5, Y4, Y11
	VMULSS (R9), X0, X6        // d * xsum
	VFMADD231SS X13, X6, X10   // accS += -8 * d * xsum
	ADDQ $18, SI
	ADDQ $32, DI
	ADDQ $4, R9
	DECQ DX
	JNZ q40q8k_block
	ADDQ $4, R8
	DECQ CX
	JNZ q40q8k_super
q40q8k_done:
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VADDSS X10, X11, X11
	VMOVSS X11, ret+40(FP)
	VZEROUPPER
	RET

// func q4_1DotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32
// Q4_1 analogue: 20-byte blocks (f16 d + f16 m + 16 nibble bytes), value =
// d*q + m. Same integer dot as Q4_0; the additive offset lands as +m*xsum.
TEXT ·q4_1DotQ8KRow(SB), NOSPLIT, $0-44
	MOVQ row+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ xsums+24(FP), R9
	MOVQ blocks+32(FP), CX
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15
	MOVL $0x00010001, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14
	VXORPS Y11, Y11, Y11       // Y11 = vector accumulator (d*xscale*intdot)
	VXORPS X10, X10, X10       // X10 = scalar accumulator (+m*xsum terms)
	TESTQ CX, CX
	JLE q41q8k_done
q41q8k_super:
	VBROADCASTSS (R8), Y9
	MOVQ $8, DX
q41q8k_block:
	MOVL (SI), AX              // f16 d | f16 m
	VMOVD AX, X0
	VCVTPH2PS X0, X0           // X0 = [d, m, ., .]
	Q40Q41EXPAND(4)
	VPMADDUBSW (DI), Y2, Y4
	VPMADDWD Y14, Y4, Y4
	VCVTDQ2PS Y4, Y4
	VMULSS X9, X0, X5          // d * xscale
	VBROADCASTSS X5, Y5
	VFMADD231PS Y5, Y4, Y11
	VPSHUFD $0x55, X0, X6      // X6[0] = m
	VMULSS (R9), X6, X6        // m * xsum
	VADDSS X6, X10, X10
	ADDQ $20, SI
	ADDQ $32, DI
	ADDQ $4, R9
	DECQ DX
	JNZ q41q8k_block
	ADDQ $4, R8
	DECQ CX
	JNZ q41q8k_super
q41q8k_done:
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VADDSS X10, X11, X11
	VMOVSS X11, ret+40(FP)
	VZEROUPPER
	RET

// Q2KQ8CHUNK processes one 2-bit-shift extraction of a 32-byte qs chunk
// already loaded in SRC (one of the row's two 32-byte quant groups),
// producing the sc-weighted integer dot for the two 16-element sub-blocks
// that share this shift position — half 0 in SRC's low 128 bits, half 1 in
// its high 128 bits, matching q[half*16+l]'s memory layout exactly, so no
// lane shuffling is needed to split them. Y14 must hold the per-byte 0x03
// mask. SCOFF/SCOFF2 are int16 stack slots (the 0(SP) scale array built in
// q2kDotQ8KRow) holding half 0/half 1's d-scale nibble (sc&0x0f), Q8OFF the
// matching 32-byte activation offset. Mirrors Q4KQ8CHUNK/Q6KQ8QUAD's
// broadcast-scale-then-VPMADDWD combine.
#define Q2KQ8CHUNK(SRC, SHIFT, Q8OFF, SCOFF, SCOFF2) \
	VPSRLW $SHIFT, SRC, Y1 \
	VPAND Y14, Y1, Y1 \
	VPMADDUBSW Q8OFF(DI), Y1, Y3 \
	VPBROADCASTW SCOFF(SP), X5 \
	VPBROADCASTW SCOFF2(SP), X6 \
	VINSERTI128 $1, X6, Y5, Y5 \
	VPMADDWD Y5, Y3, Y3 \
	VPADDD Y3, Y12, Y12

// func q2kDotQ8KRow(q *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32
// Full-row Q2_K x int8-activation dot product. Each 84-byte superblock packs
// 16 sub-blocks of 16 elements: 16 scale bytes (low nibble = d-scale 0..15,
// high nibble = min-scale 0..15, one byte per sub-block — no joint 6-bit
// table like Q4_K/Q5_K), 64 bytes of 2-bit-packed quants (four 2-bit codes
// per byte, extracted at shifts 0/2/4/6), then f16 d at +80 and f16 dmin at
// +82. q8/xscales are as in q4kDotQ8KRow; xsums must hold the per-16-element
// float sums of the ORIGINAL activations (fillQ6KXSums16 — Q2_K shares
// Q6_K's 16-element sub-block grouping, unlike Q4_K/Q5_K's 32-element one),
// unscaled, for the exact min term (dmin*(sc>>4)*xsum, no extra factor
// unlike Q6_K's -32-offset scaling).
TEXT ·q2kDotQ8KRow(SB), NOSPLIT, $32-44
	MOVQ q+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ xsums+24(FP), R9
	MOVQ blocks+32(FP), CX
	MOVL $0x03030303, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14      // Y14 = per-byte 0x03 mask
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X13
	VPBROADCASTD X13, Y13      // X13 = per-byte 0x0f mask (16 bytes; Y13 unused beyond this)
	VXORPS Y11, Y11, Y11       // Y11 = accF (d*sc integer-dot term)
	VXORPS Y10, Y10, Y10       // Y10 = accM (dmin*min*xsum term)
	TESTQ CX, CX
	JLE q2kq8k_done
q2kq8k_block:
	// d (f16 at +80) and dmin (f16 at +82)
	MOVL 80(SI), AX
	MOVQ AX, X8
	VCVTPH2PS X8, X8          // X8 = [d, dmin, ., .]
	// d-scale nibbles (sc&0x0f) for all 16 sub-blocks -> int16 stack array,
	// consumed by Q2KQ8CHUNK's SCOFF/SCOFF2 below.
	VMOVDQU (SI), X0          // 16 scale bytes
	VPAND X13, X0, X1
	VPMOVZXBW X1, Y1
	VMOVDQU Y1, 0(SP)
	// min term: accM += dmin * sum(min_i * xsums_i), folded as two groups of
	// 8 lanes (mirrors q6kDotQ8KRow's offset-term reduction) before the
	// single horizontal reduce at the very end of the row.
	VPMOVZXBD (SI), Y2
	VPSRLD $4, Y2, Y2
	VCVTDQ2PS Y2, Y2
	VMULPS (R9), Y2, Y2
	VPMOVZXBD 8(SI), Y3
	VPSRLD $4, Y3, Y3
	VCVTDQ2PS Y3, Y3
	VFMADD231PS 32(R9), Y3, Y2
	VPSHUFD $0x55, X8, X4      // X4[0] = dmin
	VBROADCASTSS X4, Y4
	VFMADD231PS Y4, Y2, Y10
	// integer dots: accI (Y12) collects sc-weighted sub-block dots. The two
	// 32-byte qs chunks (n=0 at +16, n=128 at +48) are each loaded once and
	// reused across their four shift extractions.
	VPXOR Y12, Y12, Y12
	VMOVDQU 16(SI), Y0
	Q2KQ8CHUNK(Y0, 0, 0, 0, 2)
	Q2KQ8CHUNK(Y0, 2, 32, 4, 6)
	Q2KQ8CHUNK(Y0, 4, 64, 8, 10)
	Q2KQ8CHUNK(Y0, 6, 96, 12, 14)
	VMOVDQU 48(SI), Y0
	Q2KQ8CHUNK(Y0, 0, 128, 16, 18)
	Q2KQ8CHUNK(Y0, 2, 160, 20, 22)
	Q2KQ8CHUNK(Y0, 4, 192, 24, 26)
	Q2KQ8CHUNK(Y0, 6, 224, 28, 30)
	// accF += float(accI) * (d * activation scale)
	VCVTDQ2PS Y12, Y12
	VMULSS (R8), X8, X6
	VBROADCASTSS X6, Y6
	VFMADD231PS Y6, Y12, Y11
	ADDQ $84, SI
	ADDQ $256, DI
	ADDQ $4, R8
	ADDQ $64, R9
	DECQ CX
	JNZ q2kq8k_block
q2kq8k_done:
	VSUBPS Y10, Y11, Y11
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VMOVSS X11, ret+40(FP)
	VZEROUPPER
	RET

// Q3KQ8CHUNK processes one 2-bit-shift extraction of a 32-byte qs chunk, the
// same walk Q2KQ8CHUNK does, plus Q3_K's high-bit correction.
//
// Q3_K's quant is v = ((q >> shift) & 3) - 4*(1 - h), where h is this
// element's bit HBIT of the 32-byte hmask plane: a per-ELEMENT signed offset,
// not the per-sub-block constant that Q2_K/Q4_K/Q5_K factor into an xsums
// term. That is why this format had no fast kernel. It does factor, just
// differently — into the ACTIVATION sum rather than a weight-side constant:
//
//	v   = u - 4 + 4h                  (u = (q >> shift) & 3, h in {0,1})
//	Σv·a = Σ(u + 4h)·a - 4·Σa
//
// and w = u + 4h lands in 0..7, so it is still a valid unsigned operand for
// VPMADDUBSW. The correction therefore costs one extra VPMADDUBSW against a
// vector of ones (which is exactly Σa per pair) and a shift/subtract, with no
// xsums array and no float term — Q3_K stays symmetric, unlike Q4_1/Q2_K.
//
// Intermediate ranges, since VPMADDUBSW saturates: w·a pairs reach 2·7·128 =
// 1792, 4·Σa reaches 4·256 = 1024, and their difference stays well inside
// int16. The subsequent VPMADDWD against dl (-32..31) cannot overflow int32.
//
// Y9 holds the hmask plane, Y14 the per-byte 0x03 mask, and Y15 the per-byte
// 0x01 mask, which doubles as the all-ones multiplicand for the Σa term.
#define Q3KQ8CHUNK(SRC, SHIFT, HBIT, Q8OFF, SCOFF, SCOFF2) \
	VPSRLW $SHIFT, SRC, Y1 \
	VPAND Y14, Y1, Y1 \
	VPSRLW $HBIT, Y9, Y2 \
	VPAND Y15, Y2, Y2 \
	VPSLLW $2, Y2, Y2 \
	VPADDB Y2, Y1, Y1 \
	VPMADDUBSW Q8OFF(DI), Y1, Y3 \
	VPMADDUBSW Q8OFF(DI), Y15, Y4 \
	VPSLLW $2, Y4, Y4 \
	VPSUBW Y4, Y3, Y3 \
	VPBROADCASTW SCOFF(SP), X5 \
	VPBROADCASTW SCOFF2(SP), X6 \
	VINSERTI128 $1, X6, Y5, Y5 \
	VPMADDWD Y5, Y3, Y3 \
	VPADDD Y3, Y12, Y12

// func q3kDotQ8KRow(q *byte, q8 *int8, xscales *float32, blocks int) float32
// Full-row Q3_K x int8-activation dot product. The 110-byte superblock is a
// 32-byte hmask bit-plane at +0, 64 bytes of 2-bit-packed quants at +32 (the
// Q2_K packing: four codes per byte at shifts 0/2/4/6), 12 packed 6-bit
// scale bytes at +96, and the f16 d at +108. There are 16 sub-blocks of 16
// elements, each with its own scale dl = scale - 32.
//
// Unlike Q4_K/Q5_K/Q2_K there is no dmin/min term to accumulate: Q3_K is
// symmetric once the hmask correction is folded in (see Q3KQ8CHUNK), so this
// kernel carries no xsums argument and needs no second accumulator.
//
// Stack frame: 0..15 the 16 unpacked 6-bit scale bytes, 32..63 those scales
// biased by -32 and widened to int16 for Q3KQ8CHUNK's VPBROADCASTW.
TEXT ·q3kDotQ8KRow(SB), NOSPLIT, $64-36
	MOVQ q+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ blocks+24(FP), CX
	MOVL $0x03030303, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14      // Y14 = per-byte 0x03 mask
	MOVL $0x01010101, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15      // Y15 = per-byte 0x01 mask / all-ones operand
	MOVL $0x00200020, AX
	MOVQ AX, X13
	VPBROADCASTD X13, Y13      // Y13 = int16 32, the scale bias
	VXORPS Y11, Y11, Y11       // Y11 = accF
	TESTQ CX, CX
	JLE q3kq8k_done
q3kq8k_block:
	// d is the last two bytes of the block; load exactly two so the final
	// block cannot read past the end of the row.
	MOVWLZX 108(SI), AX
	MOVQ AX, X8
	VCVTPH2PS X8, X8           // X8[0] = d
	// Unpack the 12 packed scale bytes into 16 six-bit scales. This is
	// q3KScales' shuffle with the shift folded into the mask: the original
	// (((tmp >> k) & 0x03030303) << 4) is the same as isolating bits 4..5 of
	// each byte after a single shift, since both operands stay byte-aligned.
	MOVL 96(SI), R10           // aux0 = scale bytes 0..3
	MOVL 100(SI), R11          // aux1 = scale bytes 4..7
	MOVL 104(SI), R12          // tmp  = scale bytes 8..11 (the high 2 bits)
	MOVL R10, R13
	ANDL $0x0f0f0f0f, R13
	MOVL R12, AX
	SHLL $4, AX
	ANDL $0x30303030, AX
	ORL AX, R13
	MOVL R13, 0(SP)            // out[0..3]
	MOVL R11, R13
	ANDL $0x0f0f0f0f, R13
	MOVL R12, AX
	SHLL $2, AX
	ANDL $0x30303030, AX
	ORL AX, R13
	MOVL R13, 4(SP)            // out[4..7]
	MOVL R10, R13
	SHRL $4, R13
	ANDL $0x0f0f0f0f, R13
	MOVL R12, AX
	ANDL $0x30303030, AX
	ORL AX, R13
	MOVL R13, 8(SP)            // out[8..11]
	MOVL R11, R13
	SHRL $4, R13
	ANDL $0x0f0f0f0f, R13
	MOVL R12, AX
	SHRL $2, AX
	ANDL $0x30303030, AX
	ORL AX, R13
	MOVL R13, 12(SP)           // out[12..15]
	VPMOVZXBW 0(SP), Y1        // 16 scales -> int16
	VPSUBW Y13, Y1, Y1         // dl = scale - 32
	VMOVDQU Y1, 32(SP)
	// integer dots: accI (Y12) collects dl-weighted sub-block dots. The
	// hmask plane is loaded once and reused by all eight chunks, each taking
	// a different bit of it (HBIT 0..7, continuing across the two qs chunks
	// exactly as the portable kernel's m <<= 1 does).
	VPXOR Y12, Y12, Y12
	VMOVDQU (SI), Y9
	VMOVDQU 32(SI), Y0
	Q3KQ8CHUNK(Y0, 0, 0, 0, 32, 34)
	Q3KQ8CHUNK(Y0, 2, 1, 32, 36, 38)
	Q3KQ8CHUNK(Y0, 4, 2, 64, 40, 42)
	Q3KQ8CHUNK(Y0, 6, 3, 96, 44, 46)
	VMOVDQU 64(SI), Y0
	Q3KQ8CHUNK(Y0, 0, 4, 128, 48, 50)
	Q3KQ8CHUNK(Y0, 2, 5, 160, 52, 54)
	Q3KQ8CHUNK(Y0, 4, 6, 192, 56, 58)
	Q3KQ8CHUNK(Y0, 6, 7, 224, 60, 62)
	// accF += float(accI) * (d * activation scale)
	VCVTDQ2PS Y12, Y12
	VMULSS (R8), X8, X6
	VBROADCASTSS X6, Y6
	VFMADD231PS Y6, Y12, Y11
	ADDQ $110, SI
	ADDQ $256, DI
	ADDQ $4, R8
	DECQ CX
	JNZ q3kq8k_block
q3kq8k_done:
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VMOVSS X11, ret+32(FP)
	VZEROUPPER
	RET

// mxfp4MagLUT maps an FP4 e2m1 code (0..15) to 2x its magnitude — doubling
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} makes every value an exact small integer
// {0,1,2,3,4,6,8,12}, so the dot product can run through the same int8
// VPMADDUBSW pipeline as the integer quant formats. The sign (code bit 3)
// is applied to the activation operand via VPSIGNB using mxfp4SignLUT.
DATA mxfp4MagLUT<>+0(SB)/8, $0x0c08060403020100
DATA mxfp4MagLUT<>+8(SB)/8, $0x0c08060403020100
GLOBL mxfp4MagLUT<>(SB), RODATA|NOPTR, $16

DATA mxfp4SignLUT<>+0(SB)/8, $0x0101010101010101
DATA mxfp4SignLUT<>+8(SB)/8, $0xffffffffffffffff
GLOBL mxfp4SignLUT<>(SB), RODATA|NOPTR, $16

// func mxfp4DotQ8KRow(row *byte, q8 *int8, xscales *float32, blocks int) float32
// Full-row MXFP4 x int8-activation dot product. MXFP4 packs 32 elements into
// 17 bytes: 16 nibble bytes (element 2i = low nibble of byte i, element 2i+1
// = high nibble — interleaved, unlike Q4_0's half-split) followed by one
// E8M0 scale byte e meaning 2^(e-127), materialized directly as float bits
// e<<23 (e=0 collapses a ~1e-38 denormal scale to zero — irrelevant for
// weights). The format is symmetric, so unlike Q4_0/Q4_1 no xsums offset
// term exists. blocks counts 256-element superchunks (8 blocks each) sharing
// one Q8K activation scale; the doubled LUT magnitudes are compensated by
// folding 0.5 into the per-superchunk scale.
TEXT ·mxfp4DotQ8KRow(SB), NOSPLIT, $0-36
	MOVQ row+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ blocks+24(FP), CX
	MOVL $0x0f0f0f0f, AX
	MOVQ AX, X15
	VPBROADCASTD X15, Y15          // Y15/X15 = per-byte 0x0f mask
	MOVL $0x00010001, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14          // Y14 = int16 ones for VPMADDWD
	VBROADCASTI128 mxfp4MagLUT<>(SB), Y13
	VBROADCASTI128 mxfp4SignLUT<>(SB), Y12
	MOVL $0x3f000000, AX           // 0.5f
	MOVQ AX, X7
	VXORPS Y11, Y11, Y11           // Y11 = float accumulator
	TESTQ CX, CX
	JLE mxfp4q8k_done
mxfp4q8k_super:
	VMULSS (R8), X7, X10           // X10 = 0.5 * activation scale
	MOVQ $8, DX
mxfp4q8k_block:
	VMOVDQU (SI), X1               // 16 packed nibble bytes
	VPAND X15, X1, X2              // low-nibble codes (elements 0,2,4,...)
	VPSRLW $4, X1, X3
	VPAND X15, X3, X3              // high-nibble codes (elements 1,3,5,...)
	VPUNPCKLBW X3, X2, X4          // codes of elements 0..15, in order
	VPUNPCKHBW X3, X2, X5          // codes of elements 16..31
	VINSERTI128 $1, X5, Y4, Y4     // Y4 = 32 codes in element order
	VPSHUFB Y4, Y13, Y6            // unsigned doubled magnitudes 0..12
	VPSHUFB Y4, Y12, Y8            // sign bytes +1/-1 (code bit 3)
	VMOVDQU (DI), Y9
	VPSIGNB Y8, Y9, Y9             // activations with weight signs applied
	VPMADDUBSW Y9, Y6, Y9
	VPMADDWD Y14, Y9, Y9           // 8 int32 partial dots of this block
	VCVTDQ2PS Y9, Y9
	MOVBLZX 16(SI), AX             // E8M0 scale byte
	SHLL $23, AX                   // float bits of 2^(e-127)
	VMOVD AX, X0
	VMULSS X10, X0, X0             // block scale * 0.5 * xscale
	VBROADCASTSS X0, Y0
	VFMADD231PS Y0, Y9, Y11
	ADDQ $17, SI
	ADDQ $32, DI
	DECQ DX
	JNZ mxfp4q8k_block
	ADDQ $4, R8
	DECQ CX
	JNZ mxfp4q8k_super
mxfp4q8k_done:
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VMOVSS X11, ret+32(FP)
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

// Q80CHUNK processes one 34-byte Q8_0 sub-block (BLOCKOFF(SI): 2-byte f16
// scale + 32 signed int8 weights) against the matching 32-byte slice of
// Q8K-quantized activations at Q8OFF(DI). VPMADDUBSW needs one unsigned
// operand, and both Q8_0 weights and Q8K activations are signed, so this
// uses the standard abs/sign-restore identity a*b = |a| * (sign(a)*b):
// Y2 = |weight| (now safely unsigned), Y3 = sign(weight) applied to the
// activation bytes, VPMADDUBSW(Y3, Y2) reproduces the exact signed product
// pairwise-summed as int16, VPMADDWD against the constant 1 (Y14) widens
// and folds pairs to int32 with no scale applied, and the result is
// multiplied by this sub-block's own d and folded into the per-superblock
// accumulator Y12 (the shared per-256-element activation scale is applied
// once by the caller after all 8 sub-blocks are summed).
#define Q80CHUNK(BLOCKOFF, Q8OFF) \
	MOVWLZX (BLOCKOFF)(SI), AX \
	MOVQ AX, X8 \
	VCVTPH2PS X8, X8 \
	VBROADCASTSS X8, Y8 \
	VMOVDQU (BLOCKOFF+2)(SI), Y0 \
	VMOVDQU (Q8OFF)(DI), Y1 \
	VPABSB  Y0, Y2 \
	VPSIGNB Y0, Y1, Y3 \
	VPMADDUBSW Y3, Y2, Y4 \
	VPMADDWD Y14, Y4, Y4 \
	VCVTDQ2PS Y4, Y4 \
	VFMADD231PS Y8, Y4, Y12

// func q8_0DotQ8KRow(row *byte, q8 *int8, xscales *float32, blocks int) float32
// Full-row Q8_0 x int8-activation dot product: one call per weight row. row
// points at the row's 272-byte groups (8 Q8_0 blocks = 256 elements each,
// matching one Q8K activation super-block), q8/xscales as in
// q4kDotQ8KRow/q6kDotQ8KRow. Unlike the K-quants, Q8_0 has no dmin/offset
// term (symmetric quantization), so there is no xsums argument.
TEXT ·q8_0DotQ8KRow(SB), NOSPLIT, $0-36
	MOVQ row+0(FP), SI
	MOVQ q8+8(FP), DI
	MOVQ xscales+16(FP), R8
	MOVQ blocks+24(FP), CX
	MOVL $0x00010001, AX
	MOVQ AX, X14
	VPBROADCASTD X14, Y14      // Y14 = per-16-bit-lane constant 1 (VPMADDWD widen)
	VXORPS Y11, Y11, Y11       // Y11 = row accumulator
	TESTQ CX, CX
	JLE q80q8k_done
q80q8k_block:
	VXORPS Y12, Y12, Y12       // Y12 = per-superblock lane accumulator (pre activation-scale)
	Q80CHUNK(0, 0)
	Q80CHUNK(34, 32)
	Q80CHUNK(68, 64)
	Q80CHUNK(102, 96)
	Q80CHUNK(136, 128)
	Q80CHUNK(170, 160)
	Q80CHUNK(204, 192)
	Q80CHUNK(238, 224)
	VBROADCASTSS (R8), Y6
	VFMADD231PS Y6, Y12, Y11
	ADDQ $272, SI
	ADDQ $256, DI
	ADDQ $4, R8
	DECQ CX
	JNZ q80q8k_block
q80q8k_done:
	VEXTRACTF128 $1, Y11, X1
	VADDPS X1, X11, X11
	VHADDPS X11, X11, X11
	VHADDPS X11, X11, X11
	VMOVSS X11, ret+32(FP)
	VZEROUPPER
	RET

// Vector operations
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
