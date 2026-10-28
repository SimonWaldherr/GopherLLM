//go:build arm64

#include "textflag.h"
#include "arm64_fp_compat.h"

#define SDOT(D, N, M) WORD $(0x4E809400 | ((M)<<16) | ((N)<<5) | (D))

#define LEGACYQ4INIT() \
 MOVD row+0(FP), R0 \
 MOVD q8+8(FP), R1 \
 MOVD scales+16(FP), R9 \
 MOVD sums+24(FP), R5 \
 MOVD lut+32(FP), R13 \
 MOVD cols+40(FP), R16 \
 MOVD blocks+48(FP), R14 \
 MOVD out+56(FP), R17 \
 ADD R16, R1, R2 \
 ADD R16, R2, R3 \
 ADD R16, R3, R4 \
 LSL $5, R14, R19 \
 ADD R19, R5, R6 \
 ADD R19, R6, R7 \
 ADD R19, R7, R8 \
 LSL $2, R14, R19 \
 ADD R19, R9, R10 \
 ADD R19, R10, R11 \
 ADD R19, R11, R12 \
 VMOVI $15, V30.B16 \
 VEOR V31.B16, V31.B16, V31.B16

#define LEGACYQ4SCALES() \
 VLD1.P 4(R9), V20.S[0] \
 VLD1.P 4(R10), V20.S[1] \
 VLD1.P 4(R11), V20.S[2] \
 VLD1.P 4(R12), V20.S[3] \
 MOVD $8, R15

// The two unpacked nibble vectors serve all four tokens. Pairwise integer
// reductions gather one complete 32-element dot in each lane of V16.
#define LEGACYQ4DOTS(HEADER) \
 ADD $HEADER, R0, R22 \
 VLD1 (R22), [V0.B16] \
 VAND V30.B16, V0.B16, V1.B16 \
 VUSHR $4, V0.B16, V2.B16 \
 VLD1.P 32(R1), [V3.B16, V4.B16] \
 VLD1.P 32(R2), [V5.B16, V6.B16] \
 VLD1.P 32(R3), [V7.B16, V8.B16] \
 VLD1.P 32(R4), [V9.B16, V10.B16] \
 VEOR V11.B16, V11.B16, V11.B16 \
 VEOR V12.B16, V12.B16, V12.B16 \
 VEOR V13.B16, V13.B16, V13.B16 \
 VEOR V14.B16, V14.B16, V14.B16 \
 SDOT(11, 1, 3) \
 SDOT(11, 2, 4) \
 SDOT(12, 1, 5) \
 SDOT(12, 2, 6) \
 SDOT(13, 1, 7) \
 SDOT(13, 2, 8) \
 SDOT(14, 1, 9) \
 SDOT(14, 2, 10) \
 VADDP V12.S4, V11.S4, V11.S4 \
 VADDP V14.S4, V13.S4, V13.S4 \
 VADDP V13.S4, V11.S4, V16.S4 \
 WORD $(0x4e21d800 | (16<<5) | 16) \
 MOVHU (R0), R20 \
 FMOVS (R13)(R20<<2), F17 \
 VDUP V17.S[0], V17.S4 \
 VLD1.P 4(R5), V19.S[0] \
 VLD1.P 4(R6), V19.S[1] \
 VLD1.P 4(R7), V19.S[2] \
 VLD1.P 4(R8), V19.S[3]

// Match Go's Q4_0 FMUL, fused subtraction, fused accumulation separately
// for every legacy block. In particular, never regroup eight block sums.
TEXT ·q4_0Rows4Asm(SB), NOSPLIT|NOFRAME, $0-64
 LEGACYQ4INIT()
 FMOVS $(8.0), F18
 VDUP V18.S[0], V18.S4
 CBZ R14, q40done
q40block:
 LEGACYQ4SCALES()
q40sub:
 LEGACYQ4DOTS(2)
 FMUL4S(16, 16, 20)
 VFMLS V18.S4, V19.S4, V16.S4
 VFMLA V17.S4, V16.S4, V31.S4
 ADD $18, R0
 SUB $1, R15
 CBNZ R15, q40sub
 SUB $1, R14
 CBNZ R14, q40block
q40done:
 VST1 [V31.S4], (R17)
 RET

// Q4_1 instead rounds d*xscale and its dot product before the minimum's
// fused multiply-add, then performs a distinct add to the running result.
TEXT ·q4_1Rows4Asm(SB), NOSPLIT|NOFRAME, $0-64
 LEGACYQ4INIT()
 CBZ R14, q41done
q41block:
 LEGACYQ4SCALES()
q41sub:
 LEGACYQ4DOTS(4)
 MOVHU 2(R0), R20
 FMOVS (R13)(R20<<2), F18
 VDUP V18.S[0], V18.S4
 FMUL4S(21, 20, 17)
 FMUL4S(16, 16, 21)
 VFMLA V18.S4, V19.S4, V16.S4
 FADD4S(31, 31, 16)
 ADD $20, R0
 SUB $1, R15
 CBNZ R15, q41sub
 SUB $1, R14
 CBNZ R14, q41block
q41done:
 VST1 [V31.S4], (R17)
 RET
