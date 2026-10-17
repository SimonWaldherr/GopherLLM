//go:build arm64

#include "textflag.h"

#define SDOT(Vd, Vn, Vm) WORD $(0x4E809400 | ((Vm)<<16) | ((Vn)<<5) | (Vd))

// Reuse unpacked V2..V5 for each of four prompt positions.
#define Q4BATCHDOT(PTR, OFF) \
 VLD1.P 32(PTR), [V6.B16, V7.B16] \
 VLD1.P 32(PTR), [V8.B16, V9.B16] \
 VEOR V16.B16, V16.B16, V16.B16 \
 VEOR V17.B16, V17.B16, V17.B16 \
 SDOT(16, 2, 6) \
 SDOT(16, 3, 7) \
 SDOT(17, 4, 8) \
 SDOT(17, 5, 9) \
 VADDV V16.S4, V18 \
 VADDV V17.S4, V19 \
 FMOVS F18, R8 \
 FMOVS F19, R9 \
 MOVW R8, OFF(R5) \
 MOVW R9, (OFF+4)(R5)

// func q4kQ8Dots8x4Asm(q *byte, q8 *int8, stride int, out *int32)
// out contains four consecutive arrays of eight integer sub-block dots.
TEXT ·q4kQ8Dots8x4Asm(SB), NOSPLIT|NOFRAME, $0-32
 MOVD q+0(FP), R0
 MOVD q8+8(FP), R1
 MOVD stride+16(FP), R6
 MOVD out+24(FP), R5
 ADD R6, R1, R2
 ADD R6, R2, R3
 ADD R6, R3, R4
 VMOVI $15, V20.B16
 MOVD $4, R7
batch_group:
 VLD1.P 32(R0), [V0.B16, V1.B16]
 VAND V20.B16, V0.B16, V2.B16
 VAND V20.B16, V1.B16, V3.B16
 VUSHR $4, V0.B16, V4.B16
 VUSHR $4, V1.B16, V5.B16
 Q4BATCHDOT(R1, 0)
 Q4BATCHDOT(R2, 32)
 Q4BATCHDOT(R3, 64)
 Q4BATCHDOT(R4, 96)
 ADD $8, R5
 SUB $1, R7
 CBNZ R7, batch_group
 RET

// Every lane follows one token's original ordered reduction. The two SDOTs
// cover one 32-element sub-block; pairwise reductions gather four token dots.
#define BATCHQ4DOTS(LO, HI) \
 VLD1.P 32(R1), [V6.B16, V7.B16] \
 VLD1.P 32(R2), [V8.B16, V9.B16] \
 VLD1.P 32(R3), [V10.B16, V11.B16] \
 VLD1.P 32(R4), [V12.B16, V13.B16] \
 VEOR V16.B16, V16.B16, V16.B16 \
 VEOR V17.B16, V17.B16, V17.B16 \
 VEOR V18.B16, V18.B16, V18.B16 \
 VEOR V19.B16, V19.B16, V19.B16 \
 SDOT(16, LO, 6) \
 SDOT(16, HI, 7) \
 SDOT(17, LO, 8) \
 SDOT(17, HI, 9) \
 SDOT(18, LO, 10) \
 SDOT(18, HI, 11) \
 SDOT(19, LO, 12) \
 SDOT(19, HI, 13) \
 VADDP V17.S4, V16.S4, V16.S4 \
 VADDP V19.S4, V18.S4, V18.S4 \
 VADDP V18.S4, V16.S4, V16.S4

#define BATCHQ4GROUP() \
 VLD1.P 32(R17), [V0.B16, V1.B16] \
 VAND V14.B16, V0.B16, V2.B16 \
 VAND V14.B16, V1.B16, V3.B16 \
 VUSHR $4, V0.B16, V4.B16 \
 VUSHR $4, V1.B16, V5.B16

// R23/R24 are the shared integer scale/minimum. Keep the eight FMAs in
// sub-block order, as in combineQ4KStyle and q4kRowDecodeAsm.
#define BATCHQ4COMBINE() \
 VDUP R23, V21.S4 \
 VMLA V21.S4, V16.S4, V30.S4 \
 SCVTFWS R24, F22 \
 VDUP V22.S[0], V22.S4 \
 VLD1.P 4(R5), V6.S[0] \
 VLD1.P 4(R6), V6.S[1] \
 VLD1.P 4(R7), V6.S[2] \
 VLD1.P 4(R8), V6.S[3] \
 VFMLA V22.S4, V6.S4, V26.S4

#define BATCHQ4LOW(S, LO, HI) \
 BATCHQ4DOTS(LO, HI) \
 UBFX $(8*S), R14, $6, R23 \
 UBFX $(8*S), R15, $6, R24 \
 BATCHQ4COMBINE()

#define BATCHQ4HIGH(S, LO, HI) \
 BATCHQ4DOTS(LO, HI) \
 UBFX $(8*S+6), R14, $2, R23 \
 UBFX $(8*S+6), R15, $2, R24 \
 UBFX $(8*S), R16, $4, R25 \
 UBFX $(8*S+4), R16, $4, R13 \
 ORR R23<<4, R25, R23 \
 ORR R24<<4, R13, R24 \
 BATCHQ4COMBINE()

// Two multiplies, one fused subtraction and one add preserve the scalar
// block order in each lane, including signed zero.
#define BATCHQ4FINISH() \
 VLD1.P 4(R9), V16.S[0] \
 VLD1.P 4(R10), V16.S[1] \
 VLD1.P 4(R11), V16.S[2] \
 VLD1.P 4(R12), V16.S[3] \
 VDUP V24.S[0], V24.S4 \
 VDUP V25.S[0], V25.S4 \
 VFMUL V24.S4, V16.S4, V16.S4 \
 VFMUL V30.S4, V16.S4, V16.S4 \
 VFMLS V25.S4, V26.S4, V16.S4 \
 VFADD V16.S4, V31.S4, V31.S4

// func q4kRows4Asm(row *byte, q8 *int8, scales, sums, lut *float32, cols, blocks int, out *float32)
TEXT ·q4kRows4Asm(SB), NOSPLIT|NOFRAME, $0-64
 MOVD row+0(FP), R0
 MOVD q8+8(FP), R1
 MOVD scales+16(FP), R9
 MOVD sums+24(FP), R5
 MOVD lut+32(FP), R20
 MOVD cols+40(FP), R21
 MOVD blocks+48(FP), R19
 MOVD out+56(FP), R22
 ADD R21, R1, R2
 ADD R21, R2, R3
 ADD R21, R3, R4
 LSL $5, R19, R23
 ADD R23, R5, R6
 ADD R23, R6, R7
 ADD R23, R7, R8
 LSL $2, R19, R23
 ADD R23, R9, R10
 ADD R23, R10, R11
 ADD R23, R11, R12
 VMOVI $15, V14.B16
 VEOR V31.B16, V31.B16, V31.B16
 CBZ R19, batchrowdone
batchrowblock:
 ADD $16, R0, R17
 MOVWU 4(R0), R14
 MOVWU 8(R0), R15
 MOVWU 12(R0), R16
 VEOR V30.B16, V30.B16, V30.B16
 VEOR V26.B16, V26.B16, V26.B16
 BATCHQ4GROUP()
 BATCHQ4LOW(0, 2, 3)
 BATCHQ4LOW(1, 4, 5)
 BATCHQ4GROUP()
 BATCHQ4LOW(2, 2, 3)
 BATCHQ4LOW(3, 4, 5)
 BATCHQ4GROUP()
 BATCHQ4HIGH(0, 2, 3)
 BATCHQ4HIGH(1, 4, 5)
 BATCHQ4GROUP()
 BATCHQ4HIGH(2, 2, 3)
 BATCHQ4HIGH(3, 4, 5)
 MOVHU (R0), R23
 MOVHU 2(R0), R24
 FMOVS (R20)(R23<<2), F24
 FMOVS (R20)(R24<<2), F25
 WORD $(0x4e21d800 | (30<<5) | 30) // scvtf v30.4s,v30.4s
 BATCHQ4FINISH()
 ADD $144, R0
 SUB $1, R19
 CBNZ R19, batchrowblock
batchrowdone:
 VST1 [V31.S4], (R22)
 RET
