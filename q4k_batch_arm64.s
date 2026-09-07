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
