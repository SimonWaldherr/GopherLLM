#include "textflag.h"

// Widen by shifting the stored high half of each FP32 bit pattern into
// place. Keep four SIMD accumulators and the same reduction as dotF32.
TEXT ·dotF32BF16NEON(SB), NOSPLIT|NOFRAME, $0-28
	MOVD x+0(FP), R0
	MOVD raw+8(FP), R1
	MOVD blocks+16(FP), R2
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16

loop:
	VLD1.P 32(R1), [V1.H8, V2.H8]
	VLD1.P 32(R0), [V16.S4, V17.S4]
	VLD1.P 32(R0), [V18.S4, V19.S4]
	WORD $0x2e613824 // shll  v4.4s, v1.4h, #16
	WORD $0x6e613825 // shll2 v5.4s, v1.8h, #16
	WORD $0x2e613846 // shll  v6.4s, v2.4h, #16
	WORD $0x6e613847 // shll2 v7.4s, v2.8h, #16
	VFMLA V4.S4, V16.S4, V0.S4
	VFMLA V5.S4, V17.S4, V9.S4
	VFMLA V6.S4, V18.S4, V10.S4
	VFMLA V7.S4, V19.S4, V11.S4
	SUB $1, R2
	CBNZ R2, loop

	FMOVS $(1.0), F31
	VDUP V31.S[0], V31.S4
	VFMLA V9.S4, V31.S4, V0.S4
	VFMLA V10.S4, V31.S4, V0.S4
	VFMLA V11.S4, V31.S4, V0.S4
	WORD $0x6e20d400 // faddp v0.4s, v0.4s, v0.4s
	WORD $0x6e20d400 // faddp v0.4s, v0.4s, v0.4s
	FMOVS F0, ret+24(FP)
	RET
