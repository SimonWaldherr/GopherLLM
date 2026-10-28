//go:build arm64

#include "textflag.h"

#define SDOT(D,N,M) WORD $(0x4E809400 | ((M)<<16) | ((N)<<5) | (D))

// Keep the eight integer reductions independent before applying their scales.
// Folding each dot immediately into the same FMA accumulator creates a long
// dependency chain from SDOT through horizontal reduction and conversion.
#define Q8DECODE(D,V,F) \
	MOVHU (R0), R6 \
	FMOVS (R3)(R6<<2), F \
	ADD $2, R0 \
	VLD1.P 32(R0), [V0.B16, V1.B16] \
	VLD1.P 32(R1), [V2.B16, V3.B16] \
	VEOR V.B16, V.B16, V.B16 \
	SDOT(D,0,2) \
	SDOT(D,1,3) \
	VADDV V.S4, V \
	WORD $(0x4e21d800 | ((D)<<5) | (D)) // scvtf vD.4s, vD.4s

// One complete Q8_0 row against Q8K activations. Each 256-element activation
// block has one scale and eight Q8_0 weight blocks, each with its own f16 scale.
// Both FMA reductions follow the Go kernel's order exactly.
// func q8_0RowDecodeAsm(row *byte, q8 *int8, scales, lut *float32, blocks int) float32
TEXT ·q8_0RowDecodeAsm(SB), NOSPLIT|NOFRAME, $0-44
	MOVD row+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD lut+24(FP), R3
	MOVD blocks+32(FP), R4
	FMOVS ZR, F20
	CBZ R4, q8decode_done

q8decode_block:
	FMOVS ZR, F16
	Q8DECODE(8,V8,F24)
	Q8DECODE(9,V9,F25)
	Q8DECODE(10,V10,F26)
	Q8DECODE(11,V11,F27)
	Q8DECODE(12,V12,F28)
	Q8DECODE(13,V13,F29)
	Q8DECODE(14,V14,F30)
	Q8DECODE(15,V15,F31)
	FMADDS F24, F16, F8, F16
	FMADDS F25, F16, F9, F16
	FMADDS F26, F16, F10, F16
	FMADDS F27, F16, F11, F16
	FMADDS F28, F16, F12, F16
	FMADDS F29, F16, F13, F16
	FMADDS F30, F16, F14, F16
	FMADDS F31, F16, F15, F16

	FMOVS.P 4(R2), F24
	FMADDS F24, F20, F16, F20
	SUB $1, R4
	CBNZ R4, q8decode_block

q8decode_done:
	FMOVS F20, ret+40(FP)
	RET
