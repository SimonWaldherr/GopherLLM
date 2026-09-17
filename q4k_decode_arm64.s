//go:build arm64

#include "textflag.h"

#define SDOT(D,N,M) WORD $(0x4E809400 | ((M)<<16) | ((N)<<5) | (D))

// One group contains 32 packed bytes and two 32-element activation slices.
// Use distinct accumulators for all eight sub-blocks to expose independent
// SDOT work before the scalar scale/min reduction consumes their results.
#define ROWQ4DOTS(A, AN, B, BN) \
	VLD1.P 32(R6), [V0.B16, V1.B16] \
	VAND V15.B16, V0.B16, V2.B16 \
	VAND V15.B16, V1.B16, V3.B16 \
	VUSHR $4, V0.B16, V4.B16 \
	VUSHR $4, V1.B16, V5.B16 \
	VLD1.P 32(R1), [V6.B16, V7.B16] \
	VLD1.P 32(R1), [V8.B16, V9.B16] \
	VEOR A.B16, A.B16, A.B16 \
	VEOR B.B16, B.B16, B.B16 \
	SDOT(AN, 2, 6) \
	SDOT(AN, 3, 7) \
	SDOT(BN, 4, 8) \
	SDOT(BN, 5, 9)

// Sub-blocks 0..3 store their six-bit scale/min directly in the low bits.
// Keep the minTerm FMAs in sub-block order, exactly as combineQ4KStyle.
#define ROWQ4LOW(S, ACC) \
	VADDV ACC.S4, V10 \
	FMOVS F10, R8 \
	MOVBU (4+S)(R0), R10 \
	MOVBU (8+S)(R0), R11 \
	AND $63, R10 \
	AND $63, R11 \
	MADDW R10, R7, R8, R7 \
	SCVTFWS R11, F10 \
	FMOVS.P 4(R3), F11 \
	FMADDS F10, F24, F11, F24

// Sub-blocks 4..7 use the final four packed bytes plus the high two bits
// of the earlier scale/min bytes.
#define ROWQ4HIGH(S, ACC) \
	VADDV ACC.S4, V10 \
	FMOVS F10, R8 \
	MOVBU (4+S)(R0), R10 \
	MOVBU (8+S)(R0), R11 \
	MOVBU (12+S)(R0), R12 \
	LSR $6, R10 \
	LSR $6, R11 \
	AND $15, R12, R13 \
	LSR $4, R12 \
	ORR R10<<4, R13, R10 \
	ORR R11<<4, R12, R11 \
	MADDW R10, R7, R8, R7 \
	SCVTFWS R11, F10 \
	FMOVS.P 4(R3), F11 \
	FMADDS F10, F24, F11, F24

// func q4kRowDecodeAsm(row *byte, q8 *int8, scales, sums, lut *float32, blocks int) float32
// Scalar arithmetic deliberately matches the Go ARM64 compiler's reduction:
// 8 ordered FMADDS for minTerm; two FMULS then FMSUBS for the block value;
// one FADDS to the row total. No float reassociation or precision changes.
TEXT ·q4kRowDecodeAsm(SB), NOSPLIT|NOFRAME, $0-52
	MOVD row+0(FP), R0
	MOVD q8+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD sums+24(FP), R3
	MOVD blocks+40(FP), R4
	MOVD lut+32(FP), R5
	VMOVI $15, V15.B16
	FMOVS ZR, F28
	CBZ R4, rowdone
rowblock:
	ADD $16, R0, R6
	MOVD ZR, R7
	FMOVS ZR, F24
	ROWQ4DOTS(V16, 16, V17, 17)
	ROWQ4DOTS(V18, 18, V19, 19)
	ROWQ4DOTS(V20, 20, V21, 21)
	ROWQ4DOTS(V22, 22, V23, 23)
	ROWQ4LOW(0, V16)
	ROWQ4LOW(1, V17)
	ROWQ4LOW(2, V18)
	ROWQ4LOW(3, V19)
	ROWQ4HIGH(0, V20)
	ROWQ4HIGH(1, V21)
	ROWQ4HIGH(2, V22)
	ROWQ4HIGH(3, V23)
	MOVHU (R0), R10
	MOVHU 2(R0), R11
	FMOVS (R5)(R10<<2), F26
	FMOVS (R5)(R11<<2), F27
	FMOVS.P 4(R2), F25
	SCVTFWS R7, F29
	FMULS F26, F25, F25
	FMULS F29, F25, F25
	FMSUBS F27, F25, F24, F30
	FADDS F30, F28, F28
	ADD $144, R0
	SUB $1, R4
	CBNZ R4, rowblock
rowdone:
	FMOVS F28, ret+48(FP)
	RET
