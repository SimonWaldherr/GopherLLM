#include "textflag.h"

// Check exponent bits with integer NEON operations. No floating-point
// conversions or arithmetic touch the input, including signaling NaNs.
// func allFiniteF32(values *float32, count int) bool
TEXT ·allFiniteF32(SB), NOSPLIT|NOFRAME, $0-17
	MOVD values+0(FP), R0
	MOVD count+8(FP), R1
	MOVD $0x7f800000, R2
	VDUP R2, V4.S4
	CMP $16, R1
	BLT tail
loop:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VAND V4.B16, V0.B16, V0.B16
	VAND V4.B16, V1.B16, V1.B16
	VAND V4.B16, V2.B16, V2.B16
	VAND V4.B16, V3.B16, V3.B16
	VCMEQ V4.S4, V0.S4, V0.S4
	VCMEQ V4.S4, V1.S4, V1.S4
	VCMEQ V4.S4, V2.S4, V2.S4
	VCMEQ V4.S4, V3.S4, V3.S4
	VORR V1.B16, V0.B16, V0.B16
	VORR V3.B16, V2.B16, V2.B16
	VORR V2.B16, V0.B16, V0.B16
	VMOV V0.D[0], R3
	VMOV V0.D[1], R4
	ORR R3, R4, R3
	CBNZ R3, nonfinite
	SUB $16, R1
	CMP $16, R1
	BGE loop
tail:
	CBZ R1, finite
	MOVWU.P 4(R0), R3
	AND R2, R3, R3
	CMP R2, R3
	BEQ nonfinite
	SUB $1, R1
	B tail
finite:
	MOVD $1, R0
	MOVB R0, ret+16(FP)
	RET
nonfinite:
	MOVB ZR, ret+16(FP)
	RET
