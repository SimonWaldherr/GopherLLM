#include "textflag.h"

DATA littleBitShifts<>+0(SB)/4, $31
DATA littleBitShifts<>+4(SB)/4, $30
DATA littleBitShifts<>+8(SB)/4, $29
DATA littleBitShifts<>+12(SB)/4, $28
DATA littleBitShifts<>+16(SB)/4, $27
DATA littleBitShifts<>+20(SB)/4, $26
DATA littleBitShifts<>+24(SB)/4, $25
DATA littleBitShifts<>+28(SB)/4, $24
GLOBL littleBitShifts<>(SB), RODATA|NOPTR, $32

DATA littleBitSign<>+0(SB)/4, $0x80000000
GLOBL littleBitSign<>(SB), RODATA|NOPTR, $4

// Each broadcast byte contributes one sign to each of the eight float lanes.
TEXT ·littleBitDotAVX2(SB), NOSPLIT|NOFRAME, $0-28
	MOVQ packed+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ groups+16(FP), CX
	VMOVDQU littleBitShifts<>(SB), Y6
	VPBROADCASTD littleBitSign<>(SB), Y7
	VXORPS Y0, Y0, Y0

loop:
	MOVBLZX (SI), AX
	VMOVD AX, X1
	VPBROADCASTD X1, Y1
	VPSLLVD Y6, Y1, Y1
	VPAND Y7, Y1, Y1
	VXORPS (DI), Y1, Y1
	VADDPS Y1, Y0, Y0
	INCQ SI
	ADDQ $32, DI
	DECQ CX
	JNZ loop

	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET
