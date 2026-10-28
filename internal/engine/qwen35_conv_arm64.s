//go:build arm64

#include "textflag.h"

// Four channels at a time. LD4/LD3 transpose channel-major kernel/history
// rows into one vector per tap. Each FMLA rounds exactly as the corresponding
// Go scalar FMADD; retain the zero accumulator and chronological tap order.
// func qwen35Conv4NEON(kernel, history, values *float32, channels int)
TEXT ·qwen35Conv4NEON(SB), NOSPLIT|NOFRAME, $0-32
	MOVD kernel+0(FP), R0
	MOVD history+8(FP), R1
	MOVD values+16(FP), R2
	MOVD channels+24(FP), R3

conv4_channels:
	VLD4.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD3 (R1), [V4.S4, V5.S4, V6.S4]
	VLD1 (R2), [V7.S4]
	VEOR V8.B16, V8.B16, V8.B16
	VFMLA V4.S4, V0.S4, V8.S4
	VFMLA V5.S4, V1.S4, V8.S4
	VFMLA V6.S4, V2.S4, V8.S4
	VFMLA V7.S4, V3.S4, V8.S4
	VST3.P [V5.S4, V6.S4, V7.S4], 48(R1)
	VST1.P [V8.S4], 16(R2)
	SUB $4, R3, R3
	CBNZ R3, conv4_channels
	RET
