//go:build arm64

#include "textflag.h"

// func siluMulF32NEON(gate, up, out *float32, n int)
//
// Four independent copies of numeric.SigmoidF32/ExpF32: the same saturation
// bounds, float64 ln(2) range reduction and float32 degree-six polynomial.
// Multiplication and addition remain fused exactly as in the ARM64 scalar
// expression; the final gate*sigmoid*up products are separately rounded.
// Conversion WORDs keep this source compatible with Go assemblers that do
// not yet recognize the vector FCVT/SCVTF mnemonics. All are ARMv8.0 NEON.
TEXT ·siluMulF32NEON(SB), NOSPLIT|NOFRAME, $0-32
	MOVD gate+0(FP), R0
	MOVD up+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3
	CBZ R3, swiglu_done

	FMOVS $(1.4426950408889634), F16
	VDUP V16.S[0], V16.S4
	FMOVD $(0.6931471805599453), F17
	VDUP V17.D[0], V17.D2
	FMOVS $(0.001388888888888889), F18
	VDUP V18.S[0], V18.S4
	FMOVS $(0.008333333333333333), F19
	VDUP V19.S[0], V19.S4
	FMOVS $(0.041666666666666664), F20
	VDUP V20.S[0], V20.S4
	FMOVS $(0.16666666666666666), F21
	VDUP V21.S[0], V21.S4
	FMOVS $(0.5), F22
	VDUP V22.S[0], V22.S4
	FMOVS $(1.0), F23
	VDUP V23.S[0], V23.S4
	FMOVS $(17.0), F24
	VDUP V24.S[0], V24.S4
	FMOVS $(-87.33654), F25
	VDUP V25.S[0], V25.S4
	MOVD $127, R4
	VDUP R4, V27.S4
	MOVD $0x80000000, R4
	VDUP R4, V28.S4

swiglu_loop4:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	// Saturated lanes still multiply the original gate at the end. Clamp
	// only the exponential input to keep exponent construction in range.
	VFMAX V25.S4, V0.S4, V2.S4
	VFMIN V24.S4, V2.S4, V2.S4
	VEOR V28.B16, V2.B16, V2.B16

	// k = trunc(fma(x, invLn2, copysign(0.5, x))). A comparison rather
	// than the sign bit treats -0 like the scalar t >= 0 branch.
	VFCMLT $(0.0), V2.S4, V3.S4
	VAND V28.B16, V3.B16, V3.B16
	VORR V22.B16, V3.B16, V3.B16
	VFMLA V16.S4, V2.S4, V3.S4
	WORD $0x4ea1b863 // fcvtzs v3.4s, v3.4s
	WORD $0x4e21d864 // scvtf v4.4s, v3.4s
	WORD $0x0e617845 // fcvtl v5.2d, v2.2s
	WORD $0x4e617846 // fcvtl2 v6.2d, v2.4s
	WORD $0x0e617887 // fcvtl v7.2d, v4.2s
	WORD $0x4e617888 // fcvtl2 v8.2d, v4.4s
	VFMLS V17.D2, V7.D2, V5.D2
	VFMLS V17.D2, V8.D2, V6.D2
	WORD $0x0e6168a2 // fcvtn v2.2s, v5.2d
	WORD $0x4e6168c2 // fcvtn2 v2.4s, v6.2d

	// Horner evaluation of 1+r*(1+r*(1/2+r*(1/6+...+r/720))).
	VMOV V19.B16, V9.B16
	VFMLA V18.S4, V2.S4, V9.S4
	VMOV V20.B16, V10.B16
	VFMLA V9.S4, V2.S4, V10.S4
	VMOV V21.B16, V9.B16
	VFMLA V10.S4, V2.S4, V9.S4
	VMOV V22.B16, V10.B16
	VFMLA V9.S4, V2.S4, V10.S4
	VMOV V23.B16, V9.B16
	VFMLA V10.S4, V2.S4, V9.S4
	VMOV V23.B16, V10.B16
	VFMLA V9.S4, V2.S4, V10.S4
	VADD V27.S4, V3.S4, V3.S4
	VSHL $23, V3.S4, V3.S4
	VFMUL V3.S4, V10.S4, V10.S4
	VFADD V23.S4, V10.S4, V9.S4
	VFDIV V9.S4, V23.S4, V9.S4

	// Match SigmoidF32's exact early returns at both saturation bounds.
	VFCMGE V24.S4, V0.S4, V4.S4
	VFCMGE V0.S4, V25.S4, V5.S4
	VAND V4.B16, V23.B16, V10.B16
	VBIC V4.B16, V9.B16, V9.B16
	VORR V10.B16, V9.B16, V9.B16
	VBIC V5.B16, V9.B16, V9.B16
	VFMUL V0.S4, V9.S4, V9.S4
	VFMUL V1.S4, V9.S4, V9.S4
	VST1.P [V9.S4], 16(R2)
	SUB $4, R3, R3
	CBNZ R3, swiglu_loop4
swiglu_done:
	RET
