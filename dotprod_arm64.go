//go:build arm64

package gopherllm

import "os"

// FEAT_DotProd (SDOT/UDOT) detection for the arm64 int8-activation kernels.
//
// SDOT is what makes the Q8K row dots worth having: one instruction retires 16
// int8 MACs. It is optional in ARMv8.2 and only mandatory from ARMv8.4, so it
// cannot simply be assumed the way baseline NEON can. Executing SDOT on a part
// without it is a SIGILL, not a slow path, which is why q8k_dot_arm64.s was
// originally gated to darwin && arm64: every Apple Silicon part has it.
//
// That gate cost every other arm64 target — Graviton, Ampere, Snapdragon,
// Raspberry Pi 5, Android — the entire int8 kernel suite, dropping them to the
// portable scalar path for the hottest kernel in the engine. Nearly all of them
// do have FEAT_DotProd. Probing for it at startup keeps the SIGILL impossible
// while letting those parts run the same assembly Apple Silicon does.
//
// The probe result feeds the existing per-kernel *DotAsmOK self-checks in
// q8k_dot_arm64.go, so a machine without dotprod takes exactly the path it took
// before this file existed: the portable kernels, chosen at init, with no
// per-row branch beyond the one already there.
var hasDotProd = detectDotProd()

// detectDotProd honours GOPHERLLM_DISABLE_SIMD so the scalar path stays
// reachable for A/B benchmarking, matching detectAVX2's contract on amd64.
func detectDotProd() bool {
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" {
		return false
	}
	return probeDotProd()
}
