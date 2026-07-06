//go:build amd64

package gopherllm

import "os"

// hasAVX2 reports whether the CPU (and OS) support the AVX2 + FMA instructions
// used by the hand-written amd64 kernels. It is set once at startup; every SIMD
// entry point falls back to the portable scalar path when it is false. Set
// GOPHERLLM_DISABLE_SIMD to force the scalar path (useful for A/B benchmarking).
var hasAVX2 = detectAVX2()

// hasF16C reports CPUID F16C support (VCVTPH2PS), required by the int8
// activation kernels that convert f16 block scales in-register. Every
// AVX2+FMA CPU in practice also has F16C, but the check is cheap.
var hasF16C = detectF16C()

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() uint32

func detectF16C() bool {
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" {
		return false
	}
	const f16cBit = 1 << 29 // CPUID.1:ECX.F16C
	_, _, ecx1, _ := cpuid(1, 0)
	return ecx1&f16cBit != 0
}

func detectAVX2() bool {
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" {
		return false
	}
	const (
		fmaBit     = 1 << 12 // CPUID.1:ECX.FMA
		osxsaveBit = 1 << 27 // CPUID.1:ECX.OSXSAVE
		avxBit     = 1 << 28 // CPUID.1:ECX.AVX
		avx2Bit    = 1 << 5  // CPUID.7:EBX.AVX2
	)
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&(fmaBit|osxsaveBit|avxBit) != (fmaBit | osxsaveBit | avxBit) {
		return false
	}
	// Confirm the OS has enabled XMM (bit 1) and YMM (bit 2) state saving.
	if xgetbv()&0x6 != 0x6 {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&avx2Bit != 0
}
