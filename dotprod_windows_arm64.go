//go:build windows && arm64

package gopherllm

import "syscall"

// Windows exposes CPU features through IsProcessorFeaturePresent rather than an
// auxiliary vector. PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE is the documented query
// for FEAT_DotProd (the ARMv8.2 DP instructions, i.e. SDOT/UDOT).
//
// syscall.NewLazyDLL resolves kernel32 on first use, so a Windows build that
// never reaches an int8 kernel never pays for the lookup. Every shipping Windows
// on ARM part (Snapdragon 8cx Gen 3 and later, X Elite) implements dotprod, but
// the query is one call at init and keeps the SIGILL impossible rather than
// merely unlikely.
const pfARMV82DPInstructionsAvailable = 43

var (
	kernel32DLL                   = syscall.NewLazyDLL("kernel32.dll")
	procIsProcessorFeaturePresent = kernel32DLL.NewProc("IsProcessorFeaturePresent")
)

func probeDotProd() bool {
	// IsProcessorFeaturePresent returns a BOOL and does not set a meaningful
	// last error; a zero return simply means "not present".
	r, _, _ := procIsProcessorFeaturePresent.Call(uintptr(pfARMV82DPInstructionsAvailable))
	return r != 0
}
