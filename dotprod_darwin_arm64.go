//go:build darwin && arm64 && !ios

package gopherllm

// Every shipping Apple Silicon Mac implements FEAT_DotProd: M1 and later are
// ARMv8.5, well past the ARMv8.4 baseline that makes SDOT mandatory. So on the
// targets this engine actually supports there is nothing to probe —
// sysctlbyname("hw.optional.arm.FEAT_DotProd") would need cgo or a syscall
// wrapper to answer a question with one possible answer.
//
// Note this is a claim about Macs, not about arm64 Apple silicon generally:
// FEAT_DotProd is optional in ARMv8.2/8.3, so A11/A12-class parts do NOT have
// it. iOS deliberately has a separate conservative implementation: some
// supported arm64 iPhones do not implement SDOT, so executing it there would
// be a SIGILL rather than a recoverable feature probe failure.
func probeDotProd() bool { return true }
