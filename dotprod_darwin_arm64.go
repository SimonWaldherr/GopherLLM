//go:build darwin && arm64

package gopherllm

// Every shipping Apple Silicon Mac implements FEAT_DotProd: M1 and later are
// ARMv8.5, well past the ARMv8.4 baseline that makes SDOT mandatory. So on the
// targets this engine actually supports there is nothing to probe —
// sysctlbyname("hw.optional.arm.FEAT_DotProd") would need cgo or a syscall
// wrapper to answer a question with one possible answer.
//
// Note this is a claim about Macs, not about arm64 Apple silicon generally:
// FEAT_DotProd is optional in ARMv8.2/8.3, so A11/A12-class parts do NOT have
// it. GOOS=ios also satisfies this file's constraint and would take a SIGILL on
// such a part — but iOS is not a target here (it needs cgo plus an Xcode
// toolchain to link at all, and nothing in this repo builds for it), and this
// is exactly the assumption the file made when it was gated darwin && arm64.
// If iOS ever becomes a target, retag this file `darwin && arm64 && !ios` AND
// add an ios_arm64 file returning false — dotprod_other_arm64.go's !darwin
// constraint excludes ios, so retagging alone would fail to compile.
func probeDotProd() bool { return true }
