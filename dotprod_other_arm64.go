//go:build arm64 && !darwin && !linux && !windows

package gopherllm

// The BSDs, Solaris/illumos, Plan 9 and js/wasi arm64 targets each expose CPU
// features differently (elf_aux_info on FreeBSD, sysctl machdep.* on OpenBSD,
// nothing portable elsewhere), and none of them are targets this engine is
// tuned or tested on today.
//
// Reporting no dotprod here means those targets keep exactly the behaviour they
// had before the arm64 kernels were un-gated from darwin: the portable scalar
// int8 path, which is correct everywhere. Adding a real probe for one of them
// is a self-contained change — implement probeDotProd in a new
// dotprod_<goos>_arm64.go and drop that GOOS from this file's constraint.
func probeDotProd() bool { return false }
