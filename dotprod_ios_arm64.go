//go:build ios && arm64

package gopherllm

// Dot-product instructions are optional on the ARM CPUs used by iPhones.
// Do not probe by executing an instruction: an older device would terminate
// the process with SIGILL. Keeping the portable NEON path is safe on devices
// and simulators; a future OS-supported feature query may replace this.
func probeDotProd() bool { return false }
