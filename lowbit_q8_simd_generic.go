//go:build !amd64 && !arm64

package gopherllm

// Non-x86/non-arm64 targets currently have only portable scalar Q8 dots.  A
// caller may enable those dots for experimentation or autotuning, but they do
// not make separate Q2_K/Q3_K matvecs preferable to float multi-matrix fusion.
func lowBitQ8SIMDEnabled() bool { return false }
