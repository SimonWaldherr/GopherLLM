//go:build !darwin || ios

package gopherllm

func hostHardwareFingerprint() string { return "" }
