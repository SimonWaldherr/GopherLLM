//go:build ios

package cpuinfo

// iOS does not expose macOS's hw.perflevel sysctls as a stable application
// contract. Let the existing normal worker-count policy choose a conservative
// value instead of relying on private hardware classification.
func PerformanceCores() int { return 0 }
