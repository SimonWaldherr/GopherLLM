//go:build !darwin

package gopherllm

// performanceCores returns 0 on platforms where no asymmetric-core query is
// wired up, meaning "no distinct performance-core count to prefer".
//
// x86 hybrid parts (Intel P/E cores from Alder Lake on) are asymmetric too, but
// they expose the split through CPUID leaf 0x1A per logical CPU rather than a
// single sysctl, and the autotuner already searches nproc/2 and nproc*3/4 which
// brackets the usual P-core counts there. Apple Silicon is the case where the
// P-core count (8 of 12 on an M2 Max) falls outside that bracket, so that is
// the platform with a real implementation. See cores_darwin.go.
func performanceCores() int { return 0 }
