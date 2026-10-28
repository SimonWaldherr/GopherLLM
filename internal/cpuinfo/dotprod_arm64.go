//go:build arm64

package cpuinfo

import "os"

// HasDotProd reports whether ARM dot-product instructions (SDOT/UDOT) may be used.
// It honours GOPHERLLM_DISABLE_SIMD so the scalar path remains available.
//
// Dot-product support is optional before ARMv8.4. The platform probe checks OS
// feature flags or a guaranteed baseline; it never executes optional instructions.
// Callers should cache the result before selecting their kernels.
func HasDotProd() bool {
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" {
		return false
	}
	return probeDotProd()
}
