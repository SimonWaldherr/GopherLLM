//go:build !arm64

package numeric

// AllFiniteF32 reports whether every value is finite. It accepts subnormals
// and both signs of zero, and rejects every NaN payload and either infinity.
func AllFiniteF32(values []float32) bool {
	return allFiniteF32Generic(values)
}
