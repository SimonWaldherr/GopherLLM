package numeric

// AllFiniteF32 reports whether every value is finite. It accepts subnormals
// and both signs of zero, and rejects every NaN payload and either infinity.
func AllFiniteF32(values []float32) bool {
	if len(values) < 16 {
		return allFiniteF32Generic(values)
	}
	return allFiniteF32(&values[0], len(values))
}

//go:noescape
func allFiniteF32(values *float32, count int) bool
