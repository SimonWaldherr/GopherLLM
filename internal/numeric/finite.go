package numeric

import "math"

func allFiniteF32Generic(values []float32) bool {
	for _, value := range values {
		if math.Float32bits(value)&0x7f800000 == 0x7f800000 {
			return false
		}
	}
	return true
}
