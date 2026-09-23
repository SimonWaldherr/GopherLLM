// Package numeric contains small numeric conversions shared by internal
// model and format packages.
package numeric

import "math"

var halfToFloat32 = func() [1 << 16]float32 {
	var table [1 << 16]float32
	for i := range table {
		table[i] = convertHalf(uint16(i))
	}
	return table
}()

// F16ToF32 converts an IEEE 754 binary16 bit pattern to binary32.
func F16ToF32(h uint16) float32 { return halfToFloat32[h] }

func convertHalf(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		var e uint32
		for mant&0x400 == 0 {
			mant <<= 1
			e++
		}
		mant &= 0x3ff
		return math.Float32frombits((sign << 31) | ((127 - 15 + 1 - e) << 23) | (mant << 13))
	}
	if exp == 31 {
		return math.Float32frombits((sign << 31) | (0xff << 23) | (mant << 13))
	}
	return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}
