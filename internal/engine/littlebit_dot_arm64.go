package engine

//go:noescape
func littleBitDotNEON(packed *byte, x *float32, groups int) float32

func littleBitDot(packed []byte, x []float32) float32 {
	groups := len(x) / 8
	if groups == 0 {
		return littleBitDotScalar(packed, x)
	}
	sum := littleBitDotNEON(&packed[0], &x[0], groups)
	return sum + littleBitDotScalar(packed[groups:], x[groups*8:])
}

const hasFastLittleBitDot4 = true

//go:noescape
func littleBitDot4NEON(packed *byte, x0, x1, x2, x3 *float32, groups int) (s0, s1, s2, s3 float32)

func littleBitDot4(packed []byte, x0, x1, x2, x3 []float32) (float32, float32, float32, float32) {
	groups := len(x0) / 8
	var a, b, c, d float32
	if groups != 0 {
		a, b, c, d = littleBitDot4NEON(&packed[0], &x0[0], &x1[0], &x2[0], &x3[0], groups)
	}
	tail := packed[groups:]
	return a + littleBitDotScalar(tail, x0[groups*8:]), b + littleBitDotScalar(tail, x1[groups*8:]),
		c + littleBitDotScalar(tail, x2[groups*8:]), d + littleBitDotScalar(tail, x3[groups*8:])
}

const hasFastLittleBitDotRows4 = true

//go:noescape
func littleBitDotRows4NEON(packed0, packed1, packed2, packed3 *byte, x *float32, groups int) (s0, s1, s2, s3 float32)

// littleBitDotRows4 dots one activation vector against four packed rows,
// sharing every x load across the rows instead of reloading x per row.
func littleBitDotRows4(packed0, packed1, packed2, packed3 []byte, x []float32) (float32, float32, float32, float32) {
	groups := len(x) / 8
	var a, b, c, d float32
	if groups != 0 {
		a, b, c, d = littleBitDotRows4NEON(&packed0[0], &packed1[0], &packed2[0], &packed3[0], &x[0], groups)
	}
	tail := x[groups*8:]
	return a + littleBitDotScalar(packed0[groups:], tail), b + littleBitDotScalar(packed1[groups:], tail),
		c + littleBitDotScalar(packed2[groups:], tail), d + littleBitDotScalar(packed3[groups:], tail)
}
