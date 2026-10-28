package engine

import "math"

// finite32 classifies exponent bits without converting or operating on NaNs.
// Whole slices should use numeric.AllFiniteF32 to take its vectorized path.
func finite32(value float32) bool {
	return math.Float32bits(value)&0x7f800000 != 0x7f800000
}

// siluExact preserves the float32 division used by the speech, diffusion and
// Mamba graphs. Multiplying by sigmoid would round differently.
func siluExact(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// softmaxF64InPlace normalizes float32 values with a float64 sum. Callers can
// supply a maximum computed while scaling or filling scores. TTS and BERT use
// float32 accumulation and must keep their separate arithmetic.
func softmaxF64InPlace(values []float32, maximum float32) {
	var sum float64
	for i, value := range values {
		values[i] = float32(math.Exp(float64(value - maximum)))
		sum += float64(values[i])
	}
	for i := range values {
		values[i] /= float32(sum)
	}
}
