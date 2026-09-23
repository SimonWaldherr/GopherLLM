package gopherllm

import "github.com/SimonWaldherr/GopherLLM/internal/numeric"

// Keep the historical package-local names while sharing implementations
// through the internal numeric package.
func fastExpF32(x float32) float32       { return numeric.ExpF32(x) }
func fastSigmoidF32(x float32) float32   { return numeric.SigmoidF32(x) }
func fastTanhF32(x float32) float32      { return numeric.TanhF32(x) }
func geluTanhScalar(x float32) float32   { return numeric.GELUTanhScalar(x) }
func geluMulF32(gate, up, out []float32) { numeric.GELUMulF32(gate, up, out) }
