package gopherllm

import (
	"math"
)

func normalizeDecoderInto(config Config, x, weight, bias []float32, out *[]float32) {
	if config.UseLayerNorm {
		layerNormInto(x, weight, bias, config.RMSNormEps, out)
		return
	}
	rmsNormInto(x, weight, config.RMSNormEps, out)
}

// geluTanh is the tanh-approximated GELU (gelu_pytorch_tanh) used by the
// Gemma family's FFN in place of SiLU.
func geluTanh(x float32) float32 { return geluTanhScalar(x) }

// softcapF32 applies v = cap*tanh(v/cap) elementwise — Gemma's logit
// softcapping, which bounds values to (-cap, cap) while staying smooth.
// Gemma applies this to the whole logits vector, so on a 256k-entry vocabulary
// it is a quarter-million tanh calls per token — by far the heaviest consumer of
// tanh in the engine, and the reason it uses the float32 one.
func softcapF32(v []float32, cap float32) {
	inv := 1 / cap
	for i, x := range v {
		v[i] = cap * fastTanhF32(x*inv)
	}
}

// perHeadRMSNormInPlace RMS-normalizes each head's headDim-wide slice of vec
// independently against a shared headDim-length weight — Gemma 3/4-style
// QK-norm, applied to the projected Q/K before RoPE.
func perHeadRMSNormInPlace(vec []float32, headDim, nHeads int, weight []float32, eps float32) {
	if len(weight) < headDim {
		return
	}
	for h := 0; h < nHeads; h++ {
		off := h * headDim
		if off+headDim > len(vec) {
			break
		}
		sub := vec[off : off+headDim]
		ss := DotF32(sub, sub)
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+eps)))
		mulScaleF32(sub, weight[:headDim], scale, sub)
	}
}

func normalizeProjectedQKInPlace(config Config, layer LayerWeights, q, k []float32) {
	if config.usesFullProjectionQKNorm() {
		if layer.AttnQNorm != nil {
			rmsNormInto(q, layer.AttnQNorm, config.RMSNormEps, &q)
		}
		if layer.AttnKNorm != nil {
			rmsNormInto(k, layer.AttnKNorm, config.RMSNormEps, &k)
		}
		return
	}
	if layer.AttnQNorm != nil {
		perHeadRMSNormInPlace(q, config.HeadDim, config.NHeads, layer.AttnQNorm, config.RMSNormEps)
	}
	if layer.AttnKNorm != nil {
		perHeadRMSNormInPlace(k, config.HeadDim, config.NKVHeads, layer.AttnKNorm, config.RMSNormEps)
	}
}

// rmsNormInto writes out[i] = x[i] / rms(x) * weight[i] where
// rms(x) = sqrt(mean(x²) + eps) — RMSNorm as used by all supported
// architectures (no mean subtraction, no bias).
func rmsNormInto(x, weight []float32, eps float32, out *[]float32) {
	n := len(x)
	ensureLenNoClear(out, n)
	if n == 0 {
		return
	}
	ss := DotF32(x, x)
	scale := float32(1 / math.Sqrt(float64(ss/float32(n)+eps)))

	o := *out
	_ = o[n-1]
	_ = x[n-1]

	if len(weight) >= n {
		mulScaleF32(x[:n], weight[:n], scale, o[:n])
	} else {
		for i := 0; i < n; i++ {
			w := float32(1)
			if i < len(weight) {
				w = weight[i]
			}
			o[i] = x[i] * scale * w
		}
	}
}
