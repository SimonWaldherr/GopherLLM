//go:build !arm64

package engine

func qwen35DeltaSilu(values []float32) {
	for i := range values {
		values[i] *= fastSigmoidF32(values[i])
	}
}

func qwen35DeltaGate(gate, out []float32) {
	for i, g := range gate {
		out[i] *= g * fastSigmoidF32(g)
	}
}

func qwen35DeltaHead(state, q, k, v, out, recall []float32, decay, beta float32) {
	qwen35DeltaHeadFallback(state, q, k, v, out, recall, decay, beta)
}
