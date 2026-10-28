//go:build arm64

package engine

import "unsafe"

//go:noescape
func siluF32NEON(values *float32, n int)

func qwen35DeltaSilu(values []float32) {
	n4 := len(values) &^ 3
	if n4 > 0 {
		siluF32NEON(unsafe.SliceData(values), n4)
	}
	for i := n4; i < len(values); i++ {
		values[i] *= fastSigmoidF32(values[i])
	}
}

func qwen35DeltaGate(gate, out []float32) {
	siluMulF32(gate, out, out)
}

//go:noescape
func qwen35DeltaHeadNEON(state, q, k, v, out *float32, decay, beta float32, n int)

func qwen35DeltaHead(state, q, k, v, out, recall []float32, decay, beta float32) {
	n := len(v)
	if n > 0 && n%16 == 0 && n <= len(state)/n && len(q) == n && len(k) == n && len(out) >= n {
		qwen35DeltaHeadNEON(unsafe.SliceData(state), unsafe.SliceData(q), unsafe.SliceData(k),
			unsafe.SliceData(v), unsafe.SliceData(out), decay, beta, n)
		return
	}
	qwen35DeltaHeadFallback(state, q, k, v, out, recall, decay, beta)
}
