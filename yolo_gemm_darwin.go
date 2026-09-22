//go:build darwin && cgo

package gopherllm

import "github.com/SimonWaldherr/GopherLLM/internal/voxtralblas"

// yoloGEMM computes c[m×n] = a[m×k] · b[k×n] (row-major) through
// Accelerate's multithreaded sgemm, which is several times faster than the
// portable tiled kernel for the large, dense products a YOLO forward pass is
// made of.
func yoloGEMM(m, n, k int, a, b, c []float32) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	voxtralblas.GemmNN(m, n, k, a, b, c)
}
