//go:build !js

package gopherllm

// GPUWeight is a no-op placeholder on non-js builds -- WebGPU is only ever
// reachable from a browser tab. See webgpu_js.go for the real
// implementation.
type GPUWeight struct{}

// WebGPUAvailable always reports false outside a js/wasm build.
func WebGPUAvailable() bool { return false }

// SetWebGPUForceDisabled is a no-op outside a js/wasm build.
func SetWebGPUForceDisabled(disabled bool) {}

func releaseWebGPUWeight(w *GPUWeight) {}

func prepareWebGPUWeight(data []byte, typ GGMLType, rows, cols int) *GPUWeight { return nil }

func matvecWebGPUQ4KInto(w *GPUWeight, x []float32, rows, cols int, out *[]float32) bool {
	return false
}

func matvecWebGPUQ6KInto(w *GPUWeight, x []float32, rows, cols int, out *[]float32) bool {
	return false
}
