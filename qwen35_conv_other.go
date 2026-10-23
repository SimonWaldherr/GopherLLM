//go:build !arm64

package gopherllm

func qwen35Conv4(kernel, history, values []float32) bool {
	return false
}
