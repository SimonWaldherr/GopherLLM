//go:build !amd64

package gopherllm

// withQ8Activations runs fn with the int8-activation path forced on or off.
// Non-amd64 targets also expose this switch: Apple Silicon has an SDOT-backed
// implementation, while other targets use the portable kernel when enabled.
func withQ8Activations(enabled bool, fn func()) {
	saved := useQ8Activations
	useQ8Activations = enabled
	defer func() { useQ8Activations = saved }()
	fn()
}
