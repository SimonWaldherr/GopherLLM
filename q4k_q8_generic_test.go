//go:build !amd64

package gopherllm

// withQ8Activations runs fn with the int8-activation path forced on or off.
// On non-amd64 targets the path does not exist (useQ8Activations is a
// compile-time false), so fn just runs as-is.
func withQ8Activations(enabled bool, fn func()) {
	fn()
}
