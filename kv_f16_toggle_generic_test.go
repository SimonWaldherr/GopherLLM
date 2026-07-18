//go:build !amd64

package gopherllm

// withF16KVCache: on non-amd64 targets useF16KVCache is a compile-time
// false, so fn just runs with the exact f32 cache either way.
func withF16KVCache(enabled bool, fn func()) {
	fn()
}
