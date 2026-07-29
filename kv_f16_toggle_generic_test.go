//go:build !amd64

package gopherllm

// withF16KVCache lets non-amd64 tests exercise the f16 cache, including the
// Apple Silicon SIMD implementation and portable scalar fallbacks.
func withF16KVCache(enabled bool, fn func()) {
	saved := useF16KVCache
	useF16KVCache = enabled
	defer func() { useF16KVCache = saved }()
	fn()
}
