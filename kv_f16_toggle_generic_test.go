//go:build !amd64

package gopherllm

// withF16KVCache lets non-amd64 tests exercise the f16 cache, including the
// Apple Silicon SIMD implementation and portable scalar fallbacks.
func withF16KVCache(enabled bool, fn func()) {
	saved := useF16KVCache.Load()
	useF16KVCache.Store(enabled)
	defer useF16KVCache.Store(saved)
	fn()
}
