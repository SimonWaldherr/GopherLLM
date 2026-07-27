//go:build !amd64

package gopherllm

// withF16KVCache lets generic-platform tests exercise the scalar f16 cache.
// Production keeps it opt-in because its conversion path is not SIMD-backed.
func withF16KVCache(enabled bool, fn func()) {
	saved := useF16KVCache
	useF16KVCache = enabled
	defer func() { useF16KVCache = saved }()
	fn()
}
