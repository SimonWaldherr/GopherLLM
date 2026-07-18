//go:build amd64

package gopherllm

// withF16KVCache runs fn with the f16 KV cache forced on or off. The
// generationWorkspace compatibility check keys on useF16KVCache, so cached
// workspaces from the other mode are discarded automatically.
func withF16KVCache(enabled bool, fn func()) {
	saved := useF16KVCache
	useF16KVCache = enabled && hasAVX2 && hasF16C
	defer func() { useF16KVCache = saved }()
	fn()
}
