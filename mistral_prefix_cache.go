package gopherllm

import "strings"

// DefaultMistralPromptPrefixCacheBytes is the bounded storage budget used when
// EnableMistralPromptPrefixCache receives a non-positive limit. The cache is
// deliberately opt-in because it retains system prompts and tool definitions
// in memory between requests.
const DefaultMistralPromptPrefixCacheBytes = 8 << 20

// mistralPromptPrefixCacheMaxEntries keeps a stream of distinct, tiny
// prefixes from accumulating map and LRU bookkeeping. The byte budget remains
// the primary bound for large system prompts or tool schemas.
const mistralPromptPrefixCacheMaxEntries = 32

// MistralPromptPrefixCacheStats describes the opt-in render-time cache. It is
// separate from GenerationResult.PromptCache: the latter caches KV rows for a
// complete token prefix, while this cache only avoids repeatedly tokenizing
// Mistral's static BOS/system/tools prefix during rendering and context-window
// planning.
type MistralPromptPrefixCacheStats struct {
	Enabled  bool
	MaxBytes int
	Entries  int
	Bytes    int
	Hits     uint64
	Misses   uint64
}

type mistralPromptPrefixCacheKey struct {
	// system is deliberately empty when the tokenizer has no dedicated system
	// markers: that legacy template folds it into the final user turn, which is
	// dynamic content and must never be cached here.
	system    string
	toolsJSON string
}

func (k mistralPromptPrefixCacheKey) cacheable() bool {
	return k.system != "" || k.toolsJSON != ""
}

type mistralPromptPrefixCacheEntry struct {
	tokens []uint32
	bytes  int
}

type mistralPromptPrefixCache struct {
	enabled  bool
	maxBytes int
	bytes    int
	hits     uint64
	misses   uint64
	// epoch prevents a concurrent render that missed before Clear/Disable from
	// repopulating the cache after the explicit invalidation boundary.
	epoch   uint64
	entries map[mistralPromptPrefixCacheKey]mistralPromptPrefixCacheEntry
	// order is least-recently-used first. The cache has at most 32 entries, so
	// a compact slice costs less than a linked-list allocation per entry.
	order []mistralPromptPrefixCacheKey
}

// EnableMistralPromptPrefixCache enables a per-Runner, LRU-bounded cache for
// the static prefix that Mistral/Ministral templates put before all chat
// turns: BOS, [SYSTEM_PROMPT], and [AVAILABLE_TOOLS]. It is useful when a
// caller repeatedly uses the same system prompt and tool schema, especially
// with PrepareChatContext's candidate rendering.
//
// It intentionally does not cache messages, image embeddings, logits, or KV
// rows. The normal bounded KV prefix cache already covers an exact text-only
// conversation history during generation; keeping this narrower avoids
// duplicating it and means image-dependent content can never be reused here.
// Calling Enable again clears prior entries and applies the new limit. Pass a
// non-positive limit to use DefaultMistralPromptPrefixCacheBytes.
func (r *Runner) EnableMistralPromptPrefixCache(maxBytes int) {
	if r == nil {
		return
	}
	if maxBytes <= 0 {
		maxBytes = DefaultMistralPromptPrefixCacheBytes
	}
	r.mistralPrefixCacheMu.Lock()
	epoch := r.mistralPrefixRenderCache.epoch + 1
	r.mistralPrefixRenderCache = mistralPromptPrefixCache{
		enabled:  true,
		maxBytes: maxBytes,
		epoch:    epoch,
	}
	r.mistralPrefixCacheMu.Unlock()
}

// DisableMistralPromptPrefixCache discards all retained render-time prefix
// tokens. It does not affect the generation KV prefix cache.
func (r *Runner) DisableMistralPromptPrefixCache() {
	if r == nil {
		return
	}
	r.mistralPrefixCacheMu.Lock()
	r.mistralPrefixRenderCache = mistralPromptPrefixCache{epoch: r.mistralPrefixRenderCache.epoch + 1}
	r.mistralPrefixCacheMu.Unlock()
}

// ClearMistralPromptPrefixCache invalidates cached static prefixes while
// keeping the configured cache enabled. It is useful when an application
// rotates sensitive system instructions or wants an explicit cache boundary.
func (r *Runner) ClearMistralPromptPrefixCache() {
	if r == nil {
		return
	}
	r.mistralPrefixCacheMu.Lock()
	cache := &r.mistralPrefixRenderCache
	cache.epoch++
	cache.bytes = 0
	cache.hits = 0
	cache.misses = 0
	cache.entries = nil
	cache.order = nil
	r.mistralPrefixCacheMu.Unlock()
}

// MistralPromptPrefixCacheStats returns a consistent snapshot of the opt-in
// render-time cache. It never reports the separate generation KV cache.
func (r *Runner) MistralPromptPrefixCacheStats() MistralPromptPrefixCacheStats {
	if r == nil {
		return MistralPromptPrefixCacheStats{}
	}
	r.mistralPrefixCacheMu.Lock()
	cache := &r.mistralPrefixRenderCache
	stats := MistralPromptPrefixCacheStats{
		Enabled:  cache.enabled,
		MaxBytes: cache.maxBytes,
		Entries:  len(cache.entries),
		Bytes:    cache.bytes,
		Hits:     cache.hits,
		Misses:   cache.misses,
	}
	r.mistralPrefixCacheMu.Unlock()
	return stats
}

// appendMistralPromptPrefix appends a cached immutable static prefix directly
// to dst. Holding the mutex while append copies the tokens prevents callers
// from ever receiving the cache's backing slice, so later render appends
// cannot corrupt a resident entry.
func (r *Runner) appendMistralPromptPrefix(dst []uint32, key mistralPromptPrefixCacheKey) ([]uint32, uint64, bool) {
	if r == nil || !key.cacheable() {
		return dst, 0, false
	}
	r.mistralPrefixCacheMu.Lock()
	cache := &r.mistralPrefixRenderCache
	epoch := cache.epoch
	if !cache.enabled {
		r.mistralPrefixCacheMu.Unlock()
		return dst, epoch, false
	}
	entry, ok := cache.entries[key]
	if !ok {
		cache.misses++
		r.mistralPrefixCacheMu.Unlock()
		return dst, epoch, false
	}
	cache.hits++
	cache.touch(key)
	dst = append(dst, entry.tokens...)
	r.mistralPrefixCacheMu.Unlock()
	return dst, epoch, true
}

// putMistralPromptPrefix inserts a completed static prefix after a miss. Its
// input is copied to an exact-length slice, because the renderer's output
// buffer deliberately has spare capacity for dynamic message tokens.
func (r *Runner) putMistralPromptPrefix(key mistralPromptPrefixCacheKey, tokens []uint32, epoch uint64) {
	if r == nil || !key.cacheable() || len(tokens) == 0 {
		return
	}
	r.mistralPrefixCacheMu.Lock()
	cache := &r.mistralPrefixRenderCache
	if !cache.enabled || cache.epoch != epoch {
		r.mistralPrefixCacheMu.Unlock()
		return
	}

	// Check against the configured cap before doing the multiplication or
	// addition. That keeps a malformed, enormous input from overflowing an
	// int and accidentally bypassing the cache's bound.
	if len(key.system) > cache.maxBytes || len(key.toolsJSON) > cache.maxBytes-len(key.system) {
		r.mistralPrefixCacheMu.Unlock()
		return
	}
	keyBytes := len(key.system) + len(key.toolsJSON)
	if len(tokens) > (cache.maxBytes-keyBytes)/4 {
		r.mistralPrefixCacheMu.Unlock()
		return
	}
	entryBytes := keyBytes + len(tokens)*4
	// Keep exact copies of input strings. In particular, a caller's system
	// prompt can be a small substring of a much larger request buffer; retaining
	// it directly would defeat the advertised byte bound.
	storedKey := mistralPromptPrefixCacheKey{
		system:    strings.Clone(key.system),
		toolsJSON: strings.Clone(key.toolsJSON),
	}
	if old, ok := cache.entries[storedKey]; ok {
		cache.bytes -= old.bytes
		cache.removeOrder(storedKey)
		delete(cache.entries, storedKey)
	}
	for len(cache.order) >= mistralPromptPrefixCacheMaxEntries || cache.bytes > cache.maxBytes-entryBytes {
		if len(cache.order) == 0 {
			break
		}
		cache.evictOldest()
	}
	if cache.entries == nil {
		cache.entries = make(map[mistralPromptPrefixCacheKey]mistralPromptPrefixCacheEntry)
	}
	storedTokens := make([]uint32, len(tokens))
	copy(storedTokens, tokens)
	cache.entries[storedKey] = mistralPromptPrefixCacheEntry{tokens: storedTokens, bytes: entryBytes}
	cache.order = append(cache.order, storedKey)
	cache.bytes += entryBytes
	r.mistralPrefixCacheMu.Unlock()
}

func (c *mistralPromptPrefixCache) touch(key mistralPromptPrefixCacheKey) {
	for i, candidate := range c.order {
		if candidate != key {
			continue
		}
		copy(c.order[i:], c.order[i+1:])
		c.order[len(c.order)-1] = key
		return
	}
}

func (c *mistralPromptPrefixCache) removeOrder(key mistralPromptPrefixCacheKey) {
	for i, candidate := range c.order {
		if candidate != key {
			continue
		}
		copy(c.order[i:], c.order[i+1:])
		c.order = c.order[:len(c.order)-1]
		return
	}
}

func (c *mistralPromptPrefixCache) evictOldest() {
	if len(c.order) == 0 {
		return
	}
	key := c.order[0]
	if entry, ok := c.entries[key]; ok {
		c.bytes -= entry.bytes
		delete(c.entries, key)
	}
	copy(c.order, c.order[1:])
	c.order = c.order[:len(c.order)-1]
}
