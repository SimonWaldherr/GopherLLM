package gopherllm

import "encoding/binary"

// DefaultMistralKVPrefixCacheBytes is the default cap for the opt-in KV
// snapshot cache used by Mistral/Ministral's static system-and-tools prefix.
// It is deliberately smaller than maxReusableKVCacheBytes: the normal Runner
// workspace remains useful for the most recent complete prompt, while this
// cache adds a small set of immutable snapshots for otherwise unrelated
// conversations which share their system prompt and tool schema.
const DefaultMistralKVPrefixCacheBytes int64 = 128 << 20

// mistralKVPrefixCacheMaxBytes is a hard ceiling even when a caller asks for a
// larger cache. A KV snapshot is an optimization, not a reason for one Runner
// to retain an unbounded number of copies of sensitive prompt state.
const mistralKVPrefixCacheMaxBytes int64 = maxReusableKVCacheBytes

// mistralKVPrefixCacheMaxEntries prevents a stream of tiny, distinct prefixes
// from retaining unbounded map/LRU bookkeeping. The byte cap remains the
// primary limit for real Ministral system prompts and tool schemas.
const mistralKVPrefixCacheMaxEntries = 8

// MistralKVPrefixCacheStats describes the opt-in cross-conversation KV prefix
// cache. It is distinct from GenerationResult.PromptCache, which reports the
// actual prefix reuse for one generation, including the normal most-recent
// workspace cache.
//
// Entries retain only token IDs and immutable K/V rows for the static Mistral
// prefix -- never messages, image embeddings, logits, or generated output.
type MistralKVPrefixCacheStats struct {
	Enabled  bool
	MaxBytes int64
	Entries  int
	Bytes    int64
	Hits     uint64
	Misses   uint64
}

// mistralKVPrefixCacheEntry is immutable after insertion. That lets a lookup
// take a local pointer, release mistralKVPrefixCacheMu, then copy from the
// snapshot into the generation workspace without allowing an LRU eviction to
// invalidate its source.
type mistralKVPrefixCacheEntry struct {
	cache *KVCache
	bytes int64
}

type mistralKVPrefixCache struct {
	enabled  bool
	maxBytes int64
	bytes    int64
	hits     uint64
	misses   uint64
	// epoch prevents an in-flight generation that began before Clear/Disable
	// from retaining a snapshot after that explicit invalidation boundary.
	epoch uint64
	// entries and order are keyed by the exact static token sequence. They are
	// per Runner, so model weights and tokenizer ownership never cross a model
	// boundary; different system prompts, tool schemas, or renderer output only
	// share when they produce the same token IDs, which is the only state K/V
	// rows can safely depend on.
	entries map[string]mistralKVPrefixCacheEntry
	order   []string // least-recently-used first
}

// EnableMistralKVPrefixCache enables a per-Runner, LRU-bounded cache of
// immutable K/V snapshots for the static prefix emitted by the Mistral and
// Ministral [INST] template: BOS, [SYSTEM_PROMPT], and [AVAILABLE_TOOLS].
//
// It accelerates separate conversations that have the same system prompt and
// tool definitions but different first user turns. The existing bounded
// prefix cache already handles the immediately preceding complete prompt;
// this opt-in cache covers branches which would otherwise evict that one
// workspace prefix. It is disabled by default because K/V rows retain a
// model-derived representation of system prompts and tool definitions.
//
// Pass a non-positive limit for DefaultMistralKVPrefixCacheBytes. Values above
// 512 MiB are clamped to that safety bound. Calling Enable again discards prior
// snapshots and applies the new limit. Snapshot entries are text-only and only
// used when their exact token IDs are a proper prefix of the current prompt;
// image-bearing requests never consult or populate this cache.
func (r *Runner) EnableMistralKVPrefixCache(maxBytes int64) {
	if r == nil {
		return
	}
	if maxBytes <= 0 {
		maxBytes = DefaultMistralKVPrefixCacheBytes
	}
	if maxBytes > mistralKVPrefixCacheMaxBytes {
		maxBytes = mistralKVPrefixCacheMaxBytes
	}
	r.mistralKVPrefixCacheMu.Lock()
	epoch := r.mistralKVPrefixCache.epoch + 1
	r.mistralKVPrefixCache = mistralKVPrefixCache{
		enabled:  true,
		maxBytes: maxBytes,
		epoch:    epoch,
	}
	r.mistralKVPrefixCacheMu.Unlock()
}

// DisableMistralKVPrefixCache discards all retained static-prefix K/V rows.
// It does not affect the ordinary most-recent prompt cache or the separate
// render-time Mistral prompt-prefix cache.
func (r *Runner) DisableMistralKVPrefixCache() {
	if r == nil {
		return
	}
	r.mistralKVPrefixCacheMu.Lock()
	r.mistralKVPrefixCache = mistralKVPrefixCache{epoch: r.mistralKVPrefixCache.epoch + 1}
	r.mistralKVPrefixCacheMu.Unlock()
}

// ClearMistralKVPrefixCache invalidates retained static-prefix K/V rows while
// keeping the configured cache enabled. It is useful after rotating sensitive
// system instructions or tool definitions.
func (r *Runner) ClearMistralKVPrefixCache() {
	if r == nil {
		return
	}
	r.mistralKVPrefixCacheMu.Lock()
	cache := &r.mistralKVPrefixCache
	cache.epoch++
	cache.bytes = 0
	cache.hits = 0
	cache.misses = 0
	cache.entries = nil
	cache.order = nil
	r.mistralKVPrefixCacheMu.Unlock()
}

// MistralKVPrefixCacheStats returns a consistent snapshot of the opt-in cache.
func (r *Runner) MistralKVPrefixCacheStats() MistralKVPrefixCacheStats {
	if r == nil {
		return MistralKVPrefixCacheStats{}
	}
	r.mistralKVPrefixCacheMu.Lock()
	cache := &r.mistralKVPrefixCache
	stats := MistralKVPrefixCacheStats{
		Enabled:  cache.enabled,
		MaxBytes: cache.maxBytes,
		Entries:  len(cache.entries),
		Bytes:    cache.bytes,
		Hits:     cache.hits,
		Misses:   cache.misses,
	}
	r.mistralKVPrefixCacheMu.Unlock()
	return stats
}

// mistralKVPrefixCacheEpoch reports whether snapshot storage is enabled at
// prefill start. Store uses the returned epoch as an invalidation fence once
// the complete prompt is known to have been evaluated successfully.
func (r *Runner) mistralKVPrefixCacheEpoch() (uint64, bool) {
	if r == nil {
		return 0, false
	}
	r.mistralKVPrefixCacheMu.Lock()
	cache := &r.mistralKVPrefixCache
	epoch, enabled := cache.epoch, cache.enabled
	r.mistralKVPrefixCacheMu.Unlock()
	return epoch, enabled
}

// mistralKVPrefixCacheReuse copies an immutable, exact static-prefix snapshot
// into dst. Callers already hold Runner.genLock, so dst cannot be observed or
// modified concurrently. Entries are per Runner and keyed by every static
// token, so neither a changed system/tools prompt nor a different tokenizer
// can be mistaken for a hit.
func (r *Runner) mistralKVPrefixCacheReuse(dst *KVCache, tokens []uint32) (int, bool) {
	if r == nil || dst == nil || len(tokens) == 0 {
		return 0, false
	}
	key := mistralKVPrefixTokenKey(tokens)
	r.mistralKVPrefixCacheMu.Lock()
	cache := &r.mistralKVPrefixCache
	if !cache.enabled {
		r.mistralKVPrefixCacheMu.Unlock()
		return 0, false
	}
	entry, ok := cache.entries[key]
	if !ok || !kvPrefixSnapshotCompatible(dst, entry.cache, len(tokens)) {
		cache.misses++
		r.mistralKVPrefixCacheMu.Unlock()
		return 0, false
	}
	cache.touch(key)
	cache.hits++
	r.mistralKVPrefixCacheMu.Unlock()

	if copied := copyKVPrefix(dst, entry.cache, len(tokens)); copied == len(tokens) {
		return copied, true
	}
	// The compatibility check above makes this unreachable for a normal
	// immutable entry. Treat it as cold rather than advertising stale reuse if
	// a malformed internal cache ever makes it through.
	return 0, false
}

// putMistralKVPrefixSnapshot clones a complete static prefix after prefill.
// The clone is built outside the cache mutex because it can copy tens of MiB;
// the epoch recheck at insertion makes an Enable/Clear/Disable racing that
// copy an explicit invalidation boundary rather than a stale repopulation.
func (r *Runner) putMistralKVPrefixSnapshot(src *KVCache, tokens []uint32, epoch uint64) {
	if r == nil || src == nil || len(tokens) == 0 {
		return
	}
	r.mistralKVPrefixCacheMu.Lock()
	cache := &r.mistralKVPrefixCache
	if !cache.enabled || cache.epoch != epoch {
		r.mistralKVPrefixCacheMu.Unlock()
		return
	}
	maxBytes := cache.maxBytes
	key := mistralKVPrefixTokenKey(tokens)
	// A resident entry represents precisely the same deterministic prefix. It
	// is immutable, so avoid duplicating an expensive snapshot on every hit.
	if existing, ok := cache.entries[key]; ok && kvPrefixSnapshotCompatible(src, existing.cache, len(tokens)) {
		cache.touch(key)
		r.mistralKVPrefixCacheMu.Unlock()
		return
	}
	r.mistralKVPrefixCacheMu.Unlock()

	snapshot := cloneKVPrefix(src, len(tokens))
	if snapshot == nil {
		return
	}
	// The raw-token string is the map key retained by the entry. Account for
	// it explicitly; keeping a second []uint32 copy solely for bookkeeping
	// would both duplicate this memory and make the advertised cache budget
	// misleading.
	entryBytes := kvCacheStorageBytes(snapshot) + int64(len(key))
	if entryBytes <= 0 || entryBytes > maxBytes {
		return
	}

	r.mistralKVPrefixCacheMu.Lock()
	cache = &r.mistralKVPrefixCache
	if !cache.enabled || cache.epoch != epoch || entryBytes > cache.maxBytes {
		r.mistralKVPrefixCacheMu.Unlock()
		return
	}
	if old, ok := cache.entries[key]; ok {
		cache.bytes -= old.bytes
		cache.removeOrder(key)
		delete(cache.entries, key)
	}
	for len(cache.order) >= mistralKVPrefixCacheMaxEntries || cache.bytes > cache.maxBytes-entryBytes {
		if len(cache.order) == 0 {
			break
		}
		cache.evictOldest()
	}
	if cache.entries == nil {
		cache.entries = make(map[string]mistralKVPrefixCacheEntry)
	}
	cache.entries[key] = mistralKVPrefixCacheEntry{cache: snapshot, bytes: entryBytes}
	cache.order = append(cache.order, key)
	cache.bytes += entryBytes
	r.mistralKVPrefixCacheMu.Unlock()
}

func (c *mistralKVPrefixCache) touch(key string) {
	for i, candidate := range c.order {
		if candidate != key {
			continue
		}
		copy(c.order[i:], c.order[i+1:])
		c.order[len(c.order)-1] = key
		return
	}
}

func (c *mistralKVPrefixCache) removeOrder(key string) {
	for i, candidate := range c.order {
		if candidate != key {
			continue
		}
		copy(c.order[i:], c.order[i+1:])
		c.order = c.order[:len(c.order)-1]
		return
	}
}

func (c *mistralKVPrefixCache) evictOldest() {
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

// cloneKVPrefix makes an exact-length, immutable copy of the K/V rows at
// [0, positions). It intentionally does not carry recurrent state or MTP
// state; the Mistral cache is restricted to loadedStandard, whose graph is
// fully represented by ordinary attention K/V rows.
func cloneKVPrefix(src *KVCache, positions int) *KVCache {
	if src == nil || positions <= 0 || src.layerCount() <= 0 {
		return nil
	}
	positions = min(positions, src.MaxLen)
	if positions <= 0 {
		return nil
	}
	var dst *KVCache
	switch src.kvFormat() {
	case kvI8:
		dst = NewKVCacheI8(src.layerCount(), src.PerPosKDim, src.PerPosVDim, positions)
	case kvF16:
		dst = NewKVCacheF16(src.layerCount(), src.PerPosKDim, src.PerPosVDim, positions)
	default:
		dst = NewKVCache(src.layerCount(), src.PerPosKDim, src.PerPosVDim, positions)
	}
	if copyKVPrefix(dst, src, positions) != positions {
		return nil
	}
	return dst
}

func kvPrefixSnapshotCompatible(dst, src *KVCache, positions int) bool {
	if dst == nil || src == nil || positions <= 0 || dst.kvFormat() != src.kvFormat() ||
		dst.PerPosKDim != src.PerPosKDim || dst.PerPosVDim != src.PerPosVDim ||
		dst.layerCount() != src.layerCount() {
		return false
	}
	return positions <= dst.MaxLen && positions <= src.MaxLen
}

// mistralKVPrefixTokenKey uses the exact token IDs rather than prompt text.
// K/V rows only depend on the model's token IDs, and the cache is per Runner,
// so this is both stricter than a text key across tokenizer changes and avoids
// retaining caller-owned string backing buffers in the map key.
func mistralKVPrefixTokenKey(tokens []uint32) string {
	if len(tokens) == 0 {
		return ""
	}
	// Context length is an int and a Go slice cannot be larger than max int;
	// keep the multiplication explicit so a malformed internal caller cannot
	// overflow it before make.
	if len(tokens) > int(^uint(0)>>1)/4 {
		return ""
	}
	b := make([]byte, len(tokens)*4)
	for i, token := range tokens {
		binary.LittleEndian.PutUint32(b[i*4:], token)
	}
	return string(b)
}
