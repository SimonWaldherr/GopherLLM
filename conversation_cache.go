package gopherllm

// DefaultConversationCacheBytes caps retained historical KV rows and token IDs
// separately from the live generation workspace.
const DefaultConversationCacheBytes int64 = 128 << 20
const conversationCacheMaxEntries = 4
const conversationCacheMinTokens = 32

type conversationSnapshot struct {
	tokens []uint32
	cache  *KVCache
	bytes  int64
}

// conversationCache retains immutable attention-only snapshots of displaced
// conversations. Runner.genLock protects it; the live workspace stays separate.
type conversationCache struct {
	entries []conversationSnapshot // oldest first
	bytes   int64
	limit   int64 // zero uses the default; negative disables snapshots
}

// SetConversationCacheLimit bounds extra retained KV snapshots per Runner.
// Positive limits are capped at 128 MiB; zero disables and clears this cache.
// The existing most-recent workspace cache remains available. Snapshots are
// enabled by default for attention-only text models, with at most four entries.
func (r *Runner) SetConversationCacheLimit(bytes int64) {
	if r == nil {
		return
	}
	r.genLock.Lock()
	defer r.genLock.Unlock()
	if bytes <= 0 {
		bytes = -1
	} else {
		bytes = min(bytes, DefaultConversationCacheBytes)
	}
	r.conversations = conversationCache{limit: bytes}
}

// ClearConversationCache drops both historical snapshots and the resident
// prompt's reuse metadata. It waits for any active generation to finish.
func (r *Runner) ClearConversationCache() {
	if r == nil {
		return
	}
	r.genLock.Lock()
	defer r.genLock.Unlock()
	r.conversations = conversationCache{limit: r.conversations.limit}
	r.clearPrefixCache()
}

func (r *Runner) conversationCacheSupported() bool {
	return r.conversations.limit >= 0 && !r.outOfCore &&
		(r.kind == loadedStandard && !r.config.UsesMLA || r.kind == loadedGemma4)
}

// retainDisplacedConversation runs before any snapshot is restored or prompt
// rows are overwritten. Appending to the resident sequence needs no copy.
func (r *Runner) retainDisplacedConversation(tokens []uint32) {
	state := r.prefixCache
	if !r.conversationCacheSupported() || state.cache == nil || len(state.tokens) < conversationCacheMinTokens || len(state.tokens) > state.cache.MaxLen {
		return
	}
	matched := sharedTokenPrefix(tokens, state.tokens)
	if matched == len(state.tokens) || (len(tokens) == state.promptTokens && matched == len(tokens)) {
		return
	}

	c := &r.conversations
	limit := c.limit
	if limit == 0 {
		limit = DefaultConversationCacheBytes
	}
	n := len(state.tokens)
	bytes := kvCacheBytes(state.cache.layerCount(), state.cache.PerPosKDim, state.cache.PerPosVDim, n, state.cache.kvFormat()) + int64(n)*4
	if bytes > limit {
		return
	}
	// A newer longer snapshot subsumes an older branch only when all of the
	// older tokens match. Divergent branches retain their own state.
	for i := len(c.entries) - 1; i >= 0; i-- {
		e := c.entries[i]
		if sharedTokenPrefix(e.tokens, state.tokens) == len(e.tokens) {
			c.bytes -= e.bytes
			c.remove(i)
		}
	}
	for len(c.entries) > 0 && (len(c.entries) >= conversationCacheMaxEntries || c.bytes+bytes > limit) {
		c.bytes -= c.entries[0].bytes
		c.remove(0)
	}
	snapshot := cloneKVPrefix(state.cache, n)
	if snapshot == nil {
		return
	}
	c.entries = append(c.entries, conversationSnapshot{tokens: append([]uint32(nil), state.tokens...), cache: snapshot, bytes: bytes})
	c.bytes += bytes
}

// Restore only an exact token prefix longer than the resident match. Keeping
// at least one token for prefill supplies fresh logits for an identical prompt;
// immutable snapshots intentionally do not retain vocabulary-sized logits.
func (r *Runner) reuseConversation(cache *KVCache, tokens []uint32, resident int) int {
	if !r.conversationCacheSupported() {
		return resident
	}
	c := &r.conversations
	best, index := resident, -1
	for i, e := range c.entries {
		n := min(sharedTokenPrefix(tokens, e.tokens), len(tokens)-1)
		if n >= conversationCacheMinTokens && n > best && kvPrefixSnapshotCompatible(cache, e.cache, n) {
			best, index = n, i
		}
	}
	// Pin the candidate before retaining the displaced workspace: even if
	// the byte limit evicts its LRU entry, this immutable source remains valid.
	var entry conversationSnapshot
	if index >= 0 {
		entry = c.entries[index]
		copy(c.entries[index:], c.entries[index+1:])
		c.entries[len(c.entries)-1] = entry
	}
	r.retainDisplacedConversation(tokens)
	if index < 0 || copyKVPrefix(cache, entry.cache, best) != best {
		return resident
	}

	return best
}

func (c *conversationCache) remove(i int) {
	copy(c.entries[i:], c.entries[i+1:])
	c.entries[len(c.entries)-1] = conversationSnapshot{}
	c.entries = c.entries[:len(c.entries)-1]
}
