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
	// A long conversation can still contribute its exact initial KV rows.
	// Bound the snapshot before allocating, rather than dropping all reuse as
	// soon as the complete history exceeds the budget.
	rowBytes := kvCacheBytes(state.cache.layerCount(), state.cache.PerPosKDim, state.cache.PerPosVDim, 1, state.cache.kvFormat()) + 4
	if state.cache.layerCount() <= 0 || rowBytes <= 0 {
		return
	}
	n := int(min(int64(len(state.tokens)), limit/rowBytes))
	if n < conversationCacheMinTokens {
		return
	}
	keptTokens := state.tokens[:n]
	bytes := int64(n) * rowBytes
	// An immutable longer snapshot already owns these same rows. Retaining it
	// avoids re-copying identical prompts and budget-truncated long histories.
	for i := len(c.entries) - 1; i >= 0; i-- {
		e := c.entries[i]
		if kvPrefixSnapshotCompatible(state.cache, e.cache, n) && sharedTokenPrefix(e.tokens, keptTokens) == n {
			c.touch(i)
			return
		}
	}
	// A newer longer snapshot subsumes an older branch only when all of the
	// older tokens match. Divergent branches retain their own state.
	for i := len(c.entries) - 1; i >= 0; i-- {
		e := c.entries[i]
		if len(e.tokens) <= n && kvPrefixSnapshotCompatible(state.cache, e.cache, len(e.tokens)) && sharedTokenPrefix(e.tokens, keptTokens) == len(e.tokens) {
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
	c.entries = append(c.entries, conversationSnapshot{tokens: append([]uint32(nil), keptTokens...), cache: snapshot, bytes: bytes})
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
	// Prefer the most recent source when several snapshots cover the same
	// prefix, and stop once no snapshot can improve the match further.
	for i := len(c.entries) - 1; i >= 0; i-- {
		e := c.entries[i]
		n := min(sharedTokenPrefix(tokens, e.tokens), len(tokens)-1)
		if n >= conversationCacheMinTokens && n > best && kvPrefixSnapshotCompatible(cache, e.cache, n) {
			best, index = n, i
			if best == len(tokens)-1 {
				break
			}
		}
	}
	// Pin the candidate before retaining the displaced workspace: even if
	// the byte limit evicts its LRU entry, this immutable source remains valid.
	var entry conversationSnapshot
	if index >= 0 {
		entry = c.entries[index]
		c.touch(index)
	}
	// Only skip rows whose token IDs are verified in this exact workspace.
	// A caller's match against another cache must not leave stale rows behind.
	copyStart := 0
	if index >= 0 && r.prefixCache.cache == cache {
		copyStart = min(max(0, resident), best, sharedTokenPrefix(tokens, r.prefixCache.tokens))
	}
	r.retainDisplacedConversation(tokens)
	if index < 0 || copyConversationKVRange(cache, entry.cache, copyStart, best) != best {
		return resident
	}

	return best
}

func (c *conversationCache) touch(i int) {
	entry := c.entries[i]
	copy(c.entries[i:], c.entries[i+1:])
	c.entries[len(c.entries)-1] = entry
}

func (c *conversationCache) remove(i int) {
	copy(c.entries[i:], c.entries[i+1:])
	c.entries[len(c.entries)-1] = conversationSnapshot{}
	c.entries = c.entries[:len(c.entries)-1]
}
