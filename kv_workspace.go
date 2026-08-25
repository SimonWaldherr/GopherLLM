package gopherllm

// prefixCacheState points at the retained generation workspace. tokens are
// exactly the positions resident in it. promptLogits is a bounded,
// vocab-sized snapshot taken before sampling mutates the live logits; it lets
// an identical request skip even the final prompt-token forward pass. Qwen35
// additionally needs its independent DeltaNet state to resume the prefix.
// Runner.genLock protects the whole structure.
type prefixCacheState struct {
	cache        *KVCache
	tokens       []uint32
	promptTokens int
	promptLogits []float32
	qwen35       *Qwen35Cache
	// qwen35MTP is only present for a request that opted into MTP. Its KV rows
	// remain in the shared workspace alongside ordinary Qwen K/V; this tiny
	// snapshot carries the one recurrent boundary value that those rows need.
	qwen35MTP []float32
}

func (r *Runner) cacheDims() (int, int, int, int, int) {
	if r.kind == loadedMamba2 {
		// Pure Mamba2 has no attention graph. Keep an empty KVCache shell so
		// generation's common workspace lifecycle can own its recurrent state.
		return 0, 0, 0, 0, 0
	}
	if r.kind == loadedNemotronH {
		// Soofi uses a shared attention shape on its six attention blocks;
		// Mamba/MoE blocks have no K/V entries but retain their layer index.
		return r.config.NKVHeads * r.config.HeadDim, r.config.KVDim, r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
	}
	if r.kind == loadedGemma4 {
		if r.gemma4.Native {
			maxKDim, maxVDim, maxHD, maxKV, maxVal := 0, 0, 0, 0, 0
			for _, l := range r.gemma4.Layers {
				maxKDim = max(maxKDim, l.NKVHeads*l.HeadDim)
				maxVDim = max(maxVDim, l.NKVHeads*l.ValueDim)
				maxHD = max(maxHD, l.HeadDim)
				maxKV = max(maxKV, l.NKVHeads)
				maxVal = max(maxVal, l.ValueDim)
			}
			return maxKDim, maxVDim, maxHD, maxKV, maxVal
		}
		maxHD, maxKV, maxVal := r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
		for _, l := range r.gemma4.Layers {
			maxHD = max(maxHD, l.HeadDim)
			maxKV = max(maxKV, l.NKVHeads)
			maxVal = max(maxVal, l.ValueDim)
		}
		return maxKV * maxHD, maxKV * maxVal, maxHD, maxKV, maxVal
	}
	return r.config.NKVHeads * r.config.HeadDim, r.config.KVDim, r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim
}

// kvCacheLayerCount is normally the model's decoder depth. Qwen3.5/3.6/3.8 is a
// hybrid graph, though: only its periodic full-attention layers ever access
// K/V rows. DeltaNet layers keep their separate recurrent state, so reserving
// K/V for every decoder layer wastes three quarters of the cache for the
// standard every-fourth-layer schedule.
func (r *Runner) kvCacheLayerCount() int {
	if r.kind == loadedQwen35 {
		return qwen35AttentionLayerCount(r.qwen35)
	}
	if r.kind == loadedGemma4 && r.gemma4.Native {
		return nativeGemma4KVCacheLayerCount(r.gemma4)
	}
	return r.config.NLayers
}

const maxReusableKVCacheBytes int64 = 512 << 20

func kvCacheBytes(layers, kDim, vDim, cacheLen int, format kvFormat) int64 {
	if format == kvI8 {
		// Q8_0 is 34 bytes per 32-element block (1.0625 B/elem), not a flat
		// per-element multiplier like f32/f16 — computing bytesPerRow first
		// keeps the block rounding exact instead of approximating it.
		bytesPerRow := int64(q8RowBytes(kDim) + q8RowBytes(vDim))
		return int64(layers) * bytesPerRow * int64(cacheLen)
	}
	elemBytes := int64(4)
	if format == kvF16 {
		elemBytes = 2
	}
	return int64(layers) * int64(kDim+vDim) * int64(cacheLen) * elemBytes
}

func kvCacheStorageBytes(cache *KVCache) int64 {
	if cache == nil {
		return 0
	}
	return kvCacheBytes(cache.layerCount(), cache.PerPosKDim, cache.PerPosVDim, cache.MaxLen, cache.kvFormat())
}

func grownKVCacheLen(current, required, limit int, config Config) int {
	if current <= 0 {
		return required
	}
	grown := current * 2
	if grown < current { // integer overflow: the required size is safer.
		grown = required
	}
	grown = max(grown, current+prefillChunkSize(config))
	target := max(required, grown)
	if limit > 0 {
		target = min(target, limit)
	}
	return target
}

// copyKVPrefix transfers complete token rows between shape-compatible caches.
// It intentionally excludes Nemotron-H's recurrent state; prefix reuse for
// that hybrid architecture stays disabled until its state can be copied too.
func copyKVPrefix(dst, src *KVCache, positions int) int {
	if dst == nil || src == nil || positions <= 0 || dst.kvFormat() != src.kvFormat() ||
		dst.PerPosKDim != src.PerPosKDim || dst.PerPosVDim != src.PerPosVDim ||
		dst.layerCount() != src.layerCount() {
		return 0
	}
	positions = min(positions, min(dst.MaxLen, src.MaxLen))
	if positions <= 0 {
		return 0
	}
	if dst.kvFormat() == kvI8 {
		kLen8 := positions * q8RowBytes(dst.PerPosKDim)
		vLen8 := positions * q8RowBytes(dst.PerPosVDim)
		for layer := 0; layer < dst.layerCount(); layer++ {
			copy(dst.K8[layer][:kLen8], src.K8[layer][:kLen8])
			copy(dst.V8[layer][:vLen8], src.V8[layer][:vLen8])
		}
		return positions
	}
	kLen := positions * dst.PerPosKDim
	vLen := positions * dst.PerPosVDim
	for layer := 0; layer < dst.layerCount(); layer++ {
		if dst.F16 {
			copy(dst.K16[layer][:kLen], src.K16[layer][:kLen])
			copy(dst.V16[layer][:vLen], src.V16[layer][:vLen])
			continue
		}
		copy(dst.K[layer][:kLen], src.K[layer][:kLen])
		copy(dst.V[layer][:vLen], src.V[layer][:vLen])
	}
	return positions
}

func sharedTokenPrefix(a, b []uint32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func (r *Runner) clearPrefixCache() {
	r.prefixCache = prefixCacheState{}
}

func (r *Runner) prefixCacheSupported(cache *KVCache) bool {
	return r.prefixCacheSupportedWithMTP(cache, false)
}

func (r *Runner) prefixCacheSupportedWithMTP(cache *KVCache, useMTP bool) bool {
	if cache == nil || cache != r.workspaceCache {
		return false
	}
	if r.kind == loadedQwen35 {
		// Qwen3.8's DeltaNet cache is needed in addition to K/V rows. Bound the
		// combined retained footprint to the same budget that guards normal
		// prefix caching; a 27B Qwen3.8 snapshot is about 150 MiB and fits with
		// the common short-context f16 or i8 K/V cache.
		if cache.Qwen35 == nil {
			return false
		}
		bytes := kvCacheStorageBytes(cache) + cache.Qwen35.bytes()
		if useMTP {
			if cache.Qwen35MTP == nil || cache.Qwen35MTP.KV == nil {
				return false
			}
			bytes += kvCacheStorageBytes(cache.Qwen35MTP.KV)
		}
		return bytes <= maxReusableKVCacheBytes
	}
	// Nemotron-H's recurrent state has no prefix snapshot yet. Reusing only
	// its attention K/V rows (or none at all for pure Mamba2) would be wrong.
	return r.kind != loadedNemotronH && r.kind != loadedMamba2
}

// prefixReuse returns the exact number of resident KV positions that match.
// GenerateChatStreamUntil may skip an identical prompt completely when the
// corresponding immutable pre-sampling logits snapshot is also resident.
func (r *Runner) prefixReuse(cache *KVCache, tokens []uint32) int {
	return r.prefixReuseWithMTP(cache, tokens, false)
}

func (r *Runner) prefixReuseWithMTP(cache *KVCache, tokens []uint32, useMTP bool) int {
	state := r.prefixCache
	if state.cache != cache || len(state.tokens) == 0 || cache == nil {
		return 0
	}
	matched := min(sharedTokenPrefix(tokens, state.tokens), cache.MaxLen)
	if matched == 0 {
		return 0
	}
	if r.kind == loadedQwen35 {
		// A single snapshot represents the state after the complete resident
		// token sequence, unlike ordinary K/V rows reusable at arbitrary prefixes.
		// It is therefore valid only when the complete snapshot prefix matches.
		// If the new prompt ends exactly there without matching prompt logits,
		// the generic one-token re-forward fallback would update DeltaNet state
		// twice, so leave it cold instead.
		if matched != len(state.tokens) || (matched == len(tokens) && state.promptTokens != len(tokens)) {
			return 0
		}
		// Preflight the independent MTP boundary before restoring the target
		// recurrence. A missing MTP snapshot is normal when the immediately
		// preceding request ran without MTP; returning cold after mutating only
		// Qwen35 would otherwise make its next prefill start from an unrelated
		// recurrent state.
		if useMTP && (cache.Qwen35MTP == nil || len(state.qwen35MTP) != len(cache.Qwen35MTP.PendingHidden)) {
			return 0
		}
		if !cache.Qwen35.restore(state.qwen35) {
			return 0
		}
		if useMTP && !cache.Qwen35MTP.restorePending(state.qwen35MTP) {
			return 0
		}
	}
	return matched
}

// generationWorkspace reuses request scratch behind genLock. The retained
// cache also serves as a single bounded, token-verified prefix cache for the
// most recent conversation. When it grows, copy known K/V rows geometrically
// rather than re-prefilling prior turns. Retaining at most 512 MiB avoids
// turning one unusually large context into a permanent memory commitment.
func (r *Runner) generationWorkspace(cacheLen int, enableMTP ...bool) (*KVCache, *DecodeBuffer) {
	useMTP := len(enableMTP) > 0 && enableMTP[0] && r.kind == loadedQwen35 && r.qwen35.MTP != nil
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	layers := r.kvCacheLayerCount()
	old := r.workspaceCache
	// allowI8/desiredFormat are computed once here and reused for both the
	// reuse-compatibility check below and the allocation call further down —
	// two independently-written derivations of "what format should this be"
	// could silently disagree on a stale-cache-reuse decision without either
	// one erroring, which is exactly the failure mode worth avoiding.
	allowI8 := kvI8Eligible(kDim, vDim, maxHead, maxVal)
	desiredFormat := desiredKVFormat(allowI8)
	mtpKVBytes := func(length int, format kvFormat) int64 {
		if !useMTP {
			return 0
		}
		return kvCacheBytes(1, kDim, vDim, length, format)
	}
	shapeCompatible := old != nil && old.layerCount() == layers && old.kvFormat() == desiredFormat &&
		old.PerPosKDim == kDim && old.PerPosVDim == vDim
	cache := old
	compatible := shapeCompatible && old.MaxLen >= cacheLen
	if !compatible {
		targetLen := cacheLen
		if shapeCompatible {
			targetLen = grownKVCacheLen(old.MaxLen, cacheLen, r.config.MaxSeqLen, r.config)
		}
		// Geometric headroom is useful only while it stays within the same
		// retention budget. Never make a one-off request allocate a larger
		// temporary cache merely because the next growth step crossed 512 MiB.
		if targetLen > cacheLen && kvCacheBytes(layers, kDim, vDim, targetLen, desiredFormat)+mtpKVBytes(targetLen, desiredFormat) > maxReusableKVCacheBytes {
			targetLen = cacheLen
		}
		cache = newKVCacheAuto(layers, kDim, vDim, targetLen, allowI8)
		bytes := kvCacheBytes(layers, kDim, vDim, targetLen, cache.kvFormat()) + mtpKVBytes(targetLen, cache.kvFormat())
		if bytes <= maxReusableKVCacheBytes {
			r.workspaceCache = cache
			if shapeCompatible && r.prefixCache.cache == old && r.kind != loadedNemotronH && r.kind != loadedMamba2 {
				if copied := copyKVPrefix(cache, old, len(r.prefixCache.tokens)); copied == len(r.prefixCache.tokens) {
					// Qwen's recurrent snapshot is independent from its K/V rows,
					// so a geometrically grown cache can retain both. Keep their
					// combined footprint within the normal prefix-cache budget.
					if r.kind != loadedQwen35 || (r.prefixCache.qwen35 != nil &&
						kvCacheStorageBytes(cache)+r.prefixCache.qwen35.bytes() <= maxReusableKVCacheBytes) {
						r.prefixCache.cache = cache
					} else {
						r.clearPrefixCache()
					}
				} else {
					r.clearPrefixCache()
				}
			} else if r.prefixCache.cache == old {
				r.clearPrefixCache()
			}
		} else if !shapeCompatible && r.prefixCache.cache == old {
			r.clearPrefixCache()
		}
	}
	// A cache growth during a non-MTP request copies only the target's K/V
	// rows. Preserve that useful target prefix, but invalidate the independent
	// MTP boundary snapshot: its one-layer K/V rows were intentionally not
	// allocated/copied for this request and restoring just PendingHidden later
	// would make an apparently warm MTP prefix incorrect.
	if !useMTP && cache != old && r.kind == loadedQwen35 && r.prefixCache.cache == cache {
		r.prefixCache.qwen35MTP = nil
	}
	if r.kind == loadedNemotronH || r.kind == loadedMamba2 {
		if !cache.Nemotron.compatible(r.config) {
			cache.Nemotron = newNemotronHCache(r.config)
		}
		// Unlike attention K/V, recurrent state is read before a position can
		// overwrite it, so a reused workspace must start every request at zero.
		cache.Nemotron.reset()
	}
	if r.kind == loadedQwen35 {
		recurrentLayers := qwen35RecurrentLayerCount(r.qwen35)
		if !cache.Qwen35.compatible(r.config, recurrentLayers) {
			cache.Qwen35 = newQwen35Cache(r.config, recurrentLayers)
		}
		cache.Qwen35.reset()
		if useMTP {
			oldMTP := (*Qwen35MTPState)(nil)
			if old != nil {
				oldMTP = old.Qwen35MTP
			}
			if !cache.Qwen35MTP.compatible(r.config, cache.MaxLen, cache.kvFormat()) {
				cache.Qwen35MTP = newQwen35MTPState(r.config, cache.MaxLen, cache.kvFormat())
				// A geometrically grown target cache can retain a matching MTP
				// prefix too. The base prefix remains useful even if this copy
				// cannot happen, so leave its metadata intact; the MTP-specific
				// restore will simply cold-prefill on the next request.
				if oldMTP != nil && r.prefixCache.cache == cache && len(r.prefixCache.tokens) > 0 {
					if copyQwen35MTPPrefix(cache.Qwen35MTP, oldMTP, len(r.prefixCache.tokens)) != len(r.prefixCache.tokens) {
						r.prefixCache.qwen35MTP = nil
					}
				}
			}
			cache.Qwen35MTP.reset()
		}
	}
	if r.workspaceBuf == nil {
		r.workspaceBuf = NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	}
	return cache, r.workspaceBuf
}
