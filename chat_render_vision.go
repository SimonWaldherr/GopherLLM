package gopherllm

import (
	"fmt"
	"hash/fnv"
)

// renderMessagesForGeneration is renderMessages plus image-embedding
// support. When no message carries images it is exactly renderMessages
// (nil map, nil error) — every existing caller (context_window.go's
// token-budget calculations, tests) keeps calling the plain renderMessages
// unchanged. Only the real generation entry point (GenerateChatStreamUntil)
// calls this, since it's the one place that needs the embeddings to splice
// into the forward pass. When a message does carry an image, only the
// Mistral-family template understands ChatMessage.Images, so this bypasses
// renderMessages' generic multi-template dispatch entirely rather than
// risking a silent fallback to a renderer that would just drop the image.
func (r *Runner) renderMessagesForGeneration(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]uint32, map[int][]float32, error) {
	hasImages := false
	for i := range messages {
		if len(messages[i].Images) > 0 {
			hasImages = true
			break
		}
	}
	if !hasImages {
		return r.renderMessages(messages, systemPrompt, tools), nil, nil
	}
	if r.chatTemplateKind() != "mistral-inst" {
		return nil, nil, fmt.Errorf("image content requires a Mistral-family chat template (this model uses %q)", r.chatTemplateKind())
	}
	tokens, embeds, ok, err := r.renderMistralInstMessages(messages, systemPrompt, tools)
	if err != nil {
		return nil, nil, err
	}
	if !ok {
		return nil, nil, fmt.Errorf("rendering message with image failed: this model's vocabulary is missing [INST]/[/INST]")
	}
	return tokens, embeds, nil
}

// encodeChatImage decodes, preprocesses, and runs one image through the
// vision tower, memoizing by a content hash of the raw image bytes: a
// generation call may render the same messages more than once
// (context_window.go's budget calculations call renderMessages, not this
// function, precisely to avoid that cost, but a caller could still call
// this directly more than once per request) and re-running the whole vision
// tower each time would be wasteful.
func (r *Runner) encodeChatImage(img ImageContent) (embeds [][]float32, mergedRows, mergedCols int, err error) {
	h := fnv.New64a()
	h.Write(img.Bytes)
	key := h.Sum64()
	if entry, ok := r.visionCacheGet(key); ok {
		return entry.embeds, entry.mergedRows, entry.mergedCols, nil
	}
	decoded, err := DecodeImageBytes(img.Bytes)
	if err != nil {
		return nil, 0, 0, err
	}
	vc := r.visionConfig
	// Dynamic Pixtral sizing is token-budgeted, not just longest-edge limited.
	// Matching this grid exactly keeps the number and layout of substituted
	// [IMG]/[IMG_BREAK] embeddings equal to the checkpoint's reference path.
	pre, err := PreprocessImagePixtralDynamic(decoded, vc.PatchSize, vc.SpatialMergeSize, vc.ImageMinTokens, vc.ImageMaxTokens, vc.ImageMean, vc.ImageStd)
	if err != nil {
		return nil, 0, 0, err
	}
	embeds, mergedRows, mergedCols, err = EncodeImagePixtral(vc, *r.vision, pre)
	if err != nil {
		return nil, 0, 0, err
	}
	r.visionCachePut(key, visionImageCacheEntry{embeds: embeds, mergedRows: mergedRows, mergedCols: mergedCols})
	return embeds, mergedRows, mergedCols, nil
}

// visionCacheGet returns a memoized encoding and marks it most recently used.
func (r *Runner) visionCacheGet(key uint64) (visionImageCacheEntry, bool) {
	r.visionCacheMu.Lock()
	defer r.visionCacheMu.Unlock()
	entry, ok := r.visionImageCache[key]
	if !ok {
		return visionImageCacheEntry{}, false
	}
	r.visionCacheTouch(key)
	return entry, true
}

// visionCacheTouch moves key to the most-recently-used end of the order.
// Callers hold visionCacheMu.
func (r *Runner) visionCacheTouch(key uint64) {
	for i, k := range r.visionImageOrder {
		if k != key {
			continue
		}
		r.visionImageOrder = append(append(r.visionImageOrder[:i], r.visionImageOrder[i+1:]...), key)
		return
	}
	r.visionImageOrder = append(r.visionImageOrder, key)
}

// visionCachePut stores an encoding and evicts least-recently-used entries
// until both caps hold. A single entry larger than the float cap is stored
// anyway and simply evicts everything else: refusing to cache it would make
// the pathological case (one very large image, asked about repeatedly) the one
// case that gets no benefit at all.
func (r *Runner) visionCachePut(key uint64, entry visionImageCacheEntry) {
	r.visionCacheMu.Lock()
	defer r.visionCacheMu.Unlock()
	if r.visionImageCache == nil {
		r.visionImageCache = make(map[uint64]visionImageCacheEntry)
	}
	if old, ok := r.visionImageCache[key]; ok {
		r.visionImageFloats -= visionEntryFloats(old)
	}
	r.visionImageCache[key] = entry
	r.visionImageFloats += visionEntryFloats(entry)
	r.visionCacheTouch(key)

	for len(r.visionImageOrder) > 1 &&
		(len(r.visionImageOrder) > visionImageCacheMaxEntries || r.visionImageFloats > visionImageCacheMaxFloats) {
		oldest := r.visionImageOrder[0]
		r.visionImageOrder = r.visionImageOrder[1:]
		if evicted, ok := r.visionImageCache[oldest]; ok {
			r.visionImageFloats -= visionEntryFloats(evicted)
			delete(r.visionImageCache, oldest)
		}
	}
}

func visionEntryFloats(entry visionImageCacheEntry) int {
	if len(entry.embeds) == 0 {
		return 0
	}
	return len(entry.embeds) * len(entry.embeds[0])
}

// visionCacheReset drops every memoized encoding. Used when the Runner is
// closed or its vision weights are replaced, where the entries would otherwise
// outlive the tower that produced them.
func (r *Runner) visionCacheReset() {
	r.visionCacheMu.Lock()
	defer r.visionCacheMu.Unlock()
	r.visionImageCache = nil
	r.visionImageOrder = nil
	r.visionImageFloats = 0
}
