package gopherllm

import (
	"sync"
	"testing"
)

// TestVisionCacheConcurrentAccess guards the lock, not the arithmetic.
//
// PrepareChatContext is exported and takes no generation lock, yet reaches
// this cache through renderMessages -> renderMistralInstMessages ->
// encodeChatImage. Planning a context window while a generation runs is an
// ordinary thing for an application to do, so the cache must tolerate
// concurrent use. Go's runtime detects concurrent map writes and kills the
// process outright -- with no mutex this test does not merely fail, it
// crashes the whole run, which is exactly the production failure it stands in
// for. It therefore needs no -race (unavailable here without cgo).
func TestVisionCacheConcurrentAccess(t *testing.T) {
	r := &Runner{}
	var wg sync.WaitGroup
	for worker := range 8 {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := range 400 {
				key := uint64(worker*400 + i)
				r.visionCachePut(key, cacheEntry(4, 32))
				r.visionCacheGet(key)
				r.visionCacheGet(uint64(i))
			}
		}(worker)
	}
	wg.Wait()
	if len(r.visionImageCache) > visionImageCacheMaxEntries {
		t.Errorf("%d entries resident, cap is %d", len(r.visionImageCache), visionImageCacheMaxEntries)
	}
	if len(r.visionImageOrder) != len(r.visionImageCache) {
		t.Errorf("order list has %d keys but map has %d entries", len(r.visionImageOrder), len(r.visionImageCache))
	}
}

func cacheEntry(tokens, dim int) visionImageCacheEntry {
	embeds := make([][]float32, tokens)
	for i := range embeds {
		embeds[i] = make([]float32, dim)
	}
	return visionImageCacheEntry{embeds: embeds, mergedRows: 1, mergedCols: tokens}
}

// TestVisionCacheKeepsStillsAcrossTurns is the property the cache exists for:
// a conversation that keeps re-sending the same picture must encode it once,
// not once per turn. The cache used to be wiped at the top of every
// GenerateChatStreamUntil call, which made every follow-up question pay for a
// full vision-tower pass.
func TestVisionCacheKeepsStillsAcrossTurns(t *testing.T) {
	r := &Runner{}
	entry := cacheEntry(112, 3072)
	r.visionCachePut(1, entry)

	// Two more turns of the same conversation, each looking the image up again.
	for turn := range 2 {
		got, ok := r.visionCacheGet(1)
		if !ok {
			t.Fatalf("turn %d: still image was evicted; it must survive across generation calls", turn)
		}
		if len(got.embeds) != len(entry.embeds) {
			t.Fatalf("turn %d: embeds = %d, want %d", turn, len(got.embeds), len(entry.embeds))
		}
	}
}

// TestVisionCacheEvictsLiveFrames is the other half of the same property: a
// camera produces a unique image per frame, so persisting has to be bounded or
// it becomes an ever-growing pile of multi-megabyte tensors.
func TestVisionCacheEvictsLiveFrames(t *testing.T) {
	r := &Runner{}
	for frame := range 200 {
		r.visionCachePut(uint64(frame), cacheEntry(112, 3072))
		if len(r.visionImageCache) > visionImageCacheMaxEntries {
			t.Fatalf("frame %d: %d entries resident, cap is %d", frame, len(r.visionImageCache), visionImageCacheMaxEntries)
		}
		if r.visionImageFloats > visionImageCacheMaxFloats {
			t.Fatalf("frame %d: %d floats resident, cap is %d", frame, r.visionImageFloats, visionImageCacheMaxFloats)
		}
	}
	if len(r.visionImageOrder) != len(r.visionImageCache) {
		t.Errorf("order list has %d keys but map has %d entries", len(r.visionImageOrder), len(r.visionImageCache))
	}
	// The newest frame must be the one that survived.
	if _, ok := r.visionCacheGet(199); !ok {
		t.Error("most recent frame was evicted")
	}
}

// TestVisionCacheEvictsLeastRecentlyUsed pins the eviction ORDER: a still that
// keeps being asked about must outlive newer frames, otherwise the multi-turn
// case degrades back to re-encoding as soon as anything else is seen.
func TestVisionCacheEvictsLeastRecentlyUsed(t *testing.T) {
	r := &Runner{}
	for i := range visionImageCacheMaxEntries {
		r.visionCachePut(uint64(i), cacheEntry(8, 64))
	}
	// Keep referring to key 0, then push the cache past its entry cap.
	if _, ok := r.visionCacheGet(0); !ok {
		t.Fatal("key 0 missing before eviction pressure")
	}
	r.visionCachePut(999, cacheEntry(8, 64))

	if _, ok := r.visionCacheGet(0); !ok {
		t.Error("recently used entry was evicted before older ones")
	}
	if _, ok := r.visionImageCache[1]; ok {
		t.Error("least recently used entry survived eviction")
	}
}

// TestVisionCacheAccountsForReplacement guards the incremental float
// accounting: re-storing a key must not double-count, or the cache slowly
// starves itself into evicting everything.
func TestVisionCacheAccountsForReplacement(t *testing.T) {
	r := &Runner{}
	r.visionCachePut(7, cacheEntry(10, 100))
	first := r.visionImageFloats
	r.visionCachePut(7, cacheEntry(10, 100))
	if r.visionImageFloats != first {
		t.Errorf("resident floats = %d after replacing the same key, want %d", r.visionImageFloats, first)
	}
	if len(r.visionImageOrder) != 1 {
		t.Errorf("order list has %d keys after replacing one, want 1", len(r.visionImageOrder))
	}
	r.visionCacheReset()
	if r.visionImageFloats != 0 || len(r.visionImageCache) != 0 || len(r.visionImageOrder) != 0 {
		t.Error("reset left state behind")
	}
}

// TestVisionCacheKeepsOversizedEntry: one image bigger than the float cap must
// still be cached, since "one huge picture, asked about repeatedly" is exactly
// the case where re-encoding hurts most.
func TestVisionCacheKeepsOversizedEntry(t *testing.T) {
	r := &Runner{}
	r.visionCachePut(1, cacheEntry(8, 64))
	r.visionCachePut(2, cacheEntry(visionImageCacheMaxFloats/1024+1, 1024))
	if _, ok := r.visionImageCache[2]; !ok {
		t.Fatal("oversized entry was not cached")
	}
	if len(r.visionImageCache) != 1 {
		t.Errorf("%d entries resident, want the oversized one alone", len(r.visionImageCache))
	}
}
