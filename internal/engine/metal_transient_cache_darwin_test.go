//go:build darwin && cgo && metal

package engine

import "testing"

func TestMetalTransientCacheReleasePreservesRetainedWorkspace(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	c, _, w := metalResidualNormFixture(t, "ministral", 256)
	cache, buf := metalPrefillBuffer(t, c, 32)
	if !tryMetalDenseDecode(c, w, cache, buf, 3, 0) {
		t.Fatal(MetalError())
	}
	state := buf.metalDense
	releaseTransientMetalCache(buf, cache, cache)
	if buf.metalDense != state {
		t.Fatal("retained cache was released")
	}
	other := NewKVCache(c.NLayers, c.NKVHeads*c.HeadDim, c.NKVHeads*c.ValueDim, 32)
	releaseTransientMetalCache(buf, other, nil)
	if buf.metalDense != state {
		t.Fatal("unrelated cache was released")
	}
	releaseTransientMetalCache(buf, cache, other)
	if buf.metalDense != nil || state.decoder != nil || len(state.sharedK) != 0 || len(state.refs) != 0 || len(state.owned) != 0 {
		t.Fatal("temporary GPU cache remains reachable")
	}
	releaseTransientMetalCache(buf, cache, other)
	releaseTransientMetalCache(nil, cache, other)
	// The next request can still initialize a valid decoder on the retained cache.
	if !tryMetalDenseDecode(c, w, other, buf, 3, 0) {
		t.Fatal("decoder failed after transient cleanup:", MetalError())
	}
}
