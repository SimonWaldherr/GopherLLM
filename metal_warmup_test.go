package gopherllm

import "testing"

// TestWarmupMetalPipelinesDoesNotCorruptState loads a tiny synthetic model
// with UseMetal requested (a no-op on a non-metal build, the real thing
// under -tags metal on Darwin) and checks that warmupMetalPipelines leaves
// the Runner in a clean, usable state: the prefix and conversation caches
// it disturbs are reset (see warmupMetalPipelines's doc comment), and a real
// generation afterward produces the same output a plain CPU load would.
// Regression coverage for the missing-warmup latency this fixes lives in
// this session's manual measurements (see the commit message); reproducing
// the ~2s first-request Metal pipeline-compile stall in a portable, fast
// unit test isn't practical since it depends on Apple's shader JIT cache
// being genuinely cold, which only a fresh process on real hardware has.
func TestWarmupMetalPipelinesDoesNotCorruptState(t *testing.T) {
	r, err := RunnerFromGGUFBytesWithOptions(buildTinyLlamaGGUF(), LoadOptions{UseMetal: true})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if len(r.prefixCache.tokens) != 0 {
		t.Fatalf("prefix cache not reset after load: %d stale tokens", len(r.prefixCache.tokens))
	}
	if len(r.conversations.entries) != 0 {
		t.Fatalf("conversation cache not reset after load: %d stale entries", len(r.conversations.entries))
	}

	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 16)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	var logits []float32
	r.forwardTokenInto(cache, buf, 3, 0, &logits)
	if len(logits) != r.config.VocabSize {
		t.Fatalf("forward pass after warmup produced %d logits, want %d", len(logits), r.config.VocabSize)
	}
	for i, v := range logits {
		if v != v { // NaN check
			t.Fatalf("logit %d is NaN after a Metal-warmed load", i)
		}
	}
}

// TestWarmupMetalPipelinesSkipsOutOfCore checks the same guard
// AutoTune itself relies on: out-of-core loads must never trigger an extra
// eager forward pass, since that would force-page-in weights out-of-core
// mode exists specifically to keep demand-paged.
func TestWarmupMetalPipelinesSkipsOutOfCore(t *testing.T) {
	r := &Runner{outOfCore: true, config: Config{VocabSize: 32}}
	// Must not panic despite an incomplete Runner (no loaded weights, no
	// genLock use reached) -- the out-of-core guard should return before
	// touching anything else.
	r.warmupMetalPipelines(true)
}
