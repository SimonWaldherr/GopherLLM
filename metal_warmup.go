package gopherllm

import (
	"context"
	"os"
)

// warmupMetalPipelines exercises the real prefill and decode forward paths
// once with a handful of synthetic tokens, immediately after loading, when
// Metal is active. Apple's Metal framework lazily JIT-compiles each distinct
// MTLComputePipelineState the first time it's dispatched (internal/metal's
// per-kernel "static id<MTLComputePipelineState> ... = nil" pattern, created
// on demand); without this, that compilation cost -- measured here at
// roughly 1.5-2s of extra prefill latency on a 14B model, across the dozen
// or so distinct kernels a single prefill+decode pass touches (Q4_K/Q5_K/
// Q6_K/Q8_0 matvec, fused/batched SwiGLU, argmax, decode-step variants) --
// silently lands on whichever request happens to run first. That is exactly
// the problem prefaultMappedModel already solves for CPU mmap pages (see
// out_of_core.go's doc comment on PrefaultPages); this applies the same fix
// to the GPU side, paying the one-time cost during "Loaded ... in Xs"
// instead of a user's first real request. Set GOPHERLLM_NO_METAL_WARMUP=1 to
// disable (e.g. for a one-shot CLI run making a single call, where moving
// the cost earlier doesn't change total wall-clock and isn't worth the
// added code path).
//
// Best-effort and silent: any failure here must never fail model loading,
// since this exercises the same forward-dispatch code every request already
// goes through, for architectures/checkpoints this function cannot fully
// enumerate. It reuses generationWorkspace, the same scratch a real request
// would allocate, so a successful warmup also pre-pays that allocation; the
// prefix cache it disturbs is reset afterward (mirroring AutoTune's own
// calibration cleanup) so the first real request cannot mistake these
// synthetic tokens for a cached conversation prefix.
func (r *Runner) warmupMetalPipelines(useMetal bool) {
	if r == nil || !useMetal || !MetalAvailable() || r.outOfCore || r.config.VocabSize <= 0 {
		return
	}
	if os.Getenv("GOPHERLLM_NO_METAL_WARMUP") == "1" {
		return
	}
	defer func() { _ = recover() }()

	const warmupTokens = 4
	vocab := r.config.VocabSize
	tokens := make([]uint32, warmupTokens)
	for i := range tokens {
		tokens[i] = uint32((i * 7919) % vocab)
	}

	r.genLock.Lock()
	defer r.genLock.Unlock()
	cache, buf := r.generationWorkspace(warmupTokens + 1)
	var logits []float32
	if err := r.prefillBatchedAt(context.Background(), cache, buf, tokens, 0, &logits); err == nil {
		r.forwardTokenInto(cache, buf, tokens[0], warmupTokens, &logits)
	}
	r.clearPrefixCache()
	r.conversations = conversationCache{limit: r.conversations.limit}
}
