package engine

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// Opt-in retained-memory probe. Use the same file in both builds; output
// digests check the workload, while GC removes transient heap allocations.
func TestQwen35ConfiguredMemory(t *testing.T) {
	path := os.Getenv("GOPHERLLM_QWEN_PERF_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_QWEN_PERF_MODEL to a local Qwen hybrid GGUF")
	}
	r, _ := specABLoad(t, path, os.Getenv("GOPHERLLM_QWEN_PERF_METAL") == "1")
	if r.kind != loadedQwen35 {
		t.Fatal("requires Qwen hybrid weights")
	}
	record := func(phase string) {
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		states := make(map[*Qwen35Cache]bool)
		if r.workspaceCache != nil && r.workspaceCache.Qwen35 != nil {
			states[r.workspaceCache.Qwen35] = true
		}
		if r.prefixCache.qwen35 != nil {
			states[r.prefixCache.qwen35] = true
		}
		var recurrentBytes int64
		for state := range states {
			recurrentBytes += state.bytes()
		}
		t.Logf("memory phase=%s heap_alloc_bytes=%d heap_inuse_bytes=%d recurrent_states=%d recurrent_bytes=%d",
			phase, m.HeapAlloc, m.HeapInuse, len(states), recurrentBytes)
		runtime.KeepAlive(r)
	}
	record("loaded")
	tokens := r.tok.Encode("Write a Go function that adds two integers.\nfunc add(a, b int) int {")
	tokens = tokens[:min(16, len(tokens))]
	out, wall := specABRun(t, r, "", tokens, 4, 2*time.Minute, specABMode{Name: "serial"})
	if len(out.Tokens) != 4 {
		t.Fatalf("short memory workload: got %d output tokens, want 4", len(out.Tokens))
	}
	t.Logf("prompt=%d generated=%d digest=%s prefill=%v decode=%v wall=%v",
		out.Result.Stats.PromptTokens, len(out.Tokens), specABDigest(out.Tokens),
		out.Result.Stats.PrefillTime, out.Result.Stats.DecodeTime, wall)
	record("generated")
}
