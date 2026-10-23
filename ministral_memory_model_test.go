//go:build darwin && cgo && metal

package gopherllm

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// Opt-in checkpoint probe: report the packed weight bytes newly borrowed by
// the dense decoder instead of copying its CPU-side attention projections.
func TestMinistralConfiguredWeightMemory(t *testing.T) {
	path := os.Getenv("GOPHERLLM_MINISTRAL_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_MINISTRAL_MODEL to a local Ministral GGUF")
	}
	oldF16, oldI8 := useF16KVCache.Swap(false), useI8KVCache.Swap(false)
	oldDense := metalDenseDecodeEnabled
	metalDenseDecodeEnabled = true
	t.Cleanup(func() {
		useF16KVCache.Store(oldF16)
		useI8KVCache.Store(oldI8)
		metalDenseDecodeEnabled = oldDense
	})
	t.Setenv("GOPHERLLM_NO_METAL_WARMUP", "0")
	r, _ := specABLoad(t, path, true)
	if r.kind != loadedStandard || (r.arch != "mistral3" && r.arch != "ministral") {
		t.Fatal("requires a standard Mistral-family model")
	}
	var expected int64
	for _, layer := range r.standard.Layers {
		for _, w := range []Weight{layer.WQ, layer.WK, layer.WV, layer.WO, layer.W1, layer.W3, layer.W2} {
			if w.Metal == nil {
				rowBytes, ok := w.Type.DataSize(w.Cols)
				if !ok {
					t.Fatal("unsupported packed weight")
				}
				expected += int64(w.Rows) * int64(rowBytes)
			}
		}
	}
	record := func(phase string) {
		s := r.workspaceBuf.metalDense
		if s == nil {
			t.Fatal("dense Metal decoder is inactive")
		}
		if s.borrowedWeightBytes != expected || s.copiedWeightBytes != 0 {
			t.Fatalf("dense weights are not borrowed: phase=%s borrowed=%d copied=%d expected=%d",
				phase, s.borrowedWeightBytes, s.copiedWeightBytes, expected)
		}
		runtime.GC()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		t.Logf("phase=%s dim=%d layers=%d heads=%d kv_heads=%d borrowed_weight_bytes=%d copied_weight_bytes=%d heap_alloc_bytes=%d",
			phase, r.config.Dim, r.config.NLayers, r.config.NHeads, r.config.NKVHeads,
			s.borrowedWeightBytes, s.copiedWeightBytes, m.HeapAlloc)
	}
	record("loaded")
	tokens := r.tok.Encode("Explain how a computer executes a program in five sentences.")
	tokens = tokens[:min(16, len(tokens))]
	out, wall := specABRun(t, r, "", tokens, 4, 2*time.Minute, specABMode{Name: "serial"})
	t.Logf("prompt=%d generated=%d digest=%s prefill=%v decode=%v wall=%v",
		out.Result.Stats.PromptTokens, len(out.Tokens), specABDigest(out.Tokens),
		out.Result.Stats.PrefillTime, out.Result.Stats.DecodeTime, wall)
	record("generated")
}
