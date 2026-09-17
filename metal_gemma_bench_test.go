//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
	"time"
)

func TestMetalGemmaFixedBenchmark(t *testing.T) {
	path := os.Getenv("GOPHERLLM_GEMMA_MODEL")
	if path == "" {
		t.Skip("explicit checkpoint required")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	SetNumThreads(8)
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	c := r.config
	t.Logf("arch=%s dim=%d hidden=%d layers=%d heads=%d kv=%d head_dim=%d kind=%d", c.Arch, c.Dim, c.HiddenDim, c.NLayers, c.NHeads, c.NKVHeads, c.HeadDim, r.kind)
	if r.kind == loadedGemma4 && r.gemma4.Native {
		for i, l := range r.gemma4.Layers {
			if i > 1 {
				break
			}
			t.Logf("layer %d gate=%s %dx%d metal=%v down=%s metal=%v head=%d PLE=%v", i, l.FFNGate.Type, l.FFNGate.Rows, l.FFNGate.Cols, l.FFNGate.Metal != nil, l.FFNDown.Type, l.FFNDown.Metal != nil, l.HeadDim, r.gemma4.PerLayer != nil)
		}
	} else {
		std := r.standard
		if r.kind == loadedGemma4 {
			std = r.gemma4.Standard
		}
		l := std.Layers[0]
		t.Logf("gate=%s %dx%d metal=%v down=%s metal=%v", l.W1.Type, l.W1.Rows, l.W1.Cols, l.W1.Metal != nil, l.W2.Type, l.W2.Metal != nil)
	}
	if os.Getenv("GOPHERLLM_GEMMA_INSPECT") == "1" {
		return
	}
	kd, vd, hd, kv, val := r.cacheDims()
	cache := NewKVCache(r.kvCacheLayerCount(), kd, vd, 256)
	buf := NewDecodeBuffer(c, hd, kv, val)
	var logits []float32
	tokens := make([]uint32, 224)
	state := uint32(1)
	for i := range tokens {
		state = state*1664525 + 1013904223
		tokens[i] = state % uint32(c.VocabSize)
	}
	r.forwardTokenInto(cache, buf, tokens[0], 0, &logits)
	start := time.Now()
	if r.canBatchNativeGemma4() || r.canBatchPrefill() {
		if err := r.prefillBatchedAt(context.Background(), cache, buf, tokens[:160], 0, &logits); err != nil {
			t.Fatal(err)
		}
	} else {
		for i := 0; i < 159; i++ {
			r.forwardPrefillToken(cache, buf, tokens[i], i)
		}
		r.forwardTokenInto(cache, buf, tokens[159], 159, &logits)
	}
	prefill := time.Since(start).Seconds()
	start = time.Now()
	for i := 160; i < 224; i++ {
		r.forwardTokenInto(cache, buf, tokens[i], i, &logits)
	}
	elapsed := time.Since(start).Seconds()
	for _, v := range logits {
		if !finite32(v) {
			t.Fatal("nonfinite output")
		}
	}
	raw, _ := json.Marshal(map[string]any{"model": path, "arch": c.Arch, "depth": 160, "steps": 64, "threads": 8, "kv": "f32", "seed": 1, "prefill_seconds": prefill, "decode_seconds": elapsed, "tokens_per_second": 64 / elapsed, "first_logit": logits[0]})
	fmt.Println("GEMMA_BENCH_JSON " + string(raw))
}
