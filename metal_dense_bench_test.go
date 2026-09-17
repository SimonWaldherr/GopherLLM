//go:build darwin && cgo && metal

package gopherllm

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
	"time"
)

// The paired llama harness uses the same LCG token IDs, context depth, KV
// precision and logits workload. This intentionally excludes text sampling.
func TestMetalFixedDecodeBenchmark(t *testing.T) {
	path := os.Getenv("GOPHERLLM_FIXED_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_FIXED_MODEL for an explicit benchmark")
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
	if r.kind != loadedStandard {
		t.Fatal("standard dense model required")
	}
	tokens := make([]uint32, 224)
	state := uint32(1)
	for i := range tokens {
		state = state*1664525 + 1013904223
		tokens[i] = state % uint32(c.VocabSize)
	}
	cache := NewKVCache(c.NLayers, c.NKVHeads*c.HeadDim, c.NKVHeads*c.ValueDim, 256)
	buf := NewDecodeBuffer(c, c.HeadDim, c.NKVHeads, c.ValueDim)
	defer func() {
		if buf.metalDense != nil {
			buf.metalDense.close()
		}
	}()
	var logits []float32
	ForwardInto(c, r.standard, cache, buf, tokens[0], 0, &logits)
	if metalDenseDecodeEnabled && buf.metalDense == nil {
		t.Fatalf("dense decoder was not active: %s config=%+v", MetalError(), c)
	}
	ForwardBatchInto(c, r.standard, cache, buf, tokens[:160], 0, true, &logits)
	start := time.Now()
	for i := 160; i < 224; i++ {
		ForwardInto(c, r.standard, cache, buf, tokens[i], i, &logits)
	}
	elapsed := time.Since(start).Seconds()
	if len(logits) == 0 || !finite32(logits[0]) {
		t.Fatal("invalid logits")
	}
	report := map[string]any{"engine": "GopherLLM", "dense": metalDenseDecodeEnabled, "depth": 160, "steps": 64, "threads": 8, "kv": "f32", "seed": 1, "vocab": c.VocabSize, "seconds": elapsed, "tokens_per_second": 64 / elapsed, "first_logit": logits[0]}
	raw, _ := json.Marshal(report)
	fmt.Println("FIXED_DECODE_JSON " + string(raw))
}
