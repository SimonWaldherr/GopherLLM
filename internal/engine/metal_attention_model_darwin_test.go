//go:build darwin && cgo && metal

package engine

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// Pair complete decode steps on one mapped model, overwriting the same cache
// position in ABBA order. Loading, prefill and state comparisons are untimed.
func TestMetalConfiguredAttentionBenchmark(t *testing.T) {
	path := os.Getenv("GOPHERLLM_ATTENTION_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_ATTENTION_MODEL to compare complete dense decode steps")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	setPerformanceThreads(t, 8)
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if r.kind != loadedStandard {
		t.Fatal("requires a supported dense decoder")
	}
	const steps = 32
	depth := 160
	if value := os.Getenv("GOPHERLLM_ATTENTION_DEPTH"); value != "" {
		depth, err = strconv.Atoi(value)
		if err != nil || depth < 32 || depth > 1024 {
			t.Fatal("GOPHERLLM_ATTENTION_DEPTH must be between 32 and 1024")
		}
	}
	cache, buffer := r.generationWorkspace(depth + steps)
	tokens := make([]uint32, depth+steps)
	seed := uint32(1)
	for i := range tokens {
		seed = seed*1664525 + 1013904223
		tokens[i] = seed % uint32(r.config.VocabSize)
	}
	ForwardBatchInto(r.config, r.standard, cache, buffer, tokens[:depth], 0, false, nil)
	if buffer.metalDense == nil || buffer.metalDense.copiedWeightBytes != 0 || buffer.metalDense.borrowedWeightBytes == 0 {
		t.Fatal("mapped dense decoder required")
	}
	for _, mode := range []string{"0", "1", "1", "0"} {
		t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_ATTENTION", mode)
		if _, ok := tryMetalDenseGreedy(r.config, r.standard, cache, buffer, tokens[depth], depth, tokens[depth-31:depth+1], 1.1); !ok {
			t.Fatal("dense warmup declined:", MetalError())
		}
	}
	var timings [2][]time.Duration
	var savings []time.Duration
	var samples []map[string]any
	for pos := depth; pos < depth+steps; pos++ {
		var elapsed [2]time.Duration
		var state string
		var prediction uint32
		for order, mode := range []string{"0", "1", "1", "0"} {
			t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_ATTENTION", mode)
			begin := time.Now()
			next, ok := tryMetalDenseGreedy(r.config, r.standard, cache, buffer, tokens[pos], pos, tokens[pos-31:pos+1], 1.1)
			duration := time.Since(begin)
			if !ok {
				t.Fatal("dense decode declined:", MetalError())
			}
			values := [][]float32{buffer.X, buffer.XN}
			for layer := range cache.K {
				values = append(values, cache.K[layer][pos*cache.PerPosKDim:(pos+1)*cache.PerPosKDim], cache.V[layer][pos*cache.PerPosVDim:(pos+1)*cache.PerPosVDim])
			}
			digest := ministralDecodeFloatDigest(t, values...)
			if order == 0 {
				state, prediction = digest, next
			} else if digest != state || next != prediction {
				t.Fatal("paired decode changed its prediction or live state")
			}
			elapsed[int(mode[0]-'0')] += duration / 2
		}
		for i := range timings {
			timings[i] = append(timings[i], elapsed[i])
		}
		savings = append(savings, elapsed[0]-elapsed[1])
		samples = append(samples, map[string]any{"position": pos, "separate_ns": elapsed[0].Nanoseconds(), "fused_ns": elapsed[1].Nanoseconds()})
	}
	result := map[string]any{
		"model": filepath.Base(path), "samples": samples, "bitwise_equal": true,
		"prefill_tokens":     depth,
		"separate_median_ns": specABDuration(timings[0]).Nanoseconds(), "fused_median_ns": specABDuration(timings[1]).Nanoseconds(),
		"paired_saving_median_ns": specABDuration(savings).Nanoseconds(),
		"borrowed_weight_bytes":   buffer.metalDense.borrowedWeightBytes, "copied_weight_bytes": buffer.metalDense.copiedWeightBytes,
	}
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("METAL_ATTENTION_JSON " + string(raw))
}
