//go:build darwin && cgo && coreml

package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// This optional check measures complete CPU-prefill with and without one
// selected native FFN. Unlike the layer harness it loads a Runner. Keep it
// separate so a layer comparison does not map all weights into resident RAM.
func TestCoreMLConfiguredPrefill(t *testing.T) {
	path, manifestPath := os.Getenv("GOPHERLLM_COREML_PREFILL_GGUF"), os.Getenv("GOPHERLLM_COREML_PREFILL_MANIFEST")
	if path == "" || manifestPath == "" {
		t.Skip("set GOPHERLLM_COREML_PREFILL_GGUF and GOPHERLLM_COREML_PREFILL_MANIFEST for full CPU-prefill comparison")
	}
	manifest, _, err := readCoreMLFFNManifest(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	// Use the same float-activation source reference as the layer comparison.
	// Comparing with Q8 CPU activations also measures a different approximation
	// in the original FFN, whose downstream effects need separate evaluation.
	previousQ8 := useQ8Activations.Swap(false)
	defer useQ8Activations.Store(previousQ8)
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if r.kind != loadedStandard && r.kind != loadedQwen35 {
		t.Fatal("requires a dense standard or hybrid Qwen Runner")
	}
	count := manifest.Tokens
	tokens := r.tok.Encode(strings.Repeat("Erkläre, wie ein Computer Text verarbeitet. Explain how a computer processes text. ", count))
	if len(tokens) < count {
		t.Fatal("fixture text did not cover the fixed token block")
	}
	tokens = tokens[:count]
	cache, buf := r.generationWorkspace(count)
	prefill := func() ([]float32, time.Duration) {
		t.Helper()
		if cache.Qwen35 != nil {
			cache.Qwen35.reset()
		}
		var logits []float32
		start := time.Now()
		var err error
		if r.kind == loadedQwen35 {
			err = r.prefillQwen35At(context.Background(), cache, buf, tokens, 0, &logits)
		} else {
			err = r.prefillBatchedAt(context.Background(), cache, buf, tokens, 0, &logits)
		}
		elapsed := time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
		return logits, elapsed
	}
	_, _ = prefill() // allocate and warm the ordinary CPU scratch first
	want, referenceTime := prefill()
	t.Log("source float-activation prefill complete")
	if err := r.SetCoreMLFFN([]string{manifestPath}, CoreMLFFNOptions{}); err != nil {
		t.Fatal(err)
	}
	t.Log("Core ML FFN attached with source binding and quality probe")
	_, _ = prefill()
	got, nativeTime := prefill()
	var aa, bb, dot, delta float64
	for i, a := range want {
		if !finite32(got[i]) || !finite32(a) {
			t.Fatal("non-finite prefill logits")
		}
		x, y := float64(a), float64(got[i])
		aa += x * x
		bb += y * y
		dot += x * y
		delta += (x - y) * (x - y)
	}
	cosine, relative := dot/math.Sqrt(aa*bb), math.Sqrt(delta/aa)
	if math.IsNaN(cosine) || math.IsInf(cosine, 0) || cosine < .999 || relative > .02 {
		t.Fatalf("prefill logits drifted: cosine=%g relative_l2=%g", cosine, relative)
	}
	result := struct {
		Architecture string  `json:"architecture"`
		Layer        int     `json:"layer"`
		Tokens       int     `json:"tokens"`
		ReferenceMS  float64 `json:"reference_ms"`
		CoreMLMS     float64 `json:"coreml_ms"`
		Cosine       float64 `json:"logits_cosine"`
		RelativeL2   float64 `json:"logits_relative_l2"`
	}{r.arch, manifest.Layer, count, float64(referenceTime) / 1e6, float64(nativeTime) / 1e6, cosine, relative}
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("COREML_PREFILL_JSON " + string(raw))
}
