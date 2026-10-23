//go:build darwin && cgo && metal

package gopherllm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"testing"
	"time"
)

// This fixed-input comparison measures decode separately from loading and
// prefill, using the Runner's mapped-weight workspace in both output modes.
// Set GOPHERLLM_MINISTRAL_MODEL to opt in. DECODE_CONTEXT=160, DECODE_STEPS=64
// and DECODE_REPS=3 use the same GOPHERLLM_MINISTRAL_ prefix.
// DECODE_SWEEP=1 interleaves greedy row groups 4/2/8/8/2/4 in one loaded model.
// AUTOTUNE=1 instead exercises complete quick calibration, including the native
// row-group sweep, without writing the user's tuning cache.
func TestMinistralConfiguredDecode(t *testing.T) {
	path := os.Getenv("GOPHERLLM_MINISTRAL_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_MINISTRAL_MODEL to a local Ministral GGUF")
	}
	parameter := func(name string, fallback, limit int) int {
		t.Helper()
		key := "GOPHERLLM_MINISTRAL_DECODE_" + name
		if value := os.Getenv(key); value != "" {
			n, err := strconv.Atoi(value)
			if err != nil || n < 1 || n > limit {
				t.Fatalf("%s must be within 1..%d", key, limit)
			}
			return n
		}
		return fallback
	}
	depth, steps, reps := parameter("CONTEXT", 160, 16383), parameter("STEPS", 64, 16383), parameter("REPS", 3, 25)
	if depth+steps > 16384 {
		t.Fatal("prompt and decode must fit the dense decoder's 16384-position limit")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	setPerformanceThreads(t, 8)
	oldF16, oldI8 := useF16KVCache.Swap(false), useI8KVCache.Swap(false)
	t.Cleanup(func() {
		useF16KVCache.Store(oldF16)
		useI8KVCache.Store(oldI8)
	})
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	c := r.config
	if r.kind != loadedStandard || (r.arch != "mistral3" && r.arch != "ministral") || !r.IsMapped() {
		t.Fatal("requires a mapped standard Ministral model")
	}
	if depth+steps > c.MaxSeqLen {
		t.Fatal("prompt and decode exceed the model context")
	}
	if os.Getenv("GOPHERLLM_MINISTRAL_AUTOTUNE") == "1" {
		before := CaptureRuntimeTuning()
		defer before.Apply()
		opts := AutoTuneOptionsForEffort("quick")
		opts.LogWriter = io.Discard
		result, err := r.AutoTune(opts)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, trial := range result.Trials {
			if trial.Knob == "metal-dense-rows" {
				found = true
			}
		}
		if !found {
			t.Fatal("eligible Ministral decoder did not measure its Metal row groups")
		}
		raw, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println("MINISTRAL_AUTOTUNE_JSON " + string(raw))
		return
	}
	tokens := make([]uint32, depth+steps)
	seed := uint32(1)
	for i := range tokens {
		seed = seed*1664525 + 1013904223
		tokens[i] = seed % uint32(c.VocabSize)
	}
	cache, buf := r.generationWorkspace(len(tokens))
	stateDigest := func(t *testing.T) string {
		groups := [][]float32{buf.X, buf.XN}
		for layer := range cache.K {
			groups = append(groups, cache.K[layer][:len(tokens)*cache.PerPosKDim], cache.V[layer][:len(tokens)*cache.PerPosVDim])
		}
		return ministralDecodeFloatDigest(t, groups...)
	}
	type outputMode struct {
		name, rows string
		greedy     bool
	}
	modes := []outputMode{{name: "logits"}, {name: "greedy", greedy: true}}
	if os.Getenv("GOPHERLLM_MINISTRAL_DECODE_SWEEP") == "1" {
		modes = nil
		for i, rows := range []string{"4", "2", "8", "8", "2", "4"} {
			modes = append(modes, outputMode{name: fmt.Sprintf("greedy_rows%s_%d", rows, i), rows: rows, greedy: true})
		}
	}
	var referenceState, referenceLogits, referencePredictions string
	for _, mode := range modes {
		t.Run(mode.name, func(t *testing.T) {
			if mode.rows != "" {
				t.Setenv("GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP", mode.rows)
			}
			var logits []float32
			predictions := make([]uint32, steps)
			step := func(pos int) {
				if mode.greedy {
					recent := tokens[max(0, pos+1-repeatPenaltyWindow) : pos+1]
					next, ok := tryMetalDenseGreedy(c, r.standard, cache, buf, tokens[pos], pos, recent, 1.1)
					if !ok {
						t.Fatal("dense greedy decode inactive:", MetalError())
					}
					predictions[pos-depth] = next
				} else {
					if !tryMetalDenseDecodeOutput(c, r.standard, cache, buf, tokens[pos], pos, &logits) {
						t.Fatal("dense logits decode inactive:", MetalError())
					}
					finishLogits(c, r.standard, &logits)
				}
			}
			prefill := func() {
				ForwardBatchInto(c, r.standard, cache, buf, tokens[:depth], 0, true, &logits)
				if buf.metalDense == nil || buf.metalDense.borrowedWeightBytes == 0 || buf.metalDense.copiedWeightBytes != 0 {
					t.Fatal("decode workspace lost its mapped, no-copy weight buffers")
				}
			}
			prefill()
			step(depth) // Warm the mode's output path before recording samples.
			decodeTimes, prefillTimes := make([]time.Duration, reps), make([]time.Duration, reps)
			samples := make([]map[string]any, 0, reps)
			var predictedDigest string
			for repeat := range reps {
				began := time.Now()
				prefill()
				prefillTimes[repeat] = time.Since(began)
				began = time.Now()
				for pos := depth; pos < len(tokens); pos++ {
					step(pos)
				}
				decodeTimes[repeat] = time.Since(began)
				residentDigest := stateDigest(t)
				if mode.greedy {
					// Re-evaluate the final fixed input outside the timed region to
					// compare the same dense projection's full vocabulary bits.
					ForwardInto(c, r.standard, cache, buf, tokens[len(tokens)-1], len(tokens)-1, &logits)
					if stateDigest(t) != residentDigest {
						t.Fatal("materializing logits changed the greedy residual or KV state")
					}
					penalized := append([]float32(nil), logits...)
					applyRepeatPenalty(penalized, tokens[max(0, len(tokens)-repeatPenaltyWindow):], 1.1)
					if predictions[steps-1] != argmaxFiniteToken(penalized) {
						t.Fatal("greedy token differs from the materialized penalized logits")
					}
					digest := specABDigest(predictions)
					if referencePredictions != "" && digest != referencePredictions {
						t.Fatal("greedy predictions changed across identical inputs")
					}
					predictedDigest, referencePredictions = digest, digest
				}
				logitsDigest := ministralDecodeFloatDigest(t, logits)
				if referenceState == "" {
					referenceState, referenceLogits = residentDigest, logitsDigest
				} else if residentDigest != referenceState || logitsDigest != referenceLogits {
					t.Fatal("hidden state, KV or logits differ across identical inputs/output modes")
				}
				samples = append(samples, map[string]any{
					"repetition": repeat, "seconds": decodeTimes[repeat].Seconds(), "prefill_seconds": prefillTimes[repeat].Seconds(),
					"state_sha256": residentDigest, "logits_sha256": logitsDigest, "predictions_digest": predictedDigest,
				})
			}
			state := buf.metalDense
			median := specABDuration(decodeTimes).Seconds()
			report := map[string]any{
				"mode": mode.name, "model": path, "threads": 8, "kv": "f32", "seed": 1, "depth": depth, "steps": steps,
				"input_digest": specABDigest(tokens), "median_seconds": median, "tokens_per_second": float64(steps) / median,
				"median_prefill_seconds": specABDuration(prefillTimes).Seconds(), "samples": samples,
				"borrowed_weight_bytes": state.borrowedWeightBytes, "copied_weight_bytes": state.copiedWeightBytes,
			}
			if mode.greedy {
				report["repeat_penalty"] = 1.1
			}
			if mode.rows != "" {
				report["rows_per_group"] = mode.rows
			}
			raw, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println("MINISTRAL_DECODE_JSON " + string(raw))
		})
	}
}

func ministralDecodeFloatDigest(t testing.TB, groups ...[]float32) string {
	t.Helper()
	h := sha256.New()
	var block [4096]byte
	for _, values := range groups {
		for len(values) > 0 {
			n := min(len(values), len(block)/4)
			for i, value := range values[:n] {
				if !finite32(value) {
					t.Fatal("non-finite decode state or logits")
				}
				binary.LittleEndian.PutUint32(block[i*4:], math.Float32bits(value))
			}
			_, _ = h.Write(block[:n*4])
			values = values[n:]
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
