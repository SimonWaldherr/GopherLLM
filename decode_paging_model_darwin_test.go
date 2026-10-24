//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"strconv"
	"syscall"
	"testing"
	"time"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

// Opt-in complete-model ABBA comparisons. MODEL is required; TOKENS defaults
// to 64, CYCLES to 2. Prefix variables with GOPHERLLM_DECODE_. MODE=0|1 fixes
// the backend for independent-process memory comparisons; unset alternates.
// Prefill is cold for every request; loading and warmup are logged separately.
func TestConfiguredDecodePaging(t *testing.T) {
	path := os.Getenv("GOPHERLLM_DECODE_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_DECODE_MODEL to a local GGUF")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	setPerformanceThreads(t, 8)
	oldF16, oldI8 := useF16KVCache.Swap(false), useI8KVCache.Swap(false)
	oldQ8 := useQ8Activations.Swap(q8ActivationsAvailable())
	t.Cleanup(func() { useF16KVCache.Store(oldF16); useI8KVCache.Store(oldI8); useQ8Activations.Store(oldQ8) })
	t.Setenv("GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP", "4")
	t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", "1")
	integer := func(key string, fallback int) int {
		if text := os.Getenv("GOPHERLLM_DECODE_" + key); text != "" {
			n, err := strconv.Atoi(text)
			if err != nil || n < 1 || n > 256 {
				t.Fatalf("invalid %s", key)
			}
			return n
		}
		return fallback
	}
	tokens, cycles := integer("TOKENS", 64), integer("CYCLES", 2)
	modes := []string{"0", "1", "1", "0"}
	if mode := os.Getenv("GOPHERLLM_DECODE_MODE"); mode != "" {
		if mode != "0" && mode != "1" {
			t.Fatal("invalid mode")
		}
		modes = []string{mode}
	}
	// Start fused-only memory runs without ever allocating the reference slabs.
	for _, key := range []string{"GOPHERLLM_METAL_DENSE_FUSED_ATTENTION", "GOPHERLLM_QWEN_OVERLAP_INPUTS", "GOPHERLLM_METAL_QWEN_INPLACE_SILU", "GOPHERLLM_METAL_FFN_INPLACE", "GOPHERLLM_METAL_GEMMA_PARALLEL_ARGMAX"} {
		t.Setenv(key, modes[0])
	}
	emit := func(v any) {
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println("DECODE_PAGING_JSON " + string(raw))
	}
	var beforeLoad, afterLoad syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &beforeLoad); err != nil {
		t.Fatal(err)
	}
	r, info, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultCore, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	if r.kind != loadedStandard && r.kind != loadedQwen35 && r.kind != loadedGemma4 {
		t.Fatal("requires a standard, hybrid Qwen or Gemma model")
	}
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &afterLoad); err != nil {
		t.Fatal(err)
	}
	emit(map[string]any{"phase": "load", "model": path, "arch": r.arch, "load_ns": info.LoadTime.Nanoseconds(), "file_bytes": info.FileSizeBytes, "major_faults": afterLoad.Majflt - beforeLoad.Majflt, "minor_faults": afterLoad.Minflt - beforeLoad.Minflt, "max_rss_bytes": afterLoad.Maxrss})
	if r.kind == loadedGemma4 && r.gemma4.PerLayer != nil {
		w := r.gemma4.PerLayer.TokenEmbd
		emit(map[string]any{"phase": "lookup_storage", "type": w.Type.String(), "rows": w.Rows, "cols": w.Cols,
			"raw_bytes": len(w.Raw), "f32_bytes": len(w.F32) * 4, "prepared": w.Prepared != nil, "metal": w.Metal != nil})
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens = tokens
	opts.Seed = 41
	opts.Sampler.Temperature = 0
	if text := os.Getenv("GOPHERLLM_DECODE_PENALTY"); text != "" {
		penalty, err := strconv.ParseFloat(text, 32)
		if err != nil || !finite32(float32(penalty)) || penalty <= 0 {
			t.Fatal("invalid repeat penalty")
		}
		opts.Sampler.RepeatPenalty = float32(penalty)
	}
	messages := []ChatMessage{UserMessage("Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird.")}
	var reference []uint32
	var referenceText, referenceState string
	var promptDepth int
	run := func(cycle, order int, mode string) {
		t.Helper()
		t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_ATTENTION", mode)
		t.Setenv("GOPHERLLM_QWEN_OVERLAP_INPUTS", mode)
		t.Setenv("GOPHERLLM_METAL_QWEN_INPLACE_SILU", mode)
		t.Setenv("GOPHERLLM_METAL_FFN_INPLACE", mode)
		t.Setenv("GOPHERLLM_METAL_GEMMA_PARALLEL_ARGMAX", mode)
		r.ClearConversationCache()
		_, _, startFaults, startPageins, ok := metalbackend.ProcessMemory()
		if !ok {
			t.Fatal("Mach memory counters unavailable")
		}
		var before, after syscall.Rusage
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &before); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()
		result, err := r.GenerateChat(messages, opts.WithContext(ctx))
		if err != nil {
			t.Fatal(err)
		}
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &after); err != nil {
			t.Fatal(err)
		}
		if result.Stats.GeneratedTokens != tokens {
			t.Fatalf("short request: got %d, want %d", result.Stats.GeneratedTokens, tokens)
		}
		if result.PromptCache != nil && result.PromptCache.ReusedTokens != 0 {
			t.Fatal("prompt was not cold")
		}
		promptDepth = result.Stats.PromptTokens
		b, k := r.workspaceBuf, r.workspaceCache
		ids := b.GeneratedTokens[:tokens]
		values := [][]float32{b.X, b.XN}
		depth := result.Stats.PromptTokens + tokens - 1
		for layer := range k.K {
			values = append(values, k.K[layer][:depth*k.PerPosKDim], k.V[layer][:depth*k.PerPosVDim])
		}
		if k.Qwen35 != nil {
			values = append(values, k.Qwen35.Conv, k.Qwen35.State)
		}
		state := ministralDecodeFloatDigest(t, values...)
		if reference == nil {
			reference = slices.Clone(ids)
			referenceText = result.Text
			referenceState = state
		} else if !slices.Equal(reference, ids) || referenceText != result.Text || referenceState != state {
			t.Fatal("tokens, text or complete live state differ")
		}
		var memory runtime.MemStats
		runtime.ReadMemStats(&memory)
		resident, footprint, faults, pageins, ok := metalbackend.ProcessMemory()
		if !ok {
			t.Fatal("Mach memory counters unavailable")
		}

		phase := "sample"
		if cycle < 0 {
			phase = "warmup"
		}
		row := map[string]any{"phase": phase, "model": path, "cycle": cycle, "order": order, "fused": mode == "1", "prompt_tokens": result.Stats.PromptTokens, "tokens": tokens, "prefill_ns": result.Stats.PrefillTime.Nanoseconds(), "decode_ns": result.Stats.DecodeTime.Nanoseconds(), "token_digest": specABDigest(ids), "state_sha256": state, "major_faults": after.Majflt - before.Majflt, "minor_faults": after.Minflt - before.Minflt, "rss_bytes": resident, "physical_footprint_bytes": footprint, "mach_faults": faults - startFaults, "mach_pageins": pageins - startPageins, "max_rss_bytes": after.Maxrss, "heap_alloc_bytes": memory.HeapAlloc, "metal_batch_bytes": metalbackend.BatchScratchBytes(), "kv_bytes": kvCacheStorageBytes(k)}
		row["repeat_penalty"] = opts.Sampler.RepeatPenalty
		if b.metalDense != nil {
			row["copied_weight_bytes"] = b.metalDense.copiedWeightBytes
			row["borrowed_weight_bytes"] = b.metalDense.borrowedWeightBytes
			row["metal_prefill_bytes"] = b.metalDense.decoder.PrefillScratchBytes()
		}
		emit(row)
	}
	// Warm every measured pipeline before starting the alternating series.
	run(-1, 0, modes[0])
	if len(modes) > 1 {
		run(-1, 1, "1")
	}
	for cycle := 0; cycle < cycles; cycle++ {
		for order, mode := range modes {
			run(cycle, order, mode)
		}
	}
	// Fixed-input ABBA pairs isolate decode scheduling from request-level drift.
	// Dense attention can safely overwrite the same KV row; hybrid recurrence
	// cannot, so Qwen retains only the complete chronological request comparison.
	if len(modes) > 1 && r.kind == loadedStandard {
		depth := promptDepth
		for i, token := range reference {
			var duration [2]time.Duration
			var expected uint32
			for order, mode := range modes {
				t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_ATTENTION", mode)
				start := time.Now()
				next, ok := tryMetalDenseGreedy(r.config, r.standard, r.workspaceCache, r.workspaceBuf, token, depth+i, reference[:i+1], opts.Sampler.RepeatPenalty)
				elapsed := time.Since(start)
				if !ok || (order > 0 && next != expected) {
					t.Fatal("fixed-input winner differs")
				}
				expected = next
				index := 0
				if mode == "1" {
					index = 1
				}
				duration[index] += elapsed / 2
			}
			emit(map[string]any{"phase": "decode_pair", "position": depth + i, "baseline_ns": duration[0].Nanoseconds(), "fused_ns": duration[1].Nanoseconds()})
		}
	}
	if r.kind == loadedQwen35 {
		if os.Getenv("GOPHERLLM_DECODE_QWEN_PAIRS") == "1" {
			// Optional fixed-input whole-forward comparison. The extra recurrent
			// checkpoint is test-only and allocated after the memory samples.
			b, k := r.workspaceBuf, r.workspaceCache
			checkpoint := k.Qwen35.snapshot()
			position := promptDepth + tokens - 1
			var expected string
			for pair := -1; pair < 8; pair++ {
				var duration [2]time.Duration
				for _, index := range []int{0, 1, 1, 0} {
					mode := strconv.Itoa(index)
					t.Setenv("GOPHERLLM_QWEN_OVERLAP_INPUTS", mode)
					t.Setenv("GOPHERLLM_METAL_QWEN_INPLACE_SILU", mode)
					if !k.Qwen35.restore(checkpoint) {
						t.Fatal("recurrent checkpoint mismatch")
					}
					start := time.Now()
					ForwardQwen35Into(r.config, r.qwen35, k, b, reference[len(reference)-1], position, &b.Logits)
					duration[index] += time.Since(start) / 2
					values := [][]float32{b.X, b.XN, b.Logits, k.Qwen35.Conv, k.Qwen35.State}
					for l := range k.K {
						values = append(values, k.K[l][position*k.PerPosKDim:(position+1)*k.PerPosKDim], k.V[l][position*k.PerPosVDim:(position+1)*k.PerPosVDim])
					}
					digest := ministralDecodeFloatDigest(t, values...)
					if expected != "" && expected != digest {
						t.Fatal("fixed Qwen forward state changed")
					}
					expected = digest
				}
				if pair >= 0 {
					emit(map[string]any{"phase": "qwen_forward_pair", "pair": pair, "position": position,
						"baseline_ns": duration[0].Nanoseconds(), "optimized_ns": duration[1].Nanoseconds(), "state_sha256": expected})
				}
			}
		}
		// Compare the changed projection schedule with actual model weights and
		// one fixed normalized input. This is a component latency, not token/s.
		for layer, l := range r.qwen35.Layers {
			w, b := l.DeltaNet, r.workspaceBuf
			x := slices.Clone(b.XN)
			t.Setenv("GOPHERLLM_QWEN_OVERLAP_INPUTS", "1")
			if !qwen35OverlapDeltaInputs(w, x, b) {
				continue
			}
			var expected string
			for pair := -2; pair < 32; pair++ {
				var duration [2]time.Duration
				for _, index := range []int{0, 1, 1, 0} {
					start := time.Now()
					if index == 1 {
						if !qwen35OverlapDeltaInputs(w, x, b) {
							t.Fatal("overlap declined")
						}
					} else {
						w.QKVConv.MatvecInto(x, &b.MambaIn)
						w.Gate.MatvecInto(x, &b.MambaZ)
						if !tryMatvec2Into(w.AlphaProj, w.BetaProj, x, &b.Q4KXSums, &b.MambaDT, &b.MambaBeta) {
							w.AlphaProj.MatvecInto(x, &b.MambaDT)
							w.BetaProj.MatvecInto(x, &b.MambaBeta)
						}
					}
					duration[index] += time.Since(start) / 2
					digest := ministralDecodeFloatDigest(t, b.MambaIn, b.MambaZ, b.MambaDT, b.MambaBeta)
					if expected != "" && digest != expected {
						t.Fatal("projection bits changed")
					}
					expected = digest
				}
				if pair >= 0 {
					emit(map[string]any{"phase": "projection_pair", "layer": layer, "pair": pair,
						"baseline_ns": duration[0].Nanoseconds(), "overlap_ns": duration[1].Nanoseconds(), "state_sha256": expected})
				}
			}
			break
		}
	}
	if r.kind == loadedGemma4 && r.gemma4.Native {
		var expected uint32
		for pair := -2; pair < 32; pair++ {
			var duration [2]time.Duration
			for order, index := range []int{0, 1, 1, 0} {
				t.Setenv("GOPHERLLM_METAL_GEMMA_PARALLEL_ARGMAX", strconv.Itoa(index))
				start := time.Now()
				next, ok := argmaxMetalGemmaOutput(r.gemma4.Output, r.workspaceBuf.XN, reference, 1)
				duration[index] += time.Since(start) / 2
				if !ok || (order > 0 && next != expected) {
					t.Fatal("Gemma output selection differs")
				}
				expected = next
			}
			if pair >= 0 {
				emit(map[string]any{"phase": "gemma_output_pair", "pair": pair, "baseline_ns": duration[0].Nanoseconds(), "parallel_ns": duration[1].Nanoseconds(), "token": expected})
			}
		}
	}
}
