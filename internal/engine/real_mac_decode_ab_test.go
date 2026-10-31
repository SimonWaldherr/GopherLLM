//go:build darwin && cgo && metal

package engine

import (
	"context"
	"io"
	"os"
	"runtime"
	"strconv"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

// Toggle only the pipeline selection, retaining the same mapped model, sampler
// and workload. Initial requests warm both paths before alternating pairs.
func TestRealMacDecodeKernels(t *testing.T) {
	path, kind := os.Getenv("GOPHERLLM_REAL_MAC_MODEL"), os.Getenv("GOPHERLLM_REAL_MAC_KIND")
	if path == "" {
		t.Skip("set GOPHERLLM_REAL_MAC_MODEL")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	setPerformanceThreads(t, 8)
	option := "GOPHERLLM_METAL_DENSE_Q4_SPLIT_SUM"
	if kind == "voxtral" {
		option = "GOPHERLLM_METAL_DENSE_FUSED_GATE"
	}
	pairs := 8
	if value := os.Getenv("GOPHERLLM_REAL_MAC_PAIRS"); value != "" {
		n, err := strconv.Atoi(value)
		if err != nil || n < 2 || n > 32 || n%2 != 0 {
			t.Fatal("GOPHERLLM_REAL_MAC_PAIRS must be an even number in 2..32")
		}
		pairs = n
	}

	type result struct {
		seconds, prefill float64
		digest           string
		fields           map[string]any
	}
	var run func() result
	if kind == "voxtral" {
		model, err := OpenVoxtral(context.Background(), path, WithVoxtralMetal(true))
		if err != nil {
			t.Fatal(err)
		}
		defer model.Close()
		c, w := model.source.cfg, model.source.weights
		run = func() result {
			state, err := NewVoxtralRealtimeDecoderState(c, w, c.Time.DefaultNumDelayTokens)
			if err != nil {
				t.Fatal(err)
			}
			fast := newVoxtralFastDecoder(c, w, state, model.source.metalMapping)
			if fast == nil {
				t.Fatal("Metal decoder inactive")
			}
			defer fast.Close()
			prefill, elapsed, tokens := realMacVoxtralSteps(t, c, w, fast)
			gpu := fast.(*voxtralMetalDecoder)
			return result{elapsed.Seconds(), prefill.Seconds(), specABDigest(tokens), map[string]any{"decoder_kv_bytes": int64(gpu.capacity) * int64(c.Decoder.NKVHeads) * 128 * 4 * 2 * int64(len(w.Decoder.Layers)), "decoder_kv_capacity": gpu.capacity, "decoder_copied_weight_bytes": gpu.copiedWeightBytes, "decoder_borrowed_weight_bytes": gpu.borrowedWeightBytes}}
		}
	} else if kind == "ministral" {
		r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultCore, LogWriter: io.Discard})
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		opts := DefaultGenerationOptions()
		opts.MaxTokens = 64
		opts.Seed = 41
		opts.Sampler.Temperature = 0
		run = func() result {
			got, err := r.GenerateChat([]ChatMessage{UserMessage("Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird.")}, opts)
			if err != nil {
				t.Fatal(err)
			}
			if got.Stats.GeneratedTokens != 64 {
				t.Fatalf("generated %d tokens, want 64", got.Stats.GeneratedTokens)
			}
			gpu := r.workspaceBuf.metalDense
			if gpu == nil {
				t.Fatal("dense decoder inactive")
			}
			return result{got.Stats.DecodeTime.Seconds(), got.Stats.PrefillTime.Seconds(), specABDigest(r.workspaceBuf.GeneratedTokens[:64]), map[string]any{"copied_weight_bytes": gpu.copiedWeightBytes, "prompt_cache": got.PromptCache}}
		}
	} else {
		t.Fatal("set GOPHERLLM_REAL_MAC_KIND to ministral or voxtral")
	}
	reference := ""
	sample := func(pair int, variant string) {
		t.Setenv(option, variant)
		_, _, faults, pageins, ok := metalbackend.ProcessMemory()
		if !ok {
			t.Fatal("Mach memory counters unavailable")
		}
		r := run()
		_, _, endFaults, endPageins, ok := metalbackend.ProcessMemory()
		if !ok {
			t.Fatal("Mach memory counters unavailable")
		}
		if reference != "" && reference != r.digest {
			t.Fatalf("variant %s changed tokens", variant)
		}
		reference = r.digest
		r.fields["pair"], r.fields["optimized"], r.fields["option"] = pair, variant == "1", option
		r.fields["seconds"], r.fields["prefill_seconds"] = r.seconds, r.prefill
		r.fields["tokens"], r.fields["tokens_per_second"], r.fields["token_digest"] = 64, 64/r.seconds, r.digest
		r.fields["request_page_faults"], r.fields["request_pageins"] = endFaults-faults, endPageins-pageins
		realMacRecord(t, "ab_decode", r.fields)
	}
	sample(-1, "0")
	sample(-1, "1")
	for pair := 0; pair < pairs; pair++ {
		variants := []string{"0", "1"}
		if pair%2 != 0 {
			variants = []string{"1", "0"}
		}
		for _, variant := range variants {
			sample(pair, variant)
		}
		runtime.GC()
	}
}
