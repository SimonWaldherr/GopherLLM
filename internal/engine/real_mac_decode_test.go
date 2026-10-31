//go:build darwin && cgo && metal

package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

func realMacRecord(t testing.TB, phase string, fields map[string]any) {
	t.Helper()
	var r syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &r); err != nil {
		t.Fatal(err)
	}
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fields["phase"] = phase
	fields["unix_time"] = float64(time.Now().UnixNano()) / 1e9
	fields["rss_peak_bytes"] = r.Maxrss
	fields["minor_faults"] = r.Minflt
	fields["major_faults"] = r.Majflt
	fields["heap_alloc_bytes"] = m.HeapAlloc
	fields["heap_sys_bytes"] = m.HeapSys
	data, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("REAL_MAC_JSON " + string(data))
}

// Opt-in full checkpoint runs. Each process loads one model and emits phase
// boundaries for the external RSS/VM sampler. No downloads or cache purging.
func TestRealMacDecode(t *testing.T) {
	path := os.Getenv("GOPHERLLM_REAL_MAC_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_REAL_MAC_MODEL")
	}
	kind := os.Getenv("GOPHERLLM_REAL_MAC_KIND")
	if kind != "ministral" && kind != "voxtral" {
		t.Fatal("set GOPHERLLM_REAL_MAC_KIND to ministral or voxtral")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	setPerformanceThreads(t, 8)
	realMacRecord(t, "before_load", map[string]any{})
	start := time.Now()
	if kind == "voxtral" {
		ctx := context.Background()
		var model *VoxtralModel
		var err error
		if os.Getenv("GOPHERLLM_REAL_MAC_COPY") == "1" {
			mapping, e := OpenMmap(path)
			if e != nil {
				t.Fatal(e)
			}
			defer mapping.Close()
			model, err = openVoxtralBytes(ctx, mapping.Bytes(), nil, WithVoxtralMetal(true))
		} else {
			model, err = OpenVoxtral(ctx, path, WithVoxtralMetal(true))
		}
		if err != nil {
			t.Fatal(err)
		}
		defer model.Close()
		realMacRecord(t, "loaded", map[string]any{"seconds": time.Since(start).Seconds()})
		c, w := model.source.cfg, model.source.weights
		// Full real decoder weights, deterministic embedding inputs. This is a
		// decoder test, not an audio transcription or recognition-quality test.
		var reference string
		for rep := 0; rep < 3; rep++ {
			state, e := NewVoxtralRealtimeDecoderState(c, w, c.Time.DefaultNumDelayTokens)
			if e != nil {
				t.Fatal(e)
			}
			fast := newVoxtralFastDecoder(c, w, state, model.source.metalMapping)
			if fast == nil {
				t.Fatal("Metal decoder inactive")
			}
			defer fast.Close() // Also release the decoder if a step fails.
			began := time.Now()
			for pos := 0; pos < 32; pos++ {
				x, e := EmbedVoxtralRealtimeToken(c, w, (pos+10)%c.Decoder.VocabSize)
				if e != nil {
					t.Fatal(e)
				}
				if _, e = fast.Step(x, pos, false); e != nil {
					t.Fatal(e)
				}
			}
			prefill := time.Since(began)
			tokens := make([]uint32, 64)
			began = time.Now()
			for j := range tokens {
				x, e := EmbedVoxtralRealtimeToken(c, w, j+42)
				if e != nil {
					t.Fatal(e)
				}
				token, e := fast.Step(x, j+32, true)
				if e != nil {
					t.Fatal(e)
				}
				tokens[j] = uint32(token)
			}
			elapsed := time.Since(began)
			digest := specABDigest(tokens)
			if reference != "" && digest != reference {
				t.Fatal("repeated decoder tokens differ")
			}
			reference = digest
			gpu := fast.(*voxtralMetalDecoder)
			realMacRecord(t, "decode", map[string]any{"repetition": rep, "seconds": elapsed.Seconds(), "tokens": 64, "tokens_per_second": 64 / elapsed.Seconds(), "prefill_seconds": prefill.Seconds(), "token_digest": digest, "decoder_kv_capacity": gpu.capacity, "decoder_kv_bytes": int64(gpu.capacity) * int64(c.Decoder.NKVHeads) * 128 * 4 * 2 * int64(len(w.Decoder.Layers)), "decoder_borrowed_weight_bytes": gpu.borrowedWeightBytes, "decoder_copied_weight_bytes": gpu.copiedWeightBytes})
			fast.Close()
			runtime.GC()
			realMacRecord(t, "decoder_closed", map[string]any{"repetition": rep})
		}
		return
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultCore, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	realMacRecord(t, "loaded", map[string]any{"seconds": time.Since(start).Seconds(), "dim": r.config.Dim, "head_dim": r.config.HeadDim, "value_dim": r.config.ValueDim, "heads": r.config.NHeads, "kv_heads": r.config.NKVHeads})
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 64
	opts.Seed = 41
	opts.Sampler.Temperature = 0
	longContext := os.Getenv("GOPHERLLM_REAL_MAC_LONG_CONTEXT") == "1"
	prompt := "Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird."
	if longContext {
		bytesPerPosition := kvCacheBytes(r.config.NLayers, r.config.NKVHeads*r.config.HeadDim, r.config.NKVHeads*r.config.ValueDim, 1, kvF32)
		target := int(maxReusableKVCacheBytes/bytesPerPosition) + 32
		lo, hi := 0, target*2
		for lo < hi {
			mid := lo + (hi-lo)/2
			text := strings.Repeat(" memory", mid) + "\n" + prompt
			if len(r.renderMessages([]ChatMessage{UserMessage(text)}, opts.SystemPrompt, nil)) < target {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		prompt = strings.Repeat(" memory", lo) + "\n" + prompt
	}
	var digest string
	preparedMetalWeights := 0
	for _, layer := range r.standard.Layers {
		for _, weight := range [...]Weight{layer.WQ, layer.WK, layer.WV, layer.WO, layer.W1, layer.W3, layer.W2, layer.WQKV, layer.WGateUp} {
			if weight.Metal != nil {
				preparedMetalWeights++
			}
		}
	}
	for rep := 0; rep < 4; rep++ {
		result, e := r.GenerateChat([]ChatMessage{UserMessage(prompt)}, opts)
		if e != nil {
			t.Fatal(e)
		}
		if result.Stats.GeneratedTokens != opts.MaxTokens {
			t.Fatalf("short decode: %d tokens, want %d", result.Stats.GeneratedTokens, opts.MaxTokens)
		}
		// Selective offload is also a valid workload when a checkpoint or
		// configuration cannot use the complete dense Metal graph.
		dense := r.workspaceBuf.metalDense
		retainedMetalKV := int64(0)
		if dense != nil && dense.cache != r.workspaceCache {
			retainedMetalKV = kvCacheStorageBytes(dense.cache)
		}
		if longContext {
			runtime.GC()
		}
		if dense == nil && preparedMetalWeights == 0 {
			t.Fatal("no Metal decoder or prepared projections")
		}
		d := specABDigest(r.workspaceBuf.GeneratedTokens[:result.Stats.GeneratedTokens])
		if digest != "" && digest != d {
			t.Fatal("repeated generation differs")
		}
		digest = d
		fields := map[string]any{"repetition": rep, "seconds": result.Stats.DecodeTime.Seconds(), "tokens": result.Stats.GeneratedTokens, "tokens_per_second": float64(result.Stats.GeneratedTokens) / result.Stats.DecodeTime.Seconds(), "prefill_seconds": result.Stats.PrefillTime.Seconds(), "token_digest": d, "prompt_cache": result.PromptCache, "dense_metal": dense != nil, "prepared_metal_weights": preparedMetalWeights, "long_context": longContext, "gc_before_record": longContext, "retained_transient_metal_kv_bytes": retainedMetalKV}
		if dense != nil {
			// This counter covers buffers created by the dense graph only.
			fields["copied_weight_bytes"] = dense.copiedWeightBytes
		}
		realMacRecord(t, "decode", fields)
	}
}
