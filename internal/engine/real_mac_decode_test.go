//go:build darwin && cgo && metal

package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func realMacRecord(phase string, fields map[string]any) {
	var r syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &r)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fields["phase"] = phase
	fields["rss_peak_bytes"] = r.Maxrss
	fields["minor_faults"] = r.Minflt
	fields["major_faults"] = r.Majflt
	fields["heap_alloc_bytes"] = m.HeapAlloc
	fields["heap_sys_bytes"] = m.HeapSys
	data, _ := json.Marshal(fields)
	fmt.Println("REAL_MAC_JSON " + string(data))
}

// Opt-in full checkpoint runs. Each process loads one model and emits phase
// boundaries for the external RSS/VM sampler. No downloads or cache purging.
func TestRealMacDecode(t *testing.T) {
	path := os.Getenv("GOPHERLLM_REAL_MAC_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_REAL_MAC_MODEL")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	setPerformanceThreads(t, 8)
	realMacRecord("before_load", map[string]any{})
	start := time.Now()
	if os.Getenv("GOPHERLLM_REAL_MAC_KIND") == "voxtral" {
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
		realMacRecord("loaded", map[string]any{"seconds": time.Since(start).Seconds()})
		c, w := model.source.cfg, model.source.weights
		// Full real decoder weights, deterministic embedding inputs. This is a
		// decoder test, not an audio transcription or recognition-quality test.
		var reference string
		for rep := 0; rep < 3; rep++ {
			state, e := NewVoxtralRealtimeDecoderState(c, w, c.Time.DefaultNumDelayTokens)
			if e != nil {
				t.Fatal(e)
			}
			fast := newVoxtralFastDecoder(c, w, state)
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
			realMacRecord("decode", map[string]any{"repetition": rep, "seconds": elapsed.Seconds(), "tokens": 64, "tokens_per_second": 64 / elapsed.Seconds(), "prefill_seconds": prefill.Seconds(), "token_digest": digest})
			fast.Close()
			runtime.GC()
			realMacRecord("decoder_closed", map[string]any{"repetition": rep})
		}
		return
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, Prefault: MmapPrefaultCore, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	realMacRecord("loaded", map[string]any{"seconds": time.Since(start).Seconds()})
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 64
	opts.Seed = 41
	opts.Sampler.Temperature = 0
	var digest string
	for rep := 0; rep < 4; rep++ {
		result, e := r.GenerateChat([]ChatMessage{UserMessage("Erkläre in fünf Sätzen, wie ein Computerprogramm ausgeführt wird.")}, opts)
		if e != nil {
			t.Fatal(e)
		}
		if result.Stats.GeneratedTokens != opts.MaxTokens {
			t.Fatalf("short decode: %d tokens, want %d", result.Stats.GeneratedTokens, opts.MaxTokens)
		}
		if r.workspaceBuf.metalDense == nil {
			t.Fatal("Metal decoder inactive")
		}
		d := specABDigest(r.workspaceBuf.GeneratedTokens[:result.Stats.GeneratedTokens])
		if digest != "" && digest != d {
			t.Fatal("repeated generation differs")
		}
		digest = d
		realMacRecord("decode", map[string]any{"repetition": rep, "seconds": result.Stats.DecodeTime.Seconds(), "tokens": result.Stats.GeneratedTokens, "tokens_per_second": float64(result.Stats.GeneratedTokens) / result.Stats.DecodeTime.Seconds(), "prefill_seconds": result.Stats.PrefillTime.Seconds(), "token_digest": d, "prompt_cache": result.PromptCache, "copied_weight_bytes": r.workspaceBuf.metalDense.copiedWeightBytes})
	}
}
