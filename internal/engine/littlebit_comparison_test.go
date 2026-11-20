package engine

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

func openQwen3ComparisonModel(t *testing.T) (*Runner, LoadInfo, bool, string) {
	t.Helper()
	path := os.Getenv("GOPHERLLM_QWEN3_BENCH_GGUF")
	if path == "" {
		t.Skip("set GOPHERLLM_QWEN3_BENCH_GGUF to either Qwen3-8B GGUF")
	}
	useMetal := os.Getenv("GOPHERLLM_QWEN3_BENCH_METAL") == "1"
	if useMetal && !MetalAvailable() {
		t.Fatal("Metal requested but unavailable")
	}
	r, info, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: useMetal})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	if r.Architecture() != "qwen3" || r.config.NLayers != 36 || r.config.VocabSize != 151936 {
		t.Fatalf("expected dense Qwen3-8B, got %s with %d layers and %d vocabulary rows", r.Architecture(), r.config.NLayers, r.config.VocabSize)
	}
	return r, info, useMetal, path
}

// TestQwen3LittleBitComparison runs the same generation request for either
// Qwen3-8B GGUF. Run one model per process so peak RSS is attributable.
func TestQwen3LittleBitComparison(t *testing.T) {
	r, info, useMetal, path := openQwen3ComparisonModel(t)
	analysis := AnalyzeGGUF(r.GGUF(), nil)
	opts := DefaultGenerationOptions()
	opts.MaxTokens, opts.SystemPrompt, opts.Seed = 16, "", 1
	opts.Sampler.Temperature, opts.Sampler.TopK, opts.Sampler.RepeatPenalty = 0, 1, 1
	const prompt = "Explain why the sky is blue, then name the physical effect in one short sentence."
	promptIDs, _, err := r.renderMessagesForGeneration([]ChatMessage{UserMessage(prompt)}, opts.SystemPrompt, nil, "")
	if err != nil {
		t.Fatal(err)
	}

	// Warm caches and pipeline compilation first. Reset the prompt cache in
	// each measurement so both backends process exactly the same input again.
	if _, err := r.Generate(prompt, opts); err != nil {
		t.Fatal(err)
	}
	t.Logf("path=%s variant=%d metal=%t gguf_bytes=%d weights_bytes=%d load_seconds=%.3f head_type=%s head_raw_bytes=%d head_f32_bytes=%d head_metal=%t resident_decode=%t prompt_digest=%s",
		path, analysis.LittleBitVariant, useMetal, info.FileSizeBytes, analysis.FileBytes, info.LoadTime.Seconds(),
		r.standard.Output.Type, len(r.standard.Output.Raw), len(r.standard.Output.F32)*4, r.standard.Output.Metal != nil,
		r.workspaceBuf != nil && r.workspaceBuf.metalDense != nil, specABDigest(promptIDs))
	var wantDigest string
	for i := range 3 {
		r.clearPrefixCache()
		got, err := r.Generate(prompt, opts)
		if err != nil {
			t.Fatal(err)
		}
		s := got.Stats
		if s.PromptTokens != len(promptIDs) || s.GeneratedTokens == 0 || cap(r.workspaceBuf.GeneratedTokens) < s.GeneratedTokens {
			t.Fatalf("unexpected token counts: %+v", s)
		}
		digest := specABDigest(r.workspaceBuf.GeneratedTokens[:s.GeneratedTokens])
		if i == 0 {
			wantDigest = digest
		} else if digest != wantDigest {
			t.Fatalf("greedy output changed in run %d: got %s, want %s", i+1, digest, wantDigest)
		}
		t.Logf("run=%d prompt=%d generated=%d prefill_tps=%.3f decode_tps=%.3f ttft_ms=%.3f digest=%s text=%q reasoning=%q",
			i+1, s.PromptTokens, s.GeneratedTokens, float64(s.PromptTokens)/s.PrefillTime.Seconds(),
			float64(s.GeneratedTokens)/s.DecodeTime.Seconds(), s.TTFT.Seconds()*1000, digest, got.Text, got.ReasoningText)
	}
}

// TestQwen3LittleBitFixedDecodeComparison exercises the native prefill and
// greedy decode entry points for fixed lengths, independent of model-specific
// stop tokens and chat templates. It measures compute, not response quality.
func TestQwen3LittleBitFixedDecodeComparison(t *testing.T) {
	r, _, useMetal, path := openQwen3ComparisonModel(t)
	if !r.canBatchPrefill() {
		t.Fatal("fixed-length comparison needs batched prefill enabled")
	}
	const promptTokens, decodeSteps = 128, 128
	prompt := r.tok.EncodeWithoutBOS(strings.Repeat("Air scatters blue light more than red light. ", 32))
	if len(prompt) < promptTokens {
		t.Fatalf("comparison prompt has only %d tokens", len(prompt))
	}
	prompt = prompt[:promptTokens]
	cache, buf := r.generationWorkspace(promptTokens + decodeSteps)
	var logits []float32
	var output [decodeSteps]uint32
	var expected string
	t.Logf("path=%s metal=%t prompt_digest=%s", path, useMetal, specABDigest(prompt))
	for i := 0; i <= 3; i++ {
		start := time.Now()
		if err := r.prefillBatched(context.Background(), cache, buf, prompt, &logits); err != nil {
			t.Fatal(err)
		}
		middle := time.Now()
		next := argmaxFiniteToken(logits)
		for j := range output {
			var ok bool
			next, ok = r.forwardGreedyToken(cache, buf, next, promptTokens+j, nil, 1, &logits)
			if !ok {
				next = argmaxFiniteToken(logits)
			}
			output[j] = next
		}
		end := time.Now()
		digest := specABDigest(output[:])
		if i == 0 {
			expected = digest
			continue
		}
		if digest != expected {
			t.Fatalf("greedy output changed in run %d: got %s, want %s", i, digest, expected)
		}
		t.Logf("run=%d prompt=%d steps=%d prefill_tps=%.3f decode_tps=%.3f resident_decode=%t digest=%s",
			i, promptTokens, decodeSteps, promptTokens/middle.Sub(start).Seconds(), decodeSteps/end.Sub(middle).Seconds(), buf.metalDense != nil, digest)
	}
}
