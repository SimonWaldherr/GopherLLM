package gopherllm

import (
	"context"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"testing"

	workerthreads "github.com/SimonWaldherr/GopherLLM/internal/threads"
)

// BenchmarkPrefillConfiguredModel always executes the prompt graph. Repeated
// Generate calls with one prompt would instead measure the exact-prompt cache.
// Loading, workspace allocation and warm-up are outside the timed loop.
func BenchmarkPrefillConfiguredModel(b *testing.B) {
	path := os.Getenv("GOPHERLLM_BENCH_MODEL")
	if path == "" {
		b.Skip("set GOPHERLLM_BENCH_MODEL=/path/to/model.gguf to benchmark uncached prefill")
	}
	if raw := os.Getenv("GOPHERLLM_BENCH_THREADS"); raw != "" {
		threads, err := strconv.Atoi(raw)
		if err != nil || threads < 1 {
			b.Fatalf("GOPHERLLM_BENCH_THREADS must be a positive integer, got %q", raw)
		}
		previousThreads := configuredThreads.Load()
		previousProcs := runtime.GOMAXPROCS(threads)
		SetNumThreads(threads)
		b.Cleanup(func() {
			configuredThreads.Store(previousThreads)
			workerthreads.Set(int(previousThreads))
			runtime.GOMAXPROCS(previousProcs)
		})
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{
		UseMetal: os.Getenv("GOPHERLLM_BENCH_METAL") == "1", LogWriter: io.Discard,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	if !r.canBatchPrefill() && !r.canBatchNativeGemma4() {
		b.Skip("model does not support batched prefill")
	}
	for _, n := range []int{32, 128, 512} {
		if n > r.config.MaxSeqLen {
			continue
		}
		b.Run(fmt.Sprintf("tokens=%d", n), func(b *testing.B) {
			tokens := make([]uint32, n)
			state := uint32(1)
			for i := range tokens {
				state = state*1664525 + 1013904223
				tokens[i] = state % uint32(r.config.VocabSize)
			}
			cache, buf := r.generationWorkspace(n)
			ctx := context.Background()
			var logits []float32
			if err := r.prefillBatchedAt(ctx, cache, buf, tokens, 0, &logits); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := r.prefillBatchedAt(ctx, cache, buf, tokens, 0, &logits); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
			if len(logits) == 0 || !finite32(logits[0]) {
				b.Fatal("prefill returned invalid logits")
			}
		})
	}
}
