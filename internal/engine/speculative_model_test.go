package engine

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"testing"

	workerthreads "github.com/SimonWaldherr/GopherLLM/internal/threads"
)

// This optional test validates the complete generation path against a local
// model. Timings are diagnostics for this workload, not a kernel speed claim.
func TestSpeculativeConfiguredModelParity(t *testing.T) {
	path := os.Getenv("GOPHERLLM_VALIDATE_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_VALIDATE_MODEL to a local dense model")
	}
	useMetal := os.Getenv("GOPHERLLM_VALIDATE_METAL") == "1"
	if useMetal && !MetalAvailable() {
		t.Fatal(MetalError())
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: useMetal, Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	previousThreads := configuredThreads.Load()
	previousProcs := runtime.GOMAXPROCS(8)
	SetNumThreads(8)
	t.Cleanup(func() {
		configuredThreads.Store(previousThreads)
		workerthreads.Set(int(previousThreads))
		runtime.GOMAXPROCS(previousProcs)
	})
	if r.kind != loadedStandard {
		t.Skip("configured model is not a standard dense transformer")
	}
	var passage strings.Builder
	for i := 1; i <= 12; i++ {
		fmt.Fprintf(&passage, "%02d: The red fox jumps over the blue fence.\n", i)
	}
	messages := []ChatMessage{UserMessage("Copy all twelve numbered lines between BEGIN and END exactly, preserving their order. Output only the copied text, with no introduction.\nBEGIN\n" + passage.String() + "END")}
	blocks := 0
	for _, penalty := range []float32{1, 1.1} {
		t.Run(fmt.Sprintf("penalty_%.1f", penalty), func(t *testing.T) {
			opts := DefaultGenerationOptions()
			opts.MaxTokens, opts.Seed = 96, 41
			opts.Sampler.Temperature, opts.Sampler.TopK, opts.Sampler.RepeatPenalty = 0, 1, penalty
			r.ClearConversationCache()
			want, err := r.GenerateChat(messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			r.ClearConversationCache()
			opts.PromptLookupTokens = 7
			got, err := r.GenerateChat(messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != want.Text || got.FinishReason != want.FinishReason || got.Stats.GeneratedTokens != want.Stats.GeneratedTokens {
				t.Fatalf("lookup changed complete output: got=%+v want=%+v", got, want)
			}
			if got.Stats.Speculation == nil {
				t.Fatal("lookup statistics missing")
			}
			blocks += got.Stats.Speculation.Blocks
			t.Logf("tokens=%d serial_decode=%v lookup_decode=%v speculation=%+v", got.Stats.GeneratedTokens, want.Stats.DecodeTime, got.Stats.DecodeTime, *got.Stats.Speculation)
		})
	}
	if blocks == 0 {
		t.Fatal("configured model produced no lookup matches in either completion")
	}
}
