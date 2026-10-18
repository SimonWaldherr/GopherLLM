package gopherllm

import (
	"fmt"
	"io"
	"os"
	"slices"
	"testing"
	"time"
)

func loadMoEPerformanceModel(t testing.TB) *Runner {
	t.Helper()
	path := os.Getenv("GOPHERLLM_MOE_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_MOE_MODEL to a local MoE GGUF")
	}
	setPerformanceThreads(t, 8)
	metal := os.Getenv("GOPHERLLM_MOE_METAL") == "1"
	if metal && !MetalAvailable() {
		t.Fatal(MetalError())
	}
	r, info, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: metal, PrepareQuantized: false, Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	if r.config.ExpertCount < 2 {
		t.Fatal("configured model has no routed experts")
	}
	t.Logf("model=%s arch=%s layers=%d experts=%d used=%d dim=%d file_bytes=%d mapped=%t metal=%t q8=%t threads=8", path, r.arch, r.config.NLayers, r.config.ExpertCount, r.config.ExpertUsedCount, r.config.Dim, info.FileSizeBytes, r.IsMapped(), metal, useQ8Activations.Load())
	return r
}

// This isolates one real layer's complete FFN branch. Its input is synthetic;
// these numbers are not complete-model token throughput or prompt latency.
func BenchmarkMoEConfiguredLayer(b *testing.B) {
	r := loadMoEPerformanceModel(b)
	var sparse *SparseMoEWeights
	switch r.kind {
	case loadedQwen35:
		for _, layer := range r.qwen35.Layers {
			if layer.FFN.MoE != nil {
				sparse = layer.FFN.MoE
				break
			}
		}
	case loadedStandard:
		for _, layer := range r.standard.Layers {
			if layer.MoE != nil {
				sparse = layer.MoE
				break
			}
		}
	}
	if sparse == nil {
		b.Fatal("this isolated benchmark requires the common sparse SwiGLU graph")
	}
	b.Logf("first sparse layer: gate=%s up=%s down=%s hidden=%d shared=%t", sparse.Gate.Weight.Type, sparse.Up.Weight.Type, sparse.Down.Weight.Type, sparse.Gate.Output, sparse.SharedGate != nil)
	x := randomExpertInput(sparse.Gate.Input, 36001)
	// Prepare the oracle outside b.Run: a -bench filter may omit the serial
	// subbenchmark entirely, but the optimized case still needs a reference.
	reference := func() []float32 {
		previous := disableExpertDownBatch.Swap(true)
		defer disableExpertDownBatch.Store(previous)
		buf := &DecodeBuffer{}
		sparseMoEForward(sparse, x, buf)
		return slices.Clone(buf.Proj)
	}()
	for _, disabled := range []bool{true, false} {
		b.Run(fmt.Sprintf("serial_down=%t", disabled), func(b *testing.B) {
			// testing resets GOMAXPROCS before each subbenchmark.
			setPerformanceThreads(b, 8)
			previous := disableExpertDownBatch.Swap(disabled)
			defer disableExpertDownBatch.Store(previous)
			buf := &DecodeBuffer{}
			sparseMoEForward(sparse, x, buf)
			if !slices.Equal(buf.Proj, reference) {
				b.Fatal("real-layer output differs from established expert schedule")
			}
			if !disabled && len(buf.moeDown.down) == 0 {
				b.Fatal("configured model did not execute batched down")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				sparseMoEForward(sparse, x, buf)
			}
		})
	}
}

// Full model, unchanged greedy sampling, cleared prefix caches, one warmup
// followed by three measured pairs in alternating order. This checks the
// generated IDs and complete response as well as reporting prefill/decode.
func TestMoEConfiguredGenerationParity(t *testing.T) {
	r := loadMoEPerformanceModel(t)
	previous := disableExpertDownBatch.Load()
	t.Cleanup(func() { disableExpertDownBatch.Store(previous) })
	const prompt = "Write a short Go function that returns the larger of two integers."
	var reference *specABOutput
	for round := -1; round < 3; round++ {
		for order := range 2 {
			disabled := (round+order)&1 == 0
			disableExpertDownBatch.Store(disabled)
			// Keep Gemma's new gate/up schedule out of the baseline as well.
			oldGateBatch := disableExpertGateUpBatchForTest.Load()
			if r.kind == loadedGemma4 {
				disableExpertGateUpBatchForTest.Store(disabled)
			}
			output, wall := func() (specABOutput, time.Duration) {
				defer disableExpertGateUpBatchForTest.Store(oldGateBatch)
				return specABRun(t, r, prompt, nil, 64, 2*time.Minute, specABMode{Name: fmt.Sprintf("serial_down=%t", disabled)})
			}()
			if reference == nil {
				reference = &output
			} else if err := specABCheck(*reference, output); err != nil {
				t.Fatal(err)
			}
			t.Logf("round=%d order=%d serial_down=%t tokens=%d digest=%s wall=%v prefill=%v decode=%v", round, order, disabled, len(output.Tokens), specABDigest(output.Tokens), wall, output.Result.Stats.PrefillTime, output.Result.Stats.DecodeTime)
		}
	}
}
