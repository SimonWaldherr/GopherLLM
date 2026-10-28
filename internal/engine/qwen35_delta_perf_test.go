package engine

import (
	"math"
	"os"
	"testing"
	"time"
)

// Qwen3.8-27B's recurrent layer, excluding weight projections. Keep the
// complete state and convolution work in the measurement.
func BenchmarkQwen35DeltaNet27B(b *testing.B) {
	setPerformanceThreads(b, 8)
	cfg := Config{SSMInner: 6144, SSMState: 128, SSMHeads: 48, SSMGroups: 16, SSMConv: 4, RMSNormEps: 1e-6}
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	w := Qwen35DeltaNetWeights{
		ConvKernel: Weight{F32: randomExpertInput(channels*cfg.SSMConv, 43800)},
		A:          make([]float32, cfg.SSMHeads), DTBias: randomExpertInput(cfg.SSMHeads, 43801),
		Norm: make([]float32, cfg.SSMState),
	}
	for i := range w.A {
		w.A[i] = -1
	}
	for i := range w.Norm {
		w.Norm[i] = 1
	}
	qkv, gate := randomExpertInput(channels, 43802), randomExpertInput(cfg.SSMInner, 43803)
	alpha, beta := randomExpertInput(cfg.SSMHeads, 43804), randomExpertInput(cfg.SSMHeads, 43805)
	state, buf := newQwen35Cache(cfg, 1), &DecodeBuffer{}
	qwen35DeltaNetStep(cfg, w, state, qkv, gate, alpha, beta, 0, buf)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		qwen35DeltaNetStep(cfg, w, state, qkv, gate, alpha, beta, 0, buf)
	}
	for _, x := range buf.MambaY {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			b.Fatal("non-finite recurrent output")
		}
	}
}

// Opt-in full-model measurement for comparing two builds. Each request
// clears the prefix cache; logs include token digests for output parity.
func TestQwen35ConfiguredSerialPerformance(t *testing.T) {
	path := os.Getenv("GOPHERLLM_QWEN_PERF_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_QWEN_PERF_MODEL to a local Qwen hybrid GGUF")
	}
	r, _ := specABLoad(t, path, os.Getenv("GOPHERLLM_QWEN_PERF_METAL") == "1")
	if r.kind != loadedQwen35 {
		t.Fatal("requires Qwen hybrid weights")
	}
	tokens := r.tok.Encode("Write a Go function that adds two integers.\nfunc add(a, b int) int {")
	tokens = tokens[:min(16, len(tokens))]
	for run := range 4 {
		out, wall := specABRun(t, r, "", tokens, 16, 3*time.Minute, specABMode{Name: "serial"})
		t.Logf("run=%d warmup=%t prompt=%d generated=%d digest=%s prefill=%v decode=%v wall=%v",
			run, run == 0, out.Result.Stats.PromptTokens, len(out.Tokens), specABDigest(out.Tokens),
			out.Result.Stats.PrefillTime, out.Result.Stats.DecodeTime, wall)
	}
}
