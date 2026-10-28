package engine

import (
	"fmt"
	"math/rand"
	"os"
	"slices"
	"testing"
)

// BenchmarkSpeculativeCPUConfiguredOutput isolates a real native Qwen MTP
// checkpoint's CPU vocabulary head. Inputs are fixed synthetic hidden rows;
// no target or draft layers are evaluated. Supply GOPHERLLM_SPEC_AB_RAW_MODEL
// and run with -benchtime=200ms -count=3. This compares vocabulary projection
// schedules, not complete-model generation speed. Production's P>=4 gate is
// deliberately bypassed only in this benchmark to measure P2 and P3.
func BenchmarkSpeculativeCPUConfiguredOutput(b *testing.B) {
	path := os.Getenv("GOPHERLLM_SPEC_AB_RAW_MODEL")
	if path == "" {
		b.Skip("set GOPHERLLM_SPEC_AB_RAW_MODEL to a local native Qwen MTP checkpoint")
	}
	r, threads := specABLoad(b, path, false)
	if r.kind != loadedQwen35 || r.qwen35.MTP == nil {
		b.Fatal("a native Qwen checkpoint with an MTP head is required")
	}
	if !useQ8Activations.Load() {
		b.Skip("CPU Q8 activation kernels unavailable on this host")
	}
	cfg, w := r.config, r.qwen35.Output
	if w.F32 != nil || w.Metal != nil || w.GPU != nil || w.Rows <= 0 || w.Cols != cfg.Dim || len(w.Raw) == 0 {
		b.Fatal("the benchmark requires an unexpanded CPU quantized vocabulary matrix")
	}
	weights := ModelWeights{Output: w}
	b.Logf("CPU vocabulary only: rows=%d cols=%d type=%d raw_bytes=%d threads=%d loaded_target_layers=%d evaluated_layers=0 penalty=1 synthetic_hidden=true production_min_positions_unchanged=4",
		w.Rows, w.Cols, w.Type, len(w.Raw), threads, cfg.NLayers)
	const maximumPositions = 8
	hiddenFlat := make([]float32, maximumPositions*cfg.Dim)
	hidden := make([][]float32, maximumPositions)
	rng := rand.New(rand.NewSource(3508248))
	for i := range hidden {
		hidden[i] = hiddenFlat[i*cfg.Dim : (i+1)*cfg.Dim : (i+1)*cfg.Dim]
		for j := range hidden[i] {
			hidden[i][j] = float32(rng.NormFloat64()) * .2
		}
	}
	for _, positions := range []int{2, 3, 4, 8} {
		rows := hidden[:positions]
		serialTokens, batchTokens := make([]uint32, positions), make([]uint32, positions)
		serialBuf := &DecodeBuffer{}
		var logitsFlat []float32
		var logits [][]float32
		logits = reuseBatchViews(&logitsFlat, &logits, positions, w.Rows)
		serial := func(t testing.TB) {
			for i, row := range rows {
				serialBuf.XN = row
				token, ok := argmaxOutputTokenPenalizedInto(cfg, weights, serialBuf, nil, 1, &serialBuf.Logits)
				if !ok {
					t.Fatal("the established serial CPU argmax path declined the configured head")
				}
				serialTokens[i] = token
			}
		}
		batch := func(t testing.TB) {
			if !matvecBatchQ8(w, rows, logits) {
				t.Fatal("the configured vocabulary head declined CPU Q8 batching")
			}
			for i, row := range logits {
				batchTokens[i] = argmaxFiniteToken(row)
			}
		}
		serial(b)
		batch(b)
		if !slices.Equal(serialTokens, batchTokens) {
			b.Fatalf("P%d CPU vocabulary schedule changed greedy tokens: batch=%v serial=%v", positions, batchTokens, serialTokens)
		}
		b.Logf("P%d exact_greedy_tokens=%v", positions, serialTokens)
		for _, mode := range []struct {
			name string
			run  func(testing.TB)
		}{{"serial_argmax", serial}, {"batch_q8_argmax", batch}} {
			b.Run(fmt.Sprintf("P%d/%s", positions, mode.name), func(b *testing.B) {
				setPerformanceThreads(b, threads)
				mode.run(b)
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					mode.run(b)
				}
				b.ReportMetric(float64(positions), "tokens/op")
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(positions), "ns/token")
			})
		}
	}
}
