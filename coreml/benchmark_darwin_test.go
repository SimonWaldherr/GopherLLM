//go:build darwin && cgo && coreml

package coreml_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/coreml"
)

// TestCoreMLConfiguredBenchmark measures the public embedding path, including
// tokenization and result ownership. Model loading and warm-up are reported
// separately. No request latency is asserted: other applications may share the
// CPU and Neural Engine. A compute plan describes placement, not utilization.
//
// GOPHERLLM_COREML_BENCH_MODELS is an OS path-list of compiled exports, and
// GOPHERLLM_COREML_GGUF is their common source checkpoint. The optional comma-
// separated GOPHERLLM_COREML_BENCH_UNITS defaults to cpu,ane. Use ane for export
// comparisons or ane,cpu to reverse device order. Repeat the model path-list in
// reverse order to expose drift. Only one native model is retained at a time.
func TestCoreMLConfiguredBenchmark(t *testing.T) {
	paths := os.Getenv("GOPHERLLM_COREML_BENCH_MODELS")
	if paths == "" {
		t.Skip("set GOPHERLLM_COREML_BENCH_MODELS and GOPHERLLM_COREML_GGUF")
	}
	gguf := os.Getenv("GOPHERLLM_COREML_GGUF")
	if gguf == "" {
		t.Fatal("the source GGUF is required for tokenizer and output parity")
	}
	modes := coremlBenchmarkModes(t, os.Getenv("GOPHERLLM_COREML_BENCH_UNITS"))
	tok, err := coreml.LoadTokenizer(gguf)
	if err != nil {
		t.Fatal(err)
	}
	base := []string{
		"Wie wird ein Computerprogramm ausgeführt?",
		"Arbeitsspeicher",
		"The processor executes instructions.",
		"An embedding encoder retrieves related documents.",
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	r, _, err := gopherllm.RunnerFromPathWithOptions(gguf, gopherllm.LoadOptions{
		Prefault: gopherllm.MmapPrefaultNone, LogWriter: io.Discard,
	})
	if err != nil {
		t.Fatal(err)
	}
	reference, embedErr := r.EmbedBatch(ctx, base)
	closeErr := r.Close()
	if embedErr != nil || closeErr != nil {
		t.Fatalf("GGUF reference: embed=%v close=%v", embedErr, closeErr)
	}
	for _, path := range filepath.SplitList(paths) {
		if path == "" {
			t.Fatal("empty compiled model path")
		}
		for _, mode := range modes {
			if !t.Run(filepath.Base(path)+"/"+mode.name, func(t *testing.T) {
				began := time.Now()
				e, err := coreml.OpenEmbedder(path, tok, coreml.Options{
					ComputeUnits: mode.units, PadToken: tok.PadToken,
				})
				loadTime := time.Since(began)
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := e.Close(); err != nil {
						t.Error(err)
					}
				}()
				plan := coreml.Plan{}
				planErr := fmt.Errorf("disabled via GOPHERLLM_COREML_BENCH_PLAN=0")
				if os.Getenv("GOPHERLLM_COREML_BENCH_PLAN") != "0" {
					plan, planErr = coreml.Inspect(path, mode.units)
				}
				if planErr != nil {
					// Placement diagnostics can fail independently of prediction
					// (for example a compiler/service error under memory pressure).
					// Keep that failure visible without discarding valid timings.
					t.Logf("compute plan unavailable: %v", planErr)
				}
				if planErr == nil && (plan.GPUOperations != 0 || mode.units == coreml.CPUOnly && plan.NeuralEngineOperations != 0) {
					t.Fatalf("compute plan exceeds permitted devices: %+v", plan)
				}
				for _, count := range []int{1, 4, 17, 32} {
					if !t.Run(fmt.Sprint(count), func(t *testing.T) {
						coremlBenchmarkRequests(t, ctx, e, path, mode.name, base, reference,
							count, loadTime, plan, planErr)
					}) {
						return
					}
				}
			}) {
				return
			}
		}
	}
}

type coremlBenchmarkMode struct {
	name  string
	units coreml.ComputeUnits
}

func coremlBenchmarkModes(t *testing.T, value string) []coremlBenchmarkMode {
	t.Helper()
	if value == "" {
		value = "cpu,ane"
	}
	var modes []coremlBenchmarkMode
	for _, name := range strings.Split(value, ",") {
		switch strings.TrimSpace(name) {
		case "cpu":
			modes = append(modes, coremlBenchmarkMode{"cpu", coreml.CPUOnly})
		case "ane":
			modes = append(modes, coremlBenchmarkMode{"cpu_neural_engine", coreml.CPUAndNeuralEngine})
		default:
			t.Fatalf("GOPHERLLM_COREML_BENCH_UNITS must be a comma-separated sequence of cpu and ane; got %q", value)
		}
	}
	return modes
}

func coremlBenchmarkRequests(t *testing.T, ctx context.Context, e *coreml.Embedder,
	path, mode string, base []string, reference []gopherllm.EmbeddingResult,
	count int, loadTime time.Duration, plan coreml.Plan, planErr error) {
	t.Helper()
	const warmups, sampleCount, allocationRuns = 3, 9, 3
	const sampleTarget = 50 * time.Millisecond
	texts := make([]string, count)
	inputTokens := 0
	for i := range texts {
		texts[i] = base[i%len(base)]
		inputTokens += reference[i%len(reference)].TokenCount
	}
	minimumCosine := float32(1)
	check := func(rows []gopherllm.EmbeddingResult) {
		t.Helper()
		if len(rows) != count {
			t.Fatalf("output count: got %d, want %d", len(rows), count)
		}
		for i, row := range rows {
			want := reference[i%len(reference)]
			cosine, err := gopherllm.CosineSimilarity(row.Embedding, want.Embedding)
			if err != nil || !isFiniteCoreMLBenchmark(float64(cosine)) || cosine < .995 || row.TokenCount != want.TokenCount {
				t.Fatalf("text %d source parity: cosine=%g tokens=%d/%d err=%v", i, cosine, row.TokenCount, want.TokenCount, err)
			}
			minimumCosine = min(minimumCosine, cosine)
		}
	}
	var firstWarm time.Duration
	for warmup := range warmups {
		began := time.Now()
		rows, err := e.EmbedBatch(ctx, texts)
		elapsed := time.Since(began)
		if err != nil {
			t.Fatal(err)
		}
		if warmup == 0 {
			firstWarm = elapsed
		}
		check(rows)
	}
	type sample struct {
		Requests    int     `json:"requests"`
		WallSeconds float64 `json:"wall_seconds"`
		MeanSeconds float64 `json:"mean_seconds"`
	}
	samples := make([]sample, 0, sampleCount)
	requestTimes := make([]float64, 0, 256)
	for range sampleCount {
		beganSample := time.Now()
		total, calls := time.Duration(0), 0
		var lastRows []gopherllm.EmbeddingResult
		for calls == 0 || time.Since(beganSample) < sampleTarget {
			began := time.Now()
			rows, err := e.EmbedBatch(ctx, texts)
			elapsed := time.Since(began)
			if err != nil {
				t.Fatal(err)
			}
			requestTimes = append(requestTimes, elapsed.Seconds())
			total += elapsed
			calls++
			lastRows = rows
		}
		samples = append(samples, sample{calls, time.Since(beganSample).Seconds(), total.Seconds() / float64(calls)})
		check(lastRows) // Parity and report allocations stay outside timed calls.
	}
	slices.Sort(requestTimes)
	median := requestTimes[len(requestTimes)/2]
	p90 := requestTimes[(len(requestTimes)*90+99)/100-1]

	// These are Go heap costs only. Core ML and ANE allocations are outside Go's
	// allocator and must not be inferred from these numbers.
	var allocationErr error
	allocations := testing.AllocsPerRun(allocationRuns, func() {
		_, err := e.EmbedBatch(ctx, texts)
		if allocationErr == nil {
			allocationErr = err
		}
	})
	if allocationErr != nil {
		t.Fatal(allocationErr)
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	for range allocationRuns {
		if _, err := e.EmbedBatch(ctx, texts); err != nil {
			t.Fatal(err)
		}
	}
	runtime.ReadMemStats(&after)
	info := e.Info()
	predictions := (count + info.BatchSize - 1) / info.BatchSize
	paddedPositions := predictions * info.BatchSize * info.SequenceLength
	planError := ""
	if planErr != nil {
		planError = planErr.Error()
	}
	report, err := json.Marshal(map[string]any{
		"schema_version": 1, "model": path, "mode": mode, "texts": count,
		"info": info, "compute_plan": plan, "compute_plan_available": planErr == nil, "compute_plan_error": planError,
		"measured_at":        time.Now().UTC().Format(time.RFC3339Nano),
		"model_load_seconds": loadTime.Seconds(), "first_prediction_seconds": firstWarm.Seconds(),
		"warmup_calls": warmups, "sample_target_seconds": sampleTarget.Seconds(), "samples": samples,
		"requests_measured": len(requestTimes), "median_seconds": median, "p90_seconds": p90,
		"min_seconds": requestTimes[0], "max_seconds": requestTimes[len(requestTimes)-1],
		"texts_per_second": float64(count) / median, "input_tokens_per_request": inputTokens,
		"model_predictions_per_request": predictions, "padded_token_positions_per_request": paddedPositions,
		"useful_token_fraction": float64(inputTokens) / float64(paddedPositions),
		"go_allocs_per_request": allocations, "go_bytes_per_request": float64(after.TotalAlloc-before.TotalAlloc) / allocationRuns,
		"minimum_gguf_cosine": minimumCosine,
		"host": map[string]any{"os": runtime.GOOS, "arch": runtime.GOARCH, "cpu_count": runtime.NumCPU(),
			"gomaxprocs": runtime.GOMAXPROCS(0), "go_version": runtime.Version()},
	})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("COREML_BENCHMARK_JSON " + string(report))
}

func isFiniteCoreMLBenchmark(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
