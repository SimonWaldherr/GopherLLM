//go:build darwin && cgo && coreml

package coreml

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// Compare the former scalar validation with Predict on one warm native model.
// Alternate the order within every pair to limit drift from other applications.
// This measures the FFN round trip, not an entire prefill or model generation.
func TestFFNConfiguredValidationBenchmark(t *testing.T) {
	path := os.Getenv("GOPHERLLM_COREML_FFN_PATH")
	if path == "" {
		t.Skip("set GOPHERLLM_COREML_FFN_PATH to compare validation around a native FFN")
	}
	names := os.Getenv("GOPHERLLM_COREML_FFN_UNITS")
	if names == "" {
		names = "ane"
	}
	for _, name := range strings.Split(names, ",") {
		units, ok := map[string]ComputeUnits{"ane": CPUAndNeuralEngine, "cpu": CPUOnly, "gpu": CPUAndGPU, "all": All}[name]
		if !ok {
			t.Fatalf("unknown compute units %q (expected cpu, ane, gpu or all)", name)
		}
		t.Run(name, func(t *testing.T) { compareFFNValidation(t, path, name, units) })
	}
}

func compareFFNValidation(t *testing.T, path, name string, units ComputeUnits) {
	m, err := OpenFFN(path, units)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	count := m.Info().Tokens * m.Info().Dimension
	input, before, after := make([]float32, count), make([]float32, count), make([]float32, count)
	for i := range input {
		input[i] = float32(.75*math.Sin(float64(i)*.271) + .25*math.Cos(float64(i)*.037))
	}
	scalar := func() error {
		m.mu.Lock()
		defer m.mu.Unlock()
		// Fixed, already-validated buffers let this baseline omit the shape
		// checks. This slightly favors the old path rather than the new one.
		for _, value := range input {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("non-finite benchmark input")
			}
		}
		if err := m.backend.predict(input, before); err != nil {
			return err
		}
		for _, value := range before {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return ErrFFNNonFiniteOutput
			}
		}
		return nil
	}
	run := [2]func() error{scalar, func() error { return m.Predict(input, after) }}
	for range 3 {
		for _, fn := range run {
			if err := fn(); err != nil {
				t.Fatal(err)
			}
		}
	}
	var samples [2][]float64
	var deltas []float64
	for window := 0; window < 9; window++ {
		start := time.Now()
		for pair := 0; ; pair++ {
			var elapsed [2]float64
			for order := 0; order < 2; order++ {
				which := (window + pair + order) % 2
				begin := time.Now()
				if err := run[which](); err != nil {
					t.Fatal(err)
				}
				elapsed[which] = float64(time.Since(begin)) / float64(time.Millisecond)
				samples[which] = append(samples[which], elapsed[which])
			}
			deltas = append(deltas, elapsed[0]-elapsed[1])
			if time.Since(start) >= 100*time.Millisecond {
				break
			}
		}
	}
	for i := range before {
		if math.Float32bits(before[i]) != math.Float32bits(after[i]) {
			t.Fatalf("output changed at %d: %g != %g", i, before[i], after[i])
		}
	}
	median := func(values []float64) float64 {
		sort.Float64s(values)
		return values[len(values)/2]
	}
	record := map[string]any{
		"model": filepath.Base(path), "units": name, "tokens": m.Info().Tokens, "dimension": m.Info().Dimension,
		"scalar_median_ms": median(samples[0]), "current_median_ms": median(samples[1]),
		"paired_saving_median_ms": median(deltas), "pairs": len(deltas), "windows": 9,
		"bitwise_equal": true, "measured_at": time.Now().UTC().Format(time.RFC3339),
	}
	raw, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("COREML_FFN_VALIDATION_JSON " + string(raw))
}
