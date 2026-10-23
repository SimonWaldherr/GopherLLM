//go:build darwin && cgo && coreml

package coreml

import (
	"math"
	"os"
	"testing"
)

// The opt-in native probe is hidden [2,3] -> output [2,3] float32, computing
// output = hidden*2+1. It checks both reused native I/O and aliased Go slices.
func TestFFNConfiguredProbe(t *testing.T) {
	path := os.Getenv("GOPHERLLM_COREML_FFN_PROBE")
	if path == "" {
		t.Skip("set GOPHERLLM_COREML_FFN_PROBE to a compiled hidden*2+1 [2,3] model")
	}
	for _, units := range []ComputeUnits{CPUOnly, CPUAndNeuralEngine} {
		m, err := OpenFFN(path, units)
		if err != nil {
			t.Fatal(err)
		}
		if m.Info() != (FFNInfo{Tokens: 2, Dimension: 3, Manifest: os.Getenv("GOPHERLLM_COREML_FFN_PROBE_MANIFEST")}) {
			_ = m.Close()
			t.Fatalf("unexpected probe shape: %+v", m.Info())
		}
		for iteration := range 3 {
			values := []float32{-2, -1, 0, 1, 2, float32(iteration)}
			want := append([]float32(nil), values...)
			for i := range want {
				want[i] = want[i]*2 + 1
			}
			if err := m.Predict(values, values); err != nil {
				_ = m.Close()
				t.Fatal(err)
			}
			for i := range want {
				if math.Abs(float64(values[i]-want[i])) > 1e-6 {
					_ = m.Close()
					t.Fatalf("iteration %d, output[%d]=%g, want %g", iteration, i, values[i], want[i])
				}
			}
		}
		if err := m.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFFNNativeOpenRejectsConfiguration(t *testing.T) {
	if m, err := OpenFFN("missing.mlmodelc", All+1); m != nil || err == nil {
		t.Fatalf("accepted unknown compute units: model=%v err=%v", m, err)
	}
	if m, err := OpenFFN("missing.mlmodelc", CPUOnly); m != nil || err == nil {
		t.Fatalf("accepted missing compiled path: model=%v err=%v", m, err)
	}
}
