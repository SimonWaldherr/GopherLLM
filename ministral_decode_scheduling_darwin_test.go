//go:build darwin && cgo && metal

package gopherllm

import (
	"slices"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func TestMetalDenseRowsRuntimeSchedule(t *testing.T) {
	before := CaptureRuntimeTuning()
	t.Cleanup(before.Apply)
	for _, tc := range []struct {
		name        string
		runtime     int
		vector, env string
		want        int
	}{
		{"default", 0, "", "", 4},
		{"runtime-two", 2, "1", "", 2},
		{"runtime-four", 4, "1", "", 4},
		{"runtime-eight", 8, "1", "", 8},
		{"invalid-runtime", 6, "1", "", 4},
		{"env-before-runtime", 2, "1", "8", 8},
		{"invalid-env", 2, "1", "6", 2},
		{"legacy-before-env", 2, "0", "8", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_DENSE_VECTOR", tc.vector)
			t.Setenv("GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP", tc.env)
			setMetalDenseRows(tc.runtime)
			if got := metalbackend.DenseRowsPerGroup(); got != tc.want {
				t.Fatalf("native schedule=%d, want %d", got, tc.want)
			}
		})
	}
}

func TestMetalDenseRowsPerGroupParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	before := CaptureRuntimeTuning()
	t.Cleanup(before.Apply)
	t.Setenv("GOPHERLLM_METAL_DENSE_VECTOR", "1")
	t.Setenv("GOPHERLLM_METAL_DENSE_CONCURRENT", "1")
	t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", "1")
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	oldDense := metalDenseDecodeEnabled
	metalDenseDecodeEnabled = true
	t.Cleanup(func() { metalDenseDecodeEnabled = oldDense })
	c, _, weights := metalResidualNormFixture(t, "qwen3", 256)
	// A vocabulary tail exercises SIMD groups whose row is outside the matrix.
	// The rest of this fixture contains Q4_K, Q6_K and Q8_0 projections.
	c.VocabSize = 19
	weights.TokenEmbd.F32 = append(weights.TokenEmbd.F32, make([]float32, 3*c.Dim)...)
	weights.TokenEmbd.Rows = c.VocabSize
	var raw []byte
	for row := range c.VocabSize {
		raw = append(raw, QuantizeRowQ8_0(smallWeights(c.Dim, row+71), c.Dim)...)
	}
	output := &MetalWeight{q8: metalbackend.PrepareQ8_0(raw, c.VocabSize, c.Dim, false)}
	if output.q8 == nil {
		t.Fatal(MetalError())
	}
	t.Cleanup(func() { releaseMetalWeight(output) })
	weights.Output = Weight{Raw: raw, Rows: c.VocabSize, Cols: c.Dim, Type: GGMLTypeQ8_0, Metal: output}
	run := func(t *testing.T) ([]float32, []uint32) {
		t.Helper()
		cache, buf := metalPrefillBuffer(t, c, 32)
		var logits, state []float32
		ForwardBatchInto(c, weights, cache, buf, metalPrefillTokens(17), 0, true, &logits)
		var predicted []uint32
		for _, pos := range []int{17, 18, 19, 9} {
			token := uint32(pos % 16)
			if !tryMetalDenseDecodeOutput(c, weights, cache, buf, token, pos, &logits) {
				t.Fatal("dense logits decode inactive:", MetalError())
			}
			state = append(state, logits...)
			recent := []uint32{2, 2, 7, 18, 99999}
			penalized := append([]float32(nil), logits...)
			applyRepeatPenalty(penalized, recent, 1.1)
			next, ok := tryMetalDenseGreedy(c, weights, cache, buf, token, pos, recent, 1.1)
			if !ok || next != argmaxFiniteToken(penalized) {
				t.Fatalf("greedy decode=%d/%t; want %d", next, ok, argmaxFiniteToken(penalized))
			}
			predicted = append(predicted, next)
			state = append(state, buf.X...)
			state = append(state, buf.XN...)
			for layer := range cache.K {
				state = append(state, cache.K[layer][:(pos+1)*cache.PerPosKDim]...)
				state = append(state, cache.V[layer][:(pos+1)*cache.PerPosVDim]...)
			}
		}
		return state, predicted
	}
	t.Setenv("GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP", "4")
	baseline, baselineNext := run(t)
	for _, rows := range []string{"2", "4", "8", "invalid"} {
		t.Run(rows, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP", rows)
			state, predicted := run(t)
			metalDenseMemoryExact(t, "logits/residual/hidden/KV", state, baseline)
			if !slices.Equal(predicted, baselineNext) {
				t.Fatal("greedy predictions differ from four-row scheduling")
			}
		})
	}
	t.Setenv("GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP", "")
	for _, rows := range []int{2, 8, 0} {
		setMetalDenseRows(rows)
		state, predicted := run(t)
		metalDenseMemoryExact(t, "calibrated logits/residual/hidden/KV", state, baseline)
		if !slices.Equal(predicted, baselineNext) {
			t.Fatal("calibrated schedule changed greedy predictions")
		}
	}
}
