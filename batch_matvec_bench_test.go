package gopherllm

import (
	"fmt"
	"math/rand"
	"testing"
)

// benchmarkBatchMatvecMode pins the package-wide Q8 switch outside the timed
// loop. matvecBatchWithQ8 receives the same explicit choice, but the Q8
// kernel also checks the switch internally; pinning both makes benchmark output
// independent of GOPHERLLM_Q8_ACTIVATIONS.
func benchmarkBatchMatvecMode(b *testing.B, w Weight, xs, outs [][]float32, useQ8 bool) {
	b.ReportAllocs()
	withQ8Activations(useQ8, func() {
		for b.Loop() {
			matvecBatchWithQ8(w, xs, outs, useQ8)
		}
	})
}

// BenchmarkMatvecBatchQ4KByPromptSize compares the two weight-stationary
// prefill implementations at Ministral's 3072-wide projection shape. The Q8
// form avoids expanding weights but re-runs the quantized dot per token; the
// float form expands each weight row once and then applies it to every token.
// Keep this benchmark when changing either dispatch threshold.
func BenchmarkMatvecBatchQ4KByPromptSize(b *testing.B) {
	rng := rand.New(rand.NewSource(73))
	const rows, cols = 3072, 3072
	data := make([]byte, 0, rows*(cols/256)*144)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	w := Weight{Raw: data, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols}

	for _, promptTokens := range []int{1, 4, 16, 32, 64, 96, 128} {
		xs := make([][]float32, promptTokens)
		outs := make([][]float32, promptTokens)
		for p := range xs {
			xs[p] = randomVec(rng, cols)
			outs[p] = make([]float32, rows)
		}
		for _, tc := range []struct {
			name  string
			useQ8 bool
		}{
			{"q8", true},
			{"dequant_once", false},
		} {
			b.Run(fmt.Sprintf("P%d/%s", promptTokens, tc.name), func(b *testing.B) {
				benchmarkBatchMatvecMode(b, w, xs, outs, tc.useQ8)
			})
		}
	}
}

// BenchmarkMatvecBatchQ4KMinistralFFNLongPrompt covers the dominant
// 3072-to-9216 gate/up shape once the standard prefill chunk reaches its long
// prompt size. It keeps the adaptive dispatch tied to the model shape that
// matters, rather than only a square attention projection.
func BenchmarkMatvecBatchQ4KMinistralFFNLongPrompt(b *testing.B) {
	rng := rand.New(rand.NewSource(74))
	const rows, cols = 9216, 3072
	data := make([]byte, 0, rows*(cols/256)*144)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	w := Weight{Raw: data, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols}

	for _, promptTokens := range []int{96, 128} {
		xs := make([][]float32, promptTokens)
		outs := make([][]float32, promptTokens)
		for p := range xs {
			xs[p] = randomVec(rng, cols)
			outs[p] = make([]float32, rows)
		}
		for _, tc := range []struct {
			name  string
			useQ8 bool
		}{
			{"q8", true},
			{"dequant_once", false},
		} {
			b.Run(fmt.Sprintf("P%d/%s", promptTokens, tc.name), func(b *testing.B) {
				benchmarkBatchMatvecMode(b, w, xs, outs, tc.useQ8)
			})
		}
	}
}

// BenchmarkMatvecBatchQ4KMinistralDownLongPrompt covers the reverse FFN
// projection (9216 to 3072). Together with the gate/up benchmark this keeps a
// threshold change from accidentally optimizing only one half of SwiGLU.
func BenchmarkMatvecBatchQ4KMinistralDownLongPrompt(b *testing.B) {
	rng := rand.New(rand.NewSource(75))
	const rows, cols, promptTokens = 3072, 9216, 128
	data := make([]byte, 0, rows*(cols/256)*144)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	w := Weight{Raw: data, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols}
	xs := make([][]float32, promptTokens)
	outs := make([][]float32, promptTokens)
	for p := range xs {
		xs[p] = randomVec(rng, cols)
		outs[p] = make([]float32, rows)
	}
	for _, tc := range []struct {
		name  string
		useQ8 bool
	}{
		{"q8", true},
		{"dequant_once", false},
	} {
		b.Run(tc.name, func(b *testing.B) {
			benchmarkBatchMatvecMode(b, w, xs, outs, tc.useQ8)
		})
	}
}
