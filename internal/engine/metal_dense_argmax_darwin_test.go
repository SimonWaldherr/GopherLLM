//go:build darwin && cgo && metal

package engine

import (
	"fmt"
	"math"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func TestMetalDenseParallelArgmax(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	for _, rows := range []int{16383, 16384, 16385, 131073} {
		for _, kind := range []string{"boundary-ties", "negative-boundary-ties", "zero", "nonfinite"} {
			t.Run(fmt.Sprintf("%d/%s", rows, kind), func(t *testing.T) {
				c, _, w := metalResidualNormFixture(t, "qwen3", 256)
				cache, buf := metalPrefillBuffer(t, c, 32)
				if !tryMetalDenseDecode(c, w, cache, buf, 3, 0) {
					t.Fatal("dense decode inactive:", MetalError())
				}
				// Match the hidden vector so the selected boundary rows have
				// positive, bit-identical logits and a predictable winning index.
				row := QuantizeRowQ8_0(buf.XN, c.Dim)
				raw := make([]byte, rows*len(row))
				if kind == "negative-boundary-ties" {
					negative := make([]float32, c.Dim)
					for i := range negative {
						negative[i] = -buf.XN[i]
					}
					row = QuantizeRowQ8_0(negative, c.Dim)
					for i := range rows {
						copy(raw[i*len(row):], row)
					}
					for i := range negative {
						negative[i] *= .5
					}
					row = QuantizeRowQ8_0(negative, c.Dim)
				}
				if kind == "boundary-ties" || kind == "negative-boundary-ties" {
					for _, i := range []int{4095, 4096, 8192, rows - 1} {
						copy(raw[i*len(row):], row)
					}
				} else if kind == "nonfinite" {
					for i := 0; i < len(raw); i += 34 {
						// Alternate half NaN and infinity; neither is a valid winner.
						raw[i+1] = 0x7c
						raw[i] = byte((i / 34) % 2)
					}
				}
				output := &MetalWeight{q8: metalbackend.PrepareQ8_0(raw, rows, c.Dim, false)}
				if output.q8 == nil {
					t.Fatal(MetalError())
				}
				t.Cleanup(func() { releaseMetalWeight(output) })
				c.VocabSize = rows
				w.Output = Weight{Raw: raw, Rows: rows, Cols: c.Dim, Type: GGMLTypeQ8_0, Metal: output}
				repeated := make([]uint32, 64)
				for i := range repeated {
					repeated[i] = 4095
				}
				for _, recent := range [][]uint32{nil, {4095, 4095, 4096, 8192, uint32(rows + 7)}, repeated} {
					for _, penalty := range []float32{1, 1.1, .5} {
						for _, concurrent := range []string{"0", "1"} {
							t.Setenv("GOPHERLLM_METAL_DENSE_CONCURRENT", concurrent)
							var want []float32
							var wantNext uint32
							for _, parallel := range []string{"0", "1"} {
								t.Setenv("GOPHERLLM_METAL_DENSE_PARALLEL_ARGMAX", parallel)
								var logits []float32
								var next uint32
								if !tryMetalDenseStep(c, w, cache, buf, 3, 0, &logits, recent, penalty, &next) {
									t.Fatal("dense selection inactive:", MetalError())
								}
								if next != argmaxFiniteToken(logits) {
									t.Fatal("selected token differs from the penalized logits")
								}
								if parallel == "0" {
									want, wantNext = logits, next
									continue
								}
								if next != wantNext {
									t.Fatalf("parallel winner=%d, want %d", next, wantNext)
								}
								for i := range logits {
									if math.Float32bits(logits[i]) != math.Float32bits(want[i]) {
										t.Fatalf("penalized logit %d changed: %g vs %g", i, logits[i], want[i])
									}
								}
							}
						}
					}
				}
			})
		}
	}
}
