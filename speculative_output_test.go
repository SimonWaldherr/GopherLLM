package gopherllm

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

func TestSpeculativeCPUOutputMatchesSerial(t *testing.T) {
	withQ8Activations(true, func() {
		if !useQ8Activations.Load() {
			t.Skip("CPU Q8 activation kernels unavailable on this host")
		}
		for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
			for _, count := range []int{4, 5, 8} {
				t.Run(fmt.Sprintf("%v/P%d", typ, count), func(t *testing.T) {
					const rows, cols = 37, 512
					rng := rand.New(rand.NewSource(825))
					w := Weight{Type: typ, Rows: rows, Cols: cols}
					for range rows {
						switch typ {
						case GGMLTypeQ4_K:
							w.Raw = append(w.Raw, randomQ4KRow(rng, cols)...)
						case GGMLTypeQ5_K:
							w.Raw = append(w.Raw, randomQ5KRow(rng, cols)...)
						case GGMLTypeQ6_K:
							w.Raw = append(w.Raw, randomQ6KRow(rng, cols)...)
						case GGMLTypeQ8_0:
							w.Raw = append(w.Raw, randomQ8_0Row(rng, cols)...)
						}
					}
					hidden, recent := make([][]float32, count), make([][]uint32, count)
					for i := range hidden {
						hidden[i] = randomVec(rng, cols)
						recent[i] = []uint32{uint32(i), uint32(i), uint32(rows - 1), rows + 12}
					}
					for _, penalty := range []float32{1, 1.1, .5} {
						for _, softcap := range []float32{0, .7} {
							cfg := Config{Dim: cols, VocabSize: rows, LogitScale: 2, FinalLogitSoftcap: softcap}
							weights := ModelWeights{Output: w, OutputBias: randomVec(rng, rows)}
							buf := &DecodeBuffer{}
							got := make([]uint32, count)
							if !greedyCPUQuantizedBatchOutput(cfg, weights, buf, hidden, recent, penalty, got) {
								t.Fatal("valid CPU batch declined")
							}
							for i := range hidden {
								var logits []float32
								ProjectLogitsInto(cfg, weights, &DecodeBuffer{XN: hidden[i]}, &logits)
								applyRepeatPenalty(logits, recent[i], penalty)
								if !slices.Equal(buf.speculative.outputRows[i], logits) || got[i] != argmaxFiniteToken(logits) {
									t.Fatalf("position %d: batch logits/token differ from serial (penalty %g, softcap %g)", i, penalty, softcap)
								}
							}
						}
					}
					// The established serial greedy path chooses the raw winner
					// before positive scaling/softcap. Preserve that choice even
					// when a rounded, saturated softcap would create new ties.
					cfg := Config{Dim: cols, LogitScale: 2, FinalLogitSoftcap: 1e-6}
					got := make([]uint32, count)
					if !greedyCPUQuantizedBatchOutput(cfg, ModelWeights{Output: w}, &DecodeBuffer{}, hidden, recent, 1, got) {
						t.Fatal("unbiased CPU batch declined")
					}
					for i := range hidden {
						want, ok := w.ArgmaxMatvec(hidden[i])
						if !ok || got[i] != want {
							t.Fatalf("raw greedy position %d: got %d want %d", i, got[i], want)
						}
					}
				})
			}
		}
	})
}

func TestSpeculativeCPUOutputDeclinesUnboundedVocabulary(t *testing.T) {
	withQ8Activations(true, func() {
		buf := &DecodeBuffer{}
		w := ModelWeights{Output: Weight{Type: GGMLTypeQ6_K, Rows: 1 << 30, Cols: 256}}
		if greedyCPUQuantizedBatchOutput(Config{}, w, buf, make([][]float32, 8), nil, 1, nil) {
			t.Fatal("accepted unbounded vocabulary")
		}
		if len(buf.speculative.outputLogits) != 0 {
			t.Fatal("allocated logits before checking memory bound")
		}
	})
}
