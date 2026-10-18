//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"testing"
)

func loadMetalVerifierModel(t testing.TB) *Runner {
	t.Helper()
	path := os.Getenv("GOPHERLLM_VALIDATE_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_VALIDATE_MODEL for checkpoint validation")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, LogWriter: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.Close() })
	c := r.config
	if r.kind != loadedStandard || !metalDenseEligible(c, r.standard, NewKVCache(c.NLayers, c.KVDim, c.NKVHeads*c.ValueDim, 128)) {
		t.Fatal("standard dense Metal model required")
	}
	return r
}

func TestMetalVerifierModelStates(t *testing.T) {
	r := loadMetalVerifierModel(t)
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	c := r.config
	for _, n := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("P%d", n), func(t *testing.T) {
			ck, cb := metalPrefillBuffer(t, c, 128)
			bk, bb := metalPrefillBuffer(t, c, 128)
			prefix := metalPrefillTokens(17)
			ForwardBatchInto(c, r.standard, ck, cb, prefix, 0, false, nil)
			ForwardBatchInto(c, r.standard, bk, bb, prefix, 0, false, nil)
			tokens := metalPrefillTokens(n)
			want := make([][]float32, n)
			for i, token := range tokens {
				ForwardBodyInto(c, r.standard, ck, cb, token, len(prefix)+i)
				want[i] = slices.Clone(cb.XN)
			}
			got, err := forwardBatchStatesIntoContext(context.Background(), c, r.standard, bk, bb, tokens, len(prefix))
			if err != nil {
				t.Fatal(err)
			}
			for i := range got {
				var error2, norm2 float64
				for j, v := range want[i] {
					if !finite32(got[i][j]) {
						t.Fatal("nonfinite hidden state")
					}
					d := float64(got[i][j] - v)
					error2 += d * d
					norm2 += float64(v) * float64(v)
				}
				relative := math.Sqrt(error2 / math.Max(norm2, 1e-30))
				t.Logf("token %d relative hidden L2=%g", i, relative)
				if relative > .002 {
					t.Fatal("hidden states differ")
				}
				var expected, actual []float32
				r.standard.Output.MatvecInto(want[i], &expected)
				r.standard.Output.MatvecInto(got[i], &actual)
				if argmaxFiniteToken(actual) != argmaxFiniteToken(expected) {
					t.Fatal("greedy prediction differs")
				}
			}
			// Replay after a rejected suffix must overwrite the speculative rows.
			var expected, actual []float32
			ForwardInto(c, r.standard, ck, cb, 7, len(prefix)+1, &expected)
			ForwardInto(c, r.standard, bk, bb, 7, len(prefix)+1, &actual)
			if argmaxFiniteToken(actual) != argmaxFiniteToken(expected) {
				t.Fatal("post-rejection prediction differs")
			}
		})
	}
}

// BenchmarkMetalVerifierModelBody measures the transformer body without a
// vocabulary projection or drafter. End-to-end speculative speed also depends
// on acceptance rate, output projection, and replay of rejected suffixes.
func BenchmarkMetalVerifierModelBody(b *testing.B) {
	r := loadMetalVerifierModel(b)
	c := r.config
	for _, n := range []int{2, 4, 8} {
		for _, batch := range []bool{false, true} {
			b.Run(fmt.Sprintf("P%d/batch_%v", n, batch), func(b *testing.B) {
				b.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
				k, buf := metalPrefillBuffer(b, c, 128)
				ForwardBatchInto(c, r.standard, k, buf, metalPrefillTokens(32), 0, false, nil)
				tokens := metalPrefillTokens(n)
				run := func() {
					if batch {
						if _, err := forwardBatchStatesIntoContext(context.Background(), c, r.standard, k, buf, tokens, 32); err != nil {
							b.Fatal(err)
						}
					} else {
						for i, token := range tokens {
							ForwardBodyInto(c, r.standard, k, buf, token, 32+i)
						}
					}
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run()
				}
			})
		}
	}
}

func BenchmarkMetalVerifierModelOutputPenalized(b *testing.B) {
	r := loadMetalVerifierModel(b)
	w := r.standard.Output
	if w.Type != GGMLTypeQ6_K || w.Metal == nil {
		b.Skip("Q6_K Metal output required")
	}
	for _, n := range []int{2, 4, 8} {
		xs := make([]float32, n*w.Cols)
		histories := make([][]uint32, n)
		tokens := make([]uint32, n)
		for i := range xs {
			xs[i] = float32(i%71-35) / 29
		}
		for i := range histories {
			histories[i] = make([]uint32, 64)
			for j := range histories[i] {
				histories[i][j] = uint32((i*13 + j*7) % w.Rows)
			}
		}
		for _, batch := range []bool{false, true} {
			b.Run(fmt.Sprintf("P%d/batch_%v", n, batch), func(b *testing.B) {
				run := func() {
					if batch {
						if !argmaxMetalQ6KBatchPenalized(w.Metal, xs, tokens, histories, 1.1) {
							b.Fatal(MetalError())
						}
					} else {
						for i := range n {
							if _, ok := argmaxMetalQ6KPenalized(w.Metal, xs[i*w.Cols:(i+1)*w.Cols], histories[i], 1.1); !ok {
								b.Fatal(MetalError())
							}
						}
					}
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run()
				}
			})
		}
	}
}
