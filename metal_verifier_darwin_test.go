//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"fmt"
	"slices"
	"testing"
)

func TestMetalVerifierStatesParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	for _, shared := range []string{"0", "1"} {
		for _, arch := range []string{"ministral", "qwen3", "llama"} {
			for _, n := range []int{2, 4, 8} {
				t.Run(fmt.Sprintf("shared_%s/%s/P%d", shared, arch, n), func(t *testing.T) {
					t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
					c, cpu, gpu := metalPrefillFixture(t, arch)
					c.SlidingWindow, c.SWAPattern = 5, []bool{true, false}
					ck, cb := metalPrefillBuffer(t, c, 64)
					gk, gb := metalPrefillBuffer(t, c, 64)
					// Ordinary short prefill must retain its existing dispatch policy.
					tokens := metalPrefillTokens(n)
					ForwardBatchInto(c, gpu, gk, gb, tokens, 0, false, nil)
					if gb.metalDense != nil && gb.metalDense.decoder.PrefillScratchBytes() != 0 {
						t.Fatal("normal short prefill used the verifier dispatch")
					}
					for _, start := range []int{0, n, 1} {
						want := make([][]float32, n)
						for i, token := range tokens {
							ForwardBodyInto(c, cpu, ck, cb, token, start+i)
							want[i] = slices.Clone(cb.XN)
						}
						got, err := forwardBatchStatesIntoContext(context.Background(), c, gpu, gk, gb, tokens, start)
						if err != nil {
							t.Fatal(err)
						}
						if gb.metalDense == nil || gb.metalDense.decoder.PrefillScratchBytes() == 0 {
							t.Fatal("short whole GPU verifier did not execute")
						}
						if shared == "1" && gb.metalDense.sharedK[0] == nil {
							t.Fatal("shared KV inactive")
						}
						if len(got) != n {
							t.Fatalf("got %d states, want %d", len(got), n)
						}
						for i := range want {
							assertMetalFiniteClose(t, got[i], want[i])
						}
						assertMetalFiniteClose(t, gb.XN, want[n-1])
						checkMetalPrefillKV(t, gk, ck, start+n)
						// Rejection can leave unused suffix rows; a scalar overwrite
						// followed by another short block must consume the new prefix.
						var expected, actual []float32
						ForwardInto(c, cpu, ck, cb, 7, start+1, &expected)
						ForwardInto(c, gpu, gk, gb, 7, start+1, &actual)
						assertMetalFiniteClose(t, actual, expected)
					}
				})
			}
		}
	}
}

func TestMetalVerifierRestoredPrefixAndCancellation(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	for _, shared := range []string{"0", "1"} {
		t.Run(shared, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
			c, cpu, gpu := metalPrefillFixture(t, "ministral")
			ck, cb := metalPrefillBuffer(t, c, 64)
			gk, gb := metalPrefillBuffer(t, c, 64)
			const start, n = 19, 4
			for _, branch := range []int{0, 1, 0} {
				for l := range ck.K {
					for i := range start * 128 {
						v := float32((i+branch*13+l*5)%23-11) * .03
						ck.K[l][i], gk.K[l][i] = v, v
						ck.V[l][i], gk.V[l][i] = v*.7, v*.7
					}
				}
				want, err := forwardBatchStatesIntoContext(context.Background(), c, cpu, ck, cb, metalPrefillTokens(n), start)
				if err != nil {
					t.Fatal(err)
				}
				got, err := forwardBatchStatesIntoContext(context.Background(), c, gpu, gk, gb, metalPrefillTokens(n), start)
				if err != nil {
					t.Fatal(err)
				}
				for i := range want {
					assertMetalFiniteClose(t, got[i], want[i])
				}
				checkMetalPrefillKV(t, gk, ck, start+n)
			}
			beforeK, beforeV := make([][]float32, c.NLayers), make([][]float32, c.NLayers)
			for l := range gk.K {
				beforeK[l], beforeV[l] = slices.Clone(gk.K[l]), slices.Clone(gk.V[l])
			}
			residual, output := make([]float32, n*c.Dim), make([]float32, n*c.Dim)
			canceled := &prefillCancelContext{Context: context.Background()}
			ok, err := gb.metalDense.decoder.PrefillBatch(canceled, gb.batch.XFlat, gb.batch.RopeSinFlat, gb.batch.RopeCosFlat, gb.batch.RopeTemperature, start, n, false, true, gk.K, gk.V, residual, output)
			if ok || err != context.Canceled {
				t.Fatalf("canceled short GPU block = %v, %v", ok, err)
			}
			for l := range gk.K {
				if !slices.Equal(gk.K[l], beforeK[l]) || !slices.Equal(gk.V[l], beforeV[l]) {
					t.Fatal("canceled short verifier published KV")
				}
			}
		})
	}
}

func BenchmarkMetalVerifierStates(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	c, _, gpu := metalPrefillFixture(b, "ministral")
	for _, n := range []int{2, 4, 8} {
		for _, batch := range []bool{false, true} {
			b.Run(fmt.Sprintf("P%d/batch_%v", n, batch), func(b *testing.B) {
				b.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
				k, buf := metalPrefillBuffer(b, c, 128)
				tokens := metalPrefillTokens(n)
				run := func() {
					if batch {
						if _, err := forwardBatchStatesIntoContext(context.Background(), c, gpu, k, buf, tokens, 64); err != nil {
							b.Fatal(err)
						}
					} else {
						for i, token := range tokens {
							ForwardBodyInto(c, gpu, k, buf, token, 64+i)
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
