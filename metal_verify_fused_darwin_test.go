//go:build darwin && cgo && metal

package gopherllm

import (
	"context"
	"fmt"
	"math/rand"
	"slices"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func fusedVerifierFixture(t testing.TB, q6 bool) (Config, ModelWeights, ModelWeights) {
	c, cpu, gpu := metalPrefillFixture(t, "ministral")
	if q6 {
		rng := rand.New(rand.NewSource(441983))
		raw := make([]byte, 0, c.VocabSize*210)
		for range c.VocabSize {
			row := randomQ6KRow(rng, c.Dim)
			for i := 0; i < len(row); i += 210 {
				row[i+209] = 0x04
			}
			raw = append(raw, row...)
		}
		cpu.Output = Weight{Raw: raw, Rows: c.VocabSize, Cols: c.Dim, Type: GGMLTypeQ6_K}
		gpu.Output = cpu.Output
		gpu.Output.Metal = &MetalWeight{q6: metalbackend.PrepareQ6K(raw, c.VocabSize, c.Dim, false)}
		t.Cleanup(func() { releaseMetalWeight(gpu.Output.Metal) })
	}
	return c, cpu, gpu
}

func TestMetalFusedVerifyHeadsAndHistories(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	for _, vector := range []string{"0", "1"} {
		for _, shared := range []string{"0", "1"} {
			for _, q6 := range []bool{false, true} {
				for _, n := range []int{2, 4, 8} {
					t.Run(fmt.Sprintf("vector%s/shared%s/q6_%t/P%d", vector, shared, q6, n), func(t *testing.T) {
						t.Setenv("GOPHERLLM_METAL_VERIFY_VECTOR", vector)
						t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
						c, cpu, gpu := fusedVerifierFixture(t, q6)
						c.SlidingWindow, c.SWAPattern = 5, []bool{true, false}
						ck, cb := metalPrefillBuffer(t, c, 64)
						gk, gb := metalPrefillBuffer(t, c, 64)
						for _, start := range []int{0, n, 1} {
							tokens := metalPrefillTokens(n)
							want, hidden := make([]uint32, n), make([][]float32, n)
							recent := make([][]uint32, n)
							for i, token := range tokens {
								var logits []float32
								ForwardInto(c, cpu, ck, cb, token, start+i, &logits)
								hidden[i] = slices.Clone(cb.XN)
								winner := argmaxFiniteToken(logits)
								recent[i] = []uint32{winner, winner, uint32(i), 999999}
								if i%3 == 1 {
									recent[i] = nil
								}
								applyRepeatPenalty(logits, recent[i], 1.1)
								want[i] = argmaxFiniteToken(logits)
							}
							var got []uint32
							states, handled, err := tryMetalDenseVerify(context.Background(), c, gpu, gk, gb, tokens, start, recent, 1.1, &got)
							if err != nil || !handled {
								t.Fatalf("fused verifier handled=%v err=%v: %s", handled, err, MetalError())
							}
							if !slices.Equal(got, want) {
								t.Fatalf("tokens=%v want=%v", got, want)
							}
							for i := range states {
								assertMetalFiniteClose(t, states[i], hidden[i])
							}
							checkMetalPrefillKV(t, gk, ck, start+n)
							// Overwriting rejected suffix rows must also work on the
							// next scalar GPU step using this decoder's bound LM head.
							var actual, expected []float32
							ForwardInto(c, cpu, ck, cb, 7, start+1, &expected)
							ForwardInto(c, gpu, gk, gb, 7, start+1, &actual)
							assertMetalFiniteClose(t, actual, expected)
						}
					})
				}
			}
		}
	}
}

func TestMetalFusedVerifyCancellationAndFallback(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	for _, shared := range []string{"0", "1"} {
		t.Run(shared, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
			c, _, gpu := fusedVerifierFixture(t, true)
			k, b := metalPrefillBuffer(t, c, 64)
			tokens, recent := metalPrefillTokens(4), make([][]uint32, 4)
			var got []uint32
			if _, ok, err := tryMetalDenseVerify(context.Background(), c, gpu, k, b, tokens, 0, recent, 1.1, &got); !ok || err != nil {
				t.Fatalf("warmup=%v/%v", ok, err)
			}
			beforeK, beforeV := make([][]float32, c.NLayers), make([][]float32, c.NLayers)
			for l := range k.K {
				beforeK[l], beforeV[l] = slices.Clone(k.K[l]), slices.Clone(k.V[l])
			}
			residual, output := make([]float32, 4*c.Dim), make([]float32, 4*c.Dim)
			for i := range residual {
				residual[i], output[i] = 123, -456
			}
			predictions := []uint32{99, 98, 97, 96}
			canceled := &prefillCancelContext{Context: context.Background()}
			ok, err := b.metalDense.decoder.VerifyBatch(canceled, b.batch.XFlat, b.batch.RopeSinFlat, b.batch.RopeCosFlat, b.batch.RopeTemperature, 4, 4, false, true, k.K, k.V, residual, output, recent, 1.1, predictions)
			if ok || err != context.Canceled || !slices.Equal(predictions, []uint32{99, 98, 97, 96}) {
				t.Fatalf("canceled publication=%v/%v/%v", ok, err, predictions)
			}
			for i := range residual {
				if residual[i] != 123 || output[i] != -456 {
					t.Fatal("canceled verifier published hidden states")
				}
			}
			for _, mutate := range []func(*Config, *ModelWeights){
				func(c *Config, _ *ModelWeights) { c.FinalLogitSoftcap = 1 },
				func(_ *Config, w *ModelWeights) { w.OutputBias = make([]float32, c.VocabSize) },
				func(_ *Config, w *ModelWeights) { w.Output.Metal = nil },
			} {
				cfg, weights := c, gpu
				mutate(&cfg, &weights)
				if _, ok, err := tryMetalDenseVerify(context.Background(), cfg, weights, k, b, tokens, 4, recent, 1.1, &got); ok || err != nil {
					t.Fatalf("unsupported verifier=%v/%v", ok, err)
				}
			}
			for l := range k.K {
				if !slices.Equal(k.K[l], beforeK[l]) || !slices.Equal(k.V[l], beforeV[l]) {
					t.Fatal("declined/canceled verifier published KV")
				}
			}
		})
	}
}

func TestMetalFusedVerifyRuntimeMaterializedOracle(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ8_0} {
		t.Run(fmt.Sprint(typ), func(t *testing.T) {
			r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = r.Close() })
			c, _, gpu := metalPrefillFixture(t, "llama")
			c.MaxSeqLen = 128
			rowBytes := 144
			if typ == GGMLTypeQ8_0 {
				rowBytes = c.Dim / 32 * 34
			}
			// A tied zero head forces token0 while the nonzero transformer
			// still executes every attention/FFN layer on the GPU.
			gpu.Output = Weight{Raw: make([]byte, c.VocabSize*rowBytes), Rows: c.VocabSize, Cols: c.Dim, Type: typ, Metal: &MetalWeight{}}
			if typ == GGMLTypeQ4_K {
				gpu.Output.Metal.q4 = metalbackend.PrepareQ4K(gpu.Output.Raw, c.VocabSize, c.Dim, false)
			} else {
				gpu.Output.Metal.q8 = metalbackend.PrepareQ8_0(gpu.Output.Raw, c.VocabSize, c.Dim, false)
			}
			r.config, r.standard = c, gpu
			r.tok.Vocab[0], r.tok.TokenToID["a"] = "a", 0
			messages := []ChatMessage{UserMessage("aaaaaaaaaaaaaaaa")}
			opts := speculativeRuntimeOptions()
			opts.MaxTokens, opts.Sampler.RepeatPenalty = 24, 1.1
			t.Setenv("GOPHERLLM_NO_GREEDY_ARGMAX", "1")
			want, err := r.GenerateChat(messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			resident := slices.Clone(r.prefixCache.tokens)
			for _, materialize := range []string{"", "1"} {
				t.Setenv("GOPHERLLM_NO_GREEDY_ARGMAX", materialize)
				if materialize == "1" {
					k, b := metalPrefillBuffer(t, c, 8)
					var predictions []uint32
					if _, ok, err := tryMetalDenseVerify(context.Background(), c, gpu, k, b, []uint32{0, 0}, 0, make([][]uint32, 2), 1.1, &predictions); ok || err != nil {
						t.Fatalf("materialized oracle enabled fused argmax: %v/%v", ok, err)
					}
				}
				r.ClearConversationCache()
				opts.PromptLookupTokens = 7
				got, err := r.GenerateChat(messages, opts)
				if err != nil {
					t.Fatal(err)
				}
				speculativeRequireSameResult(t, got, want)
				if got.Stats.Speculation == nil || got.Stats.Speculation.Blocks == 0 || got.Stats.Speculation.AcceptedTokens == 0 || !slices.Equal(resident, r.prefixCache.tokens) {
					t.Fatalf("lookup did not preserve the verified runtime boundary: %+v", got.Stats.Speculation)
				}
				if r.workspaceBuf.metalDense == nil {
					t.Fatal("GPU decoder inactive")
				}
			}
		})
	}
}
