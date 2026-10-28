package engine

import (
	"context"
	"fmt"
	"slices"
	"testing"
)

func speculativeFixedStandardRunner(t *testing.T) *Runner {
	t.Helper()
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	for i := range r.standard.Layers {
		clear(r.standard.Layers[i].WO.F32)
		clear(r.standard.Layers[i].W2.F32)
	}
	for i := range r.config.VocabSize {
		r.standard.TokenEmbd.F32[i*r.config.Dim] = 1
	}
	clear(r.standard.Output.F32)
	r.standard.Output.F32[3*r.config.Dim] = 1
	return r
}

func TestSpeculativeStandardLookupRuntime(t *testing.T) {
	for _, penalty := range []float32{1, 1.1} {
		for _, materialize := range []string{"", "1"} {
			t.Run(fmt.Sprintf("penalty=%.1f/materialize=%s", penalty, materialize), func(t *testing.T) {
				t.Setenv("GOPHERLLM_NO_GREEDY_ARGMAX", materialize)
				r := speculativeFixedStandardRunner(t)
				opts := speculativeRuntimeOptions()
				opts.MaxTokens, opts.Sampler.RepeatPenalty = 32, penalty
				messages := []ChatMessage{UserMessage("aaaaaaaaaaaaaaaa")}
				want, err := r.GenerateChat(messages, opts)
				if err != nil {
					t.Fatal(err)
				}
				if want.Stats.Speculation != nil {
					t.Fatal("ordinary decoding allocated speculative statistics")
				}
				wantResident := slices.Clone(r.prefixCache.tokens)
				r.ClearConversationCache()
				opts.PromptLookupTokens = 7
				got, err := r.GenerateChat(messages, opts)
				if err != nil {
					t.Fatal(err)
				}
				speculativeRequireSameResult(t, got, want)
				// A tiny verifier can finish within one Windows timer tick.
				stats := got.Stats.Speculation
				if stats == nil || stats.Blocks == 0 || stats.AcceptedTokens == 0 || stats.AcceptedTokens > stats.DraftedTokens || stats.VerificationTime < 0 {
					t.Fatalf("missing/inconsistent block statistics: %+v", stats)
				}
				if !slices.Equal(r.prefixCache.tokens, wantResident) || r.workspaceBuf.speculative.active {
					t.Fatal("lookup published an incorrect resident boundary")
				}
				t.Setenv("GOPHERLLM_NO_BLOCK_VERIFY", "1")
				r.ClearConversationCache()
				fallback, err := r.GenerateChat(messages, opts)
				if err != nil {
					t.Fatal(err)
				}
				speculativeRequireSameResult(t, fallback, want)
				if fallback.Stats.Speculation.Blocks != 0 {
					t.Fatal("block-verification opt-out was ignored")
				}
			})
		}
	}
}

func TestSpeculativeStandardCommitBoundaries(t *testing.T) {
	for _, n := range []int{2, 4, 8} {
		for keep := 0; keep <= n; keep++ {
			t.Run(fmt.Sprintf("positions=%d/keep=%d", n, keep), func(t *testing.T) {
				r := speculativeFixedStandardRunner(t)
				kd, vd, hd, kh, vh := r.cacheDims()
				cache, wantCache := NewKVCache(r.config.NLayers, kd, vd, 32), NewKVCache(r.config.NLayers, kd, vd, 32)
				buf, wantBuf := NewDecodeBuffer(r.config, hd, kh, vh), NewDecodeBuffer(r.config, hd, kh, vh)
				inputs := make([]uint32, n)
				for i := range inputs {
					inputs[i] = uint32(3 + i%4)
				}
				for _, b := range []*DecodeBuffer{buf, wantBuf} {
					copy(b.X, []float32{.25})
					copy(b.XN, []float32{.5})
				}
				ok, err := r.prepareSpeculativeBlock(context.Background(), cache, buf, inputs, 0, []uint32{inputs[0]}, 1.1)
				if err != nil || !ok {
					t.Fatalf("prepare=%v/%v", ok, err)
				}
				if err := r.finishSpeculativeBlock(context.Background(), cache, buf, keep, false, false); err != nil {
					t.Fatal(err)
				}
				for i, token := range inputs[:keep] {
					r.forwardHiddenToken(wantCache, wantBuf, token, i)
				}
				requireQwen35BatchValues(t, "boundary X", buf.X, wantBuf.X)
				requireQwen35BatchValues(t, "boundary XN", buf.XN, wantBuf.XN)
				// Continuing from this boundary must ignore every rejected row.
				var got, want []float32
				r.forwardTokenInto(cache, buf, 5, keep, &got)
				r.forwardTokenInto(wantCache, wantBuf, 5, keep, &want)
				requireQwen35BatchValues(t, "continuation logits", got, want)
				for layer := range cache.K {
					requireQwen35BatchValues(t, "visible K", cache.K[layer][:(keep+1)*kd], wantCache.K[layer][:(keep+1)*kd])
					requireQwen35BatchValues(t, "visible V", cache.V[layer][:(keep+1)*vd], wantCache.V[layer][:(keep+1)*vd])
				}
			})
		}
	}
}

func TestGreedyBatchOutputRespectsPenaltyBiasAndSoftcap(t *testing.T) {
	for _, penalty := range []float32{1, 1.1} {
		for _, softcap := range []float32{0, .7} {
			cfg := Config{Dim: 2, LogitScale: 2, FinalLogitSoftcap: softcap}
			weights := ModelWeights{Output: Weight{F32: []float32{1, -.5, -.4, 1, .3, .9}, Rows: 3, Cols: 2}, OutputBias: []float32{.1, .2, -.1}}
			hidden := [][]float32{{1, .5}, {-.1, 1}, {1, 1}}
			recent := [][]uint32{{0, 0, 0}, {1, 1}, {2, 2, 2, 2}}
			var got []uint32
			var flat []float32
			buf := &DecodeBuffer{}
			greedyBatchOutput(cfg, weights, buf, hidden, recent, penalty, &got, &flat)
			for i, row := range hidden {
				var logits []float32
				ProjectLogitsInto(cfg, weights, &DecodeBuffer{XN: row}, &logits)
				applyRepeatPenalty(logits, recent[i], penalty)
				if got[i] != argmaxFiniteToken(logits) {
					t.Fatalf("penalty=%v/softcap=%v/row=%d got=%d logits=%v", penalty, softcap, i, got[i], logits)
				}
			}
		}
	}
}
