package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
)

// Keep nonzero convolution, DeltaNet and attention computations, but neutralize
// their residual outputs so the first residual component stays positive. A
// one-row head then forces the target token; the all-zero MTP head forces zero.
// This exercises real state updates with guaranteed accepted/rejected drafts.
func speculativeFixedQwenRunner(t *testing.T, target uint32) *Runner {
	t.Helper()
	r := qwen35DenseRuntimeRunner(t, true)
	r.config.MaxSeqLen = 128
	r.tok.Vocab[0], r.tok.Vocab[3] = "a", "b"
	r.tok.TokenToID["a"], r.tok.TokenToID["b"] = 0, 3
	for token := range r.config.VocabSize {
		r.qwen35.TokenEmbd.F32[token*r.config.Dim] = 1
	}
	for i := range r.qwen35.OutputNorm {
		r.qwen35.OutputNorm[i] = 1
	}
	for i := range r.qwen35.Layers {
		layer := &r.qwen35.Layers[i]
		clear(layer.FFN.Down.F32)
		if layer.Kind == qwen35Attention {
			clear(layer.Attention.O.F32)
		} else {
			clear(layer.DeltaNet.Out.F32)
		}
	}
	head := make([]float32, r.config.VocabSize*r.config.Dim)
	head[int(target)*r.config.Dim] = 1
	r.qwen35.Output = Weight{F32: head, Rows: r.config.VocabSize, Cols: r.config.Dim}
	r.qwen35.MTP.Output = Weight{F32: make([]float32, len(head)), Rows: r.config.VocabSize, Cols: r.config.Dim}
	return r
}

func speculativeRuntimeOptions() GenerationOptions {
	o := DefaultGenerationOptions()
	o.SystemPrompt, o.MaxTokens, o.Seed = "", 12, 19
	o.Sampler.Temperature, o.Sampler.TopK, o.Sampler.RepeatPenalty = 0, 1, 1
	return o
}

func speculativeRequireSameResult(t *testing.T, got, want GenerationResult) {
	t.Helper()
	if got.Text != want.Text || got.ReasoningText != want.ReasoningText || got.FinishReason != want.FinishReason || got.Stats.GeneratedTokens != want.Stats.GeneratedTokens || got.Stats.PromptTokens != want.Stats.PromptTokens {
		t.Fatalf("speculative result differs: got=%+v want=%+v", got, want)
	}
}

func speculativeRequireSameVisibleState(t *testing.T, got, want *Runner) {
	t.Helper()
	if !slices.Equal(got.prefixCache.tokens, want.prefixCache.tokens) {
		t.Fatalf("resident token boundary differs: got=%v want=%v", got.prefixCache.tokens, want.prefixCache.tokens)
	}
	gc, wc := got.prefixCache.cache, want.prefixCache.cache
	if gc == nil || wc == nil || got.prefixCache.qwen35 == nil || want.prefixCache.qwen35 == nil {
		t.Fatal("successful prefill did not retain the recurrent prefix")
	}
	requireQwen35BatchValues(t, "retained conv", got.prefixCache.qwen35.Conv, want.prefixCache.qwen35.Conv)
	requireQwen35BatchValues(t, "retained recurrence", got.prefixCache.qwen35.State, want.prefixCache.qwen35.State)
	speculativeRequireSameCacheRows(t, gc, wc, len(got.prefixCache.tokens))
	if got.prefixCache.qwen35MTP != nil {
		requireQwen35BatchValues(t, "accepted pending target hidden", got.prefixCache.qwen35MTP, got.workspaceBuf.XN)
	}
}

func speculativeRequireSameCacheRows(t *testing.T, gc, wc *KVCache, n int) {
	t.Helper()
	if gc.kvFormat() != wc.kvFormat() || gc.layerCount() != wc.layerCount() {
		t.Fatal("KV cache layouts differ")
	}
	for layer := range gc.layerCount() {
		k, v := n*gc.PerPosKDim, n*gc.PerPosVDim
		switch gc.kvFormat() {
		case kvF32:
			requireQwen35BatchValues(t, "visible K", gc.K[layer][:k], wc.K[layer][:k])
			requireQwen35BatchValues(t, "visible V", gc.V[layer][:v], wc.V[layer][:v])
		case kvF16:
			requireQwen35BatchF16(t, "visible K", gc.K16[layer][:k], wc.K16[layer][:k])
			requireQwen35BatchF16(t, "visible V", gc.V16[layer][:v], wc.V16[layer][:v])
		default:
			t.Fatal("tiny test's sub-block heads unexpectedly use I8 KV")
		}
	}
}

// Cancellation may retain an earlier valid checkpoint than token-at-a-time
// decode. Verify its own advertised boundary rather than requiring identical
// cache-hit lengths or replaying expensive target work during cancellation.
func speculativeRequireConsistentBoundary(t *testing.T, r *Runner) {
	t.Helper()
	if r.prefixCache.cache == nil || r.prefixCache.qwen35 == nil || len(r.prefixCache.tokens) == 0 {
		t.Fatal("canceled decode lost its completed prompt boundary")
	}
	cache := qwen35BatchCache(r.config, r.qwen35, r.prefixCache.cache.kvFormat())
	buf := NewDecodeBuffer(r.config, r.config.HeadDim, r.config.NKVHeads, r.config.ValueDim)
	var mtp *Qwen35MTPState
	if r.prefixCache.qwen35MTP != nil {
		mtp = newQwen35MTPState(r.config, cache.MaxLen, cache.kvFormat())
	}
	for pos, token := range r.prefixCache.tokens {
		ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
		if mtp != nil {
			qwen35MTPProcess(r.config, r.qwen35.MTP, mtp, token, buf.XN, pos)
		}
	}
	requireQwen35BatchValues(t, "checkpoint conv at advertised boundary", r.prefixCache.qwen35.Conv, cache.Qwen35.Conv)
	requireQwen35BatchValues(t, "checkpoint recurrence at advertised boundary", r.prefixCache.qwen35.State, cache.Qwen35.State)
	requireQwen35BatchValues(t, "live conv at advertised boundary", r.prefixCache.cache.Qwen35.Conv, cache.Qwen35.Conv)
	requireQwen35BatchValues(t, "live recurrence at advertised boundary", r.prefixCache.cache.Qwen35.State, cache.Qwen35.State)
	speculativeRequireSameCacheRows(t, r.prefixCache.cache, cache, len(r.prefixCache.tokens))
	if r.prefixCache.qwen35MTP != nil {
		requireQwen35BatchValues(t, "canceled MTP pending hidden", r.prefixCache.qwen35MTP, buf.XN)
		speculativeRequireSameCacheRows(t, r.prefixCache.cache.Qwen35MTP.KV, mtp.KV, len(r.prefixCache.tokens))
	}
}

func TestSpeculativeMTPRuntimeMatchesTargetAndWarmFollowup(t *testing.T) {
	for _, target := range []uint32{0, 3} {
		for _, draft := range []int{1, 2, 4, 7} {
			for _, penalty := range []float32{1, 1.1} {
				t.Run(fmt.Sprintf("target=%d/draft=%d/penalty=%.1f", target, draft, penalty), func(t *testing.T) {
					baseline, speculative := speculativeFixedQwenRunner(t, target), speculativeFixedQwenRunner(t, target)
					opts := speculativeRuntimeOptions()
					opts.Sampler.RepeatPenalty = penalty
					messages := []ChatMessage{UserMessage("aaaaaaaa")}
					want, err := baseline.GenerateChat(messages, opts)
					if err != nil {
						t.Fatal(err)
					}
					opts.MTPDraftTokens = draft
					got, err := speculative.GenerateChat(messages, opts)
					if err != nil {
						t.Fatal(err)
					}
					speculativeRequireSameResult(t, got, want)
					if got.Stats.GeneratedTokens != opts.MaxTokens {
						t.Fatal("fixed target did not exercise a full speculative completion")
					}
					if speculative.workspaceBuf.speculative.checkpoint == nil {
						t.Fatal("eligible native MTP did not enter block verification")
					}
					speculativeRequireSameVisibleState(t, speculative, baseline)
					followup := append(slices.Clone(messages), AssistantMessage(got.Text), UserMessage("aaaa"))
					baseOpts := opts
					baseOpts.MTPDraftTokens = 0
					want, err = baseline.GenerateChat(followup, baseOpts)
					if err != nil {
						t.Fatal(err)
					}
					got, err = speculative.GenerateChat(followup, opts)
					if err != nil {
						t.Fatal(err)
					}
					speculativeRequireSameResult(t, got, want)
					if got.PromptCache == nil || !got.PromptCache.Hit {
						t.Fatal("speculative target lost its reusable warm prefix")
					}
					speculativeRequireSameVisibleState(t, speculative, baseline)
				})
			}
		}
	}
}

func TestSpeculativeMTPRuntimeStopAndCancellation(t *testing.T) {
	for _, mode := range []string{"stop", "callback", "context", "eos"} {
		t.Run(mode, func(t *testing.T) {
			var runs [2]*Runner
			var results [2]GenerationResult
			var returned [2]error
			messages := []ChatMessage{UserMessage("aaaaaaaa")}
			for i := range runs {
				target := uint32(0)
				if mode == "eos" {
					target = 2
				}
				r := speculativeFixedQwenRunner(t, target)
				runs[i] = r
				opts := speculativeRuntimeOptions()
				if i == 1 {
					opts.MTPDraftTokens = 7
				}
				ctx, cancel := context.WithCancel(context.Background())
				if mode == "context" {
					opts = opts.WithContext(ctx)
				}
				if mode == "stop" {
					opts.StopSequences = []string{"aaa"}
				}
				calls := 0
				results[i], returned[i] = r.GenerateChatStreamUntil(messages, opts, func(string) bool {
					calls++
					if calls == 2 {
						if mode == "context" {
							cancel()
						}
						if mode == "callback" {
							return false
						}
					}
					return true
				})
				cancel()
			}
			speculativeRequireSameResult(t, results[1], results[0])
			switch mode {
			case "callback":
				if !errors.Is(returned[0], ErrGenerationCanceled) || !errors.Is(returned[1], ErrGenerationCanceled) {
					t.Fatalf("callback cancellation errors=%v", returned)
				}
			case "context":
				if !errors.Is(returned[0], context.Canceled) || !errors.Is(returned[1], context.Canceled) {
					t.Fatalf("context cancellation errors=%v", returned)
				}
			default:
				if returned[0] != nil || returned[1] != nil || results[1].FinishReason != "stop" {
					t.Fatalf("natural stop errors=%v results=%+v", returned, results)
				}
			}
			if mode == "callback" || mode == "context" {
				for _, r := range runs {
					speculativeRequireConsistentBoundary(t, r)
				}
			} else {
				speculativeRequireSameVisibleState(t, runs[1], runs[0])
			}
			if mode == "eos" {
				if results[1].Stats.GeneratedTokens != 0 || results[1].Text != "" {
					t.Fatal("EOS was emitted as a visible generated token")
				}
				return
			}
			followup := append(slices.Clone(messages), AssistantMessage(results[1].Text), UserMessage("aaaa"))
			baseOpts := speculativeRuntimeOptions()
			want, err := runs[0].GenerateChat(followup, baseOpts)
			if err != nil {
				t.Fatal(err)
			}
			baseOpts.MTPDraftTokens = 7
			got, err := runs[1].GenerateChat(followup, baseOpts)
			if err != nil {
				t.Fatal(err)
			}
			speculativeRequireSameResult(t, got, want)
			speculativeRequireSameVisibleState(t, runs[1], runs[0])
		})
	}
}

func TestSpeculativeMTPUnmodifiedDenseRuntimeParity(t *testing.T) {
	for _, penalty := range []float32{1, 1.1} {
		t.Run(fmt.Sprintf("penalty=%.1f", penalty), func(t *testing.T) {
			baseline, speculative := qwen35DenseRuntimeRunner(t, true), qwen35DenseRuntimeRunner(t, true)
			baseline.config.MaxSeqLen, speculative.config.MaxSeqLen = 128, 128
			opts := speculativeRuntimeOptions()
			opts.Sampler.RepeatPenalty = penalty
			messages := []ChatMessage{UserMessage(strings.Repeat("ab", 4))}
			want, err := baseline.GenerateChat(messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			opts.MTPDraftTokens = 7
			got, err := speculative.GenerateChat(messages, opts)
			if err != nil {
				t.Fatal(err)
			}
			speculativeRequireSameResult(t, got, want)
			speculativeRequireSameVisibleState(t, speculative, baseline)
		})
	}
}

func TestSpeculativePromptLookupRuntimeAndMTPMix(t *testing.T) {
	for _, target := range []uint32{0, 3} {
		for _, lookup := range []int{1, 4, 7} {
			for _, mtp := range []int{0, 3} {
				for _, penalty := range []float32{1, 1.1} {
					t.Run(fmt.Sprintf("target=%d/lookup=%d/mtp=%d/penalty=%.1f", target, lookup, mtp, penalty), func(t *testing.T) {
						baseline, speculative := speculativeFixedQwenRunner(t, target), speculativeFixedQwenRunner(t, target)
						opts := speculativeRuntimeOptions()
						opts.Sampler.RepeatPenalty = penalty
						messages := []ChatMessage{UserMessage("aaaaaaaa")}
						want, err := baseline.GenerateChat(messages, opts)
						if err != nil {
							t.Fatal(err)
						}
						opts.PromptLookupTokens, opts.MTPDraftTokens = lookup, mtp
						got, err := speculative.GenerateChat(messages, opts)
						if err != nil {
							t.Fatal(err)
						}
						speculativeRequireSameResult(t, got, want)
						if speculative.workspaceBuf.speculative.checkpoint == nil {
							t.Fatal("repeated history did not exercise prompt-lookup block verification")
						}
						speculativeRequireSameVisibleState(t, speculative, baseline)
					})
				}
			}
		}
	}
}

func TestSpeculativeBlockCapabilityFallbacks(t *testing.T) {
	t.Setenv("GOPHERLLM_NO_BLOCK_VERIFY", "")
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "")
	for _, quantized := range []bool{false, true} {
		cfg, weights := qwen35BatchFixture(quantized)
		r := &Runner{kind: loadedQwen35, config: cfg, qwen35: weights}
		cache := qwen35BatchCache(cfg, weights, kvF32)
		want := !quantized || useQ8Activations.Load()
		if r.canVerifyBlock(cache) != want {
			t.Fatalf("quant=%t Q8=%t block eligibility=%t, want %t", quantized, useQ8Activations.Load(), r.canVerifyBlock(cache), want)
		}
		r.outOfCore = true
		if r.canVerifyBlock(cache) {
			t.Fatal("out-of-core Qwen entered speculative batch verification")
		}
		r.outOfCore = false
		r.config.Arch = "qwen35moe"
		if r.canVerifyBlock(cache) {
			t.Fatal("hybrid MoE entered dense block verification")
		}
		r.config.Arch = "qwen35"
		r.qwen35.Layers[0].DeltaNet.QKVConv.Metal = new(MetalWeight)
		if r.canVerifyBlock(cache) {
			t.Fatal("selectively offloaded Qwen entered CPU block verification")
		}
	}
	baseline, speculative := speculativeFixedQwenRunner(t, 0), speculativeFixedQwenRunner(t, 0)
	opts := speculativeRuntimeOptions()
	messages := []ChatMessage{UserMessage("aaaaaaaa")}
	want, err := baseline.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOPHERLLM_NO_BLOCK_VERIFY", "1")
	opts.MTPDraftTokens, opts.PromptLookupTokens = 7, 7
	got, err := speculative.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	speculativeRequireSameResult(t, got, want)
	speculativeRequireSameVisibleState(t, speculative, baseline)
	if speculative.workspaceBuf.speculative.checkpoint != nil {
		t.Fatal("disabled block verification created a recurrent checkpoint")
	}
}

func TestSpeculativeQwenBlockCommitBoundaries(t *testing.T) {
	for _, positions := range []int{2, 4, 8} {
		for keep := 0; keep <= positions; keep++ {
			t.Run(fmt.Sprintf("positions=%d/keep=%d", positions, keep), func(t *testing.T) {
				batch, serial := speculativeFixedQwenRunner(t, 0), speculativeFixedQwenRunner(t, 0)
				bc, bb := batch.generationWorkspace(48, true)
				sc, sb := serial.generationWorkspace(48, true)
				prefix := []uint32{3, 0, 3}
				for pos, token := range prefix {
					ForwardQwen35BodyInto(batch.config, batch.qwen35, bc, bb, token, pos)
					qwen35MTPProcess(batch.config, batch.qwen35.MTP, bc.Qwen35MTP, token, bb.XN, pos)
					ForwardQwen35BodyInto(serial.config, serial.qwen35, sc, sb, token, pos)
					qwen35MTPProcess(serial.config, serial.qwen35.MTP, sc.Qwen35MTP, token, sb.XN, pos)
				}
				inputs := make([]uint32, positions)
				for i := range inputs {
					inputs[i] = uint32(i % 2 * 3)
				}
				recent := append(slices.Clone(prefix), inputs[0])
				handled, err := batch.prepareSpeculativeBlock(context.Background(), bc, bb, inputs, len(prefix), recent, 1.1)
				if err != nil || !handled || !bb.speculative.active {
					t.Fatalf("block preparation: handled=%t active=%t err=%v", handled, bb.speculative.active, err)
				}
				if err := batch.finishSpeculativeBlock(context.Background(), bc, bb, keep, true, false); err != nil || bb.speculative.active {
					t.Fatalf("block commit: active=%t err=%v", bb.speculative.active, err)
				}
				for i, token := range inputs[:keep] {
					ForwardQwen35BodyInto(serial.config, serial.qwen35, sc, sb, token, len(prefix)+i)
					qwen35MTPProcess(serial.config, serial.qwen35.MTP, sc.Qwen35MTP, token, sb.XN, len(prefix)+i)
				}
				requireQwen35BatchValues(t, "committed conv", bc.Qwen35.Conv, sc.Qwen35.Conv)
				requireQwen35BatchValues(t, "committed recurrence", bc.Qwen35.State, sc.Qwen35.State)
				requireQwen35BatchValues(t, "committed residual", bb.X, sb.X)
				requireQwen35BatchValues(t, "committed target hidden", bb.XN, sb.XN)
				requireQwen35BatchValues(t, "committed MTP pending", bc.Qwen35MTP.PendingHidden, sc.Qwen35MTP.PendingHidden)
				speculativeRequireSameCacheRows(t, bc, sc, len(prefix)+keep)
				speculativeRequireSameCacheRows(t, bc.Qwen35MTP.KV, sc.Qwen35MTP.KV, len(prefix)+keep)
			})
		}
	}
}

func TestSpeculativeLookupMiddleRejectionAndMTPCatchup(t *testing.T) {
	// The generated seed completes the final eight-token suffix. Its sole
	// earlier occurrence continues with [0,3]: the constant target accepts 0
	// and rejects 3 in the middle of this first two-candidate window.
	prompt := []uint32{4, 5, 6, 7, 8, 9, 10, 0, 0, 3, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10}
	if drafts := promptLookupDraft(prompt, []uint32{0}, 2, nil); !slices.Equal(drafts, []uint32{0, 3}) {
		t.Fatalf("fixture continuation=%v, want accepted token then rejected token", drafts)
	}
	for _, mtp := range []int{0, 3} {
		t.Run(fmt.Sprintf("mtp=%d", mtp), func(t *testing.T) {
			baseline, speculative := speculativeFixedQwenRunner(t, 0), speculativeFixedQwenRunner(t, 0)
			opts := speculativeRuntimeOptions()
			opts.MaxTokens = 4
			messages := []ChatMessage{UserMessage("controlled lookup history")}
			// Use an owned preflight-token fixture to make the exact lookup
			// history independent of chat-template suffixes. All generation,
			// verification, emission and prefix-cache code still runs normally.
			generate := func(r *Runner, tokens []uint32, options GenerationOptions) (GenerationResult, error) {
				prepared := &preparedChatData{tokens: slices.Clone(tokens)}
				return r.generateChatStreamUntil(messages, options, nil, prepared)
			}
			want, err := generate(baseline, prompt, opts)
			if err != nil {
				t.Fatal(err)
			}
			opts.PromptLookupTokens, opts.MTPDraftTokens = 2, mtp
			got, err := generate(speculative, prompt, opts)
			if err != nil {
				t.Fatal(err)
			}
			speculativeRequireSameResult(t, got, want)
			stats := got.Stats.Speculation
			if got.Text != "aaaa" || stats == nil || stats.Blocks != 1 || stats.DraftedTokens != 2 || stats.AcceptedTokens != 1 || stats.DiscardedInputs != 1 {
				t.Fatalf("middle rejection did not accept one token and discard the rejected input: text=%q stats=%+v", got.Text, stats)
			}
			speculativeRequireSameVisibleState(t, speculative, baseline)
			speculativeRequireConsistentBoundary(t, speculative)
			followup := append(slices.Clone(prompt), 0, 0, 0, 0, 13, 14)
			baseOpts := opts
			baseOpts.MTPDraftTokens, baseOpts.PromptLookupTokens = 0, 0
			want, err = generate(baseline, followup, baseOpts)
			if err != nil {
				t.Fatal(err)
			}
			got, err = generate(speculative, followup, opts)
			if err != nil {
				t.Fatal(err)
			}
			speculativeRequireSameResult(t, got, want)
			if got.PromptCache == nil || !got.PromptCache.Hit {
				t.Fatal("middle rejection lost its valid warm prefix")
			}
			speculativeRequireSameVisibleState(t, speculative, baseline)
			speculativeRequireConsistentBoundary(t, speculative)
		})
	}
}

func TestSpeculativeMTPSerialPauseKeepsDraftCacheSynchronized(t *testing.T) {
	// The target always chooses 3 while MTP chooses 0. Three rejected
	// three-candidate windows shrink to one; three more trigger a serial pause.
	baseline, speculative := speculativeFixedQwenRunner(t, 3), speculativeFixedQwenRunner(t, 3)
	opts := speculativeRuntimeOptions()
	messages := []ChatMessage{UserMessage("aaaaaaaa")}
	want, err := baseline.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	opts.MTPDraftTokens = 7
	got, err := speculative.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	speculativeRequireSameResult(t, got, want)
	stats := got.Stats.Speculation
	if stats == nil || stats.Blocks != 6 || stats.DraftedTokens != 12 || stats.AcceptedTokens != 0 {
		t.Fatalf("rejected MTP work did not enter the serial pause: %+v", stats)
	}
	speculativeRequireSameVisibleState(t, speculative, baseline)
	speculativeRequireConsistentBoundary(t, speculative)
}
