package gopherllm

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
)

func (r *Runner) forwardTokenInto(cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	switch r.kind {
	case loadedNemotronH:
		ForwardNemotronHInto(r.config, r.nemotronH, cache, buf, token, pos, logits)
	case loadedMamba2:
		ForwardMamba2Into(r.config, r.mamba2, cache, buf, token, pos, logits)
	case loadedQwen35:
		ForwardQwen35Into(r.config, r.qwen35, cache, buf, token, pos, logits)
	case loadedGptOss:
		ForwardGptOssInto(r.config, r.gptOss, cache, buf, token, pos, logits)
	case loadedGemma4:
		ForwardGemma4Into(r.config, r.gemma4, cache, buf, token, pos, logits)
	default:
		ForwardInto(r.config, r.standard, cache, buf, token, pos, logits)
	}
}

func (r *Runner) canGreedyOutputFastPath(options GenerationOptions) bool {
	s := options.Sampler
	return os.Getenv("GOPHERLLM_NO_GREEDY_ARGMAX") != "1" &&
		(s.Temperature < 1e-6 || s.TopK == 1)
}

// mtpDraftTokenCount validates the opt-in MTP path at the point where both
// the loaded architecture and the sampler are known. MTP candidates are
// accepted only when they equal the target's deterministic choice, which is
// exact for greedy decoding; using that shortcut for stochastic sampling would
// bias the output distribution, so it is deliberately refused.
func (r *Runner) mtpDraftTokenCount(options GenerationOptions) (int, error) {
	if options.MTPDraftTokens == 0 {
		return 0, nil
	}
	if r.kind != loadedQwen35 || r.qwen35.MTP == nil {
		return 0, fmt.Errorf("mtp_draft_tokens requires a Qwen3.5/3.6/3.8 GGUF with a loaded NextN/MTP draft layer")
	}
	if options.Sampler.Temperature >= 1e-6 && options.Sampler.TopK != 1 {
		return 0, fmt.Errorf("mtp_draft_tokens requires greedy sampling (temperature 0 or top_k 1) to preserve exact output")
	}
	return options.MTPDraftTokens, nil
}

// greedyOutputToken finds the deterministic next token from a transformer body
// already left in buf.XN. Keeping this separate from forwardGreedyToken lets
// greedy prompt prefill use the same no-logits argmax path as decode: a
// vocabulary-sized projection does not need to cross back to Go merely to
// select one token. If an output form cannot take that shortcut, this function
// restores the established materialized-logits fallback for the sampler.
func (r *Runner) greedyOutputToken(buf *DecodeBuffer, recent []uint32, repeatPenalty float32, logits *[]float32) (uint32, bool) {
	switch r.kind {
	case loadedNemotronH:
		if next, ok := argmaxOutputTokenPenalizedInto(r.config, ModelWeights{Output: r.nemotronH.Output}, buf, recent, repeatPenalty, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, ModelWeights{Output: r.nemotronH.Output}, buf, logits)
	case loadedMamba2:
		if next, ok := argmaxOutputTokenPenalizedInto(r.config, ModelWeights{Output: r.mamba2.Output}, buf, recent, repeatPenalty, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, ModelWeights{Output: r.mamba2.Output}, buf, logits)
	case loadedQwen35:
		if next, ok := argmaxOutputTokenPenalizedInto(r.config, ModelWeights{Output: r.qwen35.Output}, buf, recent, repeatPenalty, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, ModelWeights{Output: r.qwen35.Output}, buf, logits)
	case loadedGptOss:
		if next, ok := argmaxOutputTokenPenalizedInto(r.config, r.gptOss.Standard, buf, recent, repeatPenalty, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.gptOss.Standard, buf, logits)
	case loadedGemma4:
		if r.gemma4.Native {
			nativeOutput := ModelWeights{Output: r.gemma4.Output}
			if next, ok := argmaxOutputTokenPenalizedInto(r.config, nativeOutput, buf, recent, repeatPenalty, logits); ok {
				return next, true
			}
			projectNativeGemma4Logits(r.config, r.gemma4, buf, logits)
			break
		}
		if next, ok := argmaxOutputTokenPenalizedInto(r.config, r.gemma4.Standard, buf, recent, repeatPenalty, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.gemma4.Standard, buf, logits)
	default:
		if next, ok := argmaxOutputTokenPenalizedInto(r.config, r.standard, buf, recent, repeatPenalty, logits); ok {
			return next, true
		}
		ProjectLogitsInto(r.config, r.standard, buf, logits)
	}
	return 0, false
}

func (r *Runner) forwardGreedyToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int, recent []uint32, repeatPenalty float32, logits *[]float32) (uint32, bool) {
	switch r.kind {
	case loadedNemotronH:
		ForwardNemotronHBodyInto(r.config, r.nemotronH, cache, buf, token, pos)
	case loadedMamba2:
		ForwardMamba2BodyInto(r.config, r.mamba2, cache, buf, token, pos)
	case loadedQwen35:
		ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
	case loadedGptOss:
		ForwardBodyInto(r.config, r.gptOss.Standard, cache, buf, token, pos)
	case loadedGemma4:
		if r.gemma4.Native {
			forwardNativeGemma4BodyInto(r.config, r.gemma4, cache, buf, token, pos)
		} else {
			ForwardBodyInto(r.config, r.gemma4.Standard, cache, buf, token, pos)
		}
	default:
		ForwardBodyInto(r.config, r.standard, cache, buf, token, pos)
	}
	return r.greedyOutputToken(buf, recent, repeatPenalty, logits)
}

func (r *Runner) forwardHiddenToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	switch r.kind {
	case loadedNemotronH:
		ForwardNemotronHBodyInto(r.config, r.nemotronH, cache, buf, token, pos)
	case loadedMamba2:
		ForwardMamba2BodyInto(r.config, r.mamba2, cache, buf, token, pos)
	case loadedQwen35:
		ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
	case loadedGptOss:
		ForwardBodyInto(r.config, r.gptOss.Standard, cache, buf, token, pos)
	case loadedGemma4:
		if r.gemma4.Native {
			forwardNativeGemma4BodyInto(r.config, r.gemma4, cache, buf, token, pos)
		} else {
			ForwardBodyInto(r.config, r.gemma4.Standard, cache, buf, token, pos)
		}
	default:
		ForwardBodyInto(r.config, r.standard, cache, buf, token, pos)
	}
	return buf.XN
}

// batchPrefillWeights returns the ModelWeights ForwardBatchInto should stream
// for the current runner, and whether r.kind is even eligible for the batched
// path. loadedStandard is the obvious case. loadedGemma4 also qualifies
// whenever it is not the native Gemma 4 graph: LoadGemma4Model builds that
// case's Gemma4Weights.Standard via the exact same LoadModel/loadLayer path
// used for loadedStandard (see model.go), and every non-native decode/prefill
// call (ForwardGemma4Into, forwardPrefillToken, forwardHiddenToken) already
// dispatches straight to the standard ForwardInto/ForwardBodyInto/ForwardPrefill
// on that same weights value. So a non-native Gemma/Gemma2/Gemma3/dense-Gemma4
// model is structurally just a loadedStandard model stored in a different
// Runner field, and can reuse the identical batched graph with no new math.
// Native Gemma 4 (PLE/shared-expert graph) and every other specialised kind
// (Qwen3.5, GPT-OSS, NemotronH, Mamba2, MLA) keep the per-token path because
// their forward functions are not ForwardBatchInto-shaped.
func (r *Runner) batchPrefillWeights() (ModelWeights, bool) {
	switch r.kind {
	case loadedStandard:
		return r.standard, true
	case loadedGemma4:
		if r.gemma4.Native {
			return ModelWeights{}, false
		}
		return r.gemma4.Standard, true
	default:
		return ModelWeights{}, false
	}
}

// canBatchPrefill reports whether the model uses the standard non-fused
// transformer path that ForwardBatchInto supports.
func (r *Runner) canBatchPrefill() bool {
	if r.outOfCore || r.config.UsesMLA {
		return false
	}
	weights, ok := r.batchPrefillWeights()
	if !ok || len(weights.Layers) == 0 {
		return false
	}
	if os.Getenv("GOPHERLLM_NO_BATCH_PREFILL") != "" {
		return false
	}
	// The batched graph implements RMSNorm, LayerNorm, QK norm, post norm, and
	// StableLM's parallel attention/FFN residual. Sparse experts retain the
	// per-token path because their token-dependent routing needs a separate
	// batched execution strategy.
	for _, l := range weights.Layers {
		if l.MoE != nil {
			return false
		}
	}
	return true
}

// prefillBatched processes the prompt in chunks, streaming each weight once per
// chunk instead of once per token.
//
// Generic synthetic sweeps peak near 32 as activation buffers outgrow cache,
// while the dense 3B path benefits from more dequantization amortization and
// uses the model-aware default below. Deployment-specific A/B runs can override
// either choice through GOPHERLLM_PREFILL_CHUNK.
func (r *Runner) prefillBatched(ctx context.Context, cache *KVCache, buf *DecodeBuffer, tokens []uint32, logits *[]float32) error {
	return r.prefillBatchedAt(ctx, cache, buf, tokens, 0, logits)
}

// prefillBatchedAt is prefillBatched with an absolute KV-cache offset. The
// offset is what lets an append-only chat process only the new rendered-token
// suffix while attention still reads the cached prefix rows.
func (r *Runner) prefillBatchedAt(ctx context.Context, cache *KVCache, buf *DecodeBuffer, tokens []uint32, startPos int, logits *[]float32) error {
	weights, ok := r.batchPrefillWeights()
	if !ok {
		return fmt.Errorf("gopherllm: prefillBatchedAt called for a kind that cannot batch prefill")
	}
	chunk := r.prefillChunkSize()
	n := len(tokens)
	for start := 0; start < n; start += chunk {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := min(start+chunk, n)
		ForwardBatchInto(r.config, weights, cache, buf, tokens[start:end], startPos+start, end == n, logits)
	}
	return nil
}

// prefillChunkOverride, when positive, wins over both the env var and the
// model-aware default. The autotuner sets it after measuring the real batched
// forward on the real hardware, which beats any static heuristic.
var prefillChunkOverride atomic.Int64

// SetPrefillChunk pins the prompt-prefill chunk size; 0 restores the default
// (env var, else the model-aware heuristic).
func SetPrefillChunk(n int) {
	if n < 0 {
		n = 0
	}
	prefillChunkOverride.Store(int64(n))
}

// prefillChunkSize resolves the chunk size actually used: an explicit override
// (set by the autotuner) wins, otherwise the env var, otherwise the
// model-aware default.
func prefillChunkSize(config Config) int {
	if n := int(prefillChunkOverride.Load()); n > 0 {
		return n
	}
	return prefillChunkDefault(config)
}

// prefillChunkSize resolves a runner-local prefill default without changing
// the package-level heuristic used by CPU-only and non-standard models. A
// complete direct-Metal SwiGLU graph benefits from keeping the whole 256-token
// slab in its GPU workspace: splitting it at the ordinary 128-token default
// pays the command-buffer setup and synchronization cost twice. Any explicit
// process override (the autotuner) or environment value remains authoritative.
//
// The Metal predicate is deliberately runner-local: it verifies the real
// prepared handles for every layer rather than guessing from model geometry.
// This means models with a partial/off Metal load, an unsupported FFN, or a
// non-standard graph retain the established CPU default.
func (r *Runner) prefillChunkSize() int {
	chunk := prefillChunkSize(r.config)
	if prefillChunkOverrideValue() > 0 || strings.TrimSpace(os.Getenv("GOPHERLLM_PREFILL_CHUNK")) != "" {
		return chunk
	}
	if metalChunk := r.metalBatchFFNPrefillChunk(); metalChunk > chunk {
		return metalChunk
	}
	return chunk
}

// prefillChunkOverrideValue reports the raw override, 0 when unset. Callers that
// save and restore the setting must use this rather than prefillChunkSize:
// re-applying a resolved value would pin it as an explicit override and thereby
// suppress the env var and the model-aware default.
func prefillChunkOverrideValue() int { return int(prefillChunkOverride.Load()) }

// prefillChunkDefault keeps a conservative default for larger models while using
// larger chunks on small dense models such as Ministral-3-3B. Bottleneck:
// prompt prefill is chunk-size sensitive because larger chunks amortize
// dequantization but grow activation working sets. Change: small models default
// to 128 after real Ministral measurement; GOPHERLLM_PREFILL_CHUNK can override
// it for A/B testing. Expected effect: lower TTFT on 3B-class models. Risk:
// too-large chunks can regress cache locality on bigger models. Rollback: set
// GOPHERLLM_PREFILL_CHUNK=32. --auto measures this instead of guessing it.
func prefillChunkDefault(config Config) int {
	const def = 32
	raw := strings.TrimSpace(os.Getenv("GOPHERLLM_PREFILL_CHUNK"))
	if raw == "" {
		if config.Dim > 0 && config.Dim <= 3072 && config.HiddenDim <= 12288 {
			return 128
		}
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return def
	}
	if n > 256 {
		return 256
	}
	return n
}

func (r *Runner) forwardPrefillToken(cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	switch r.kind {
	case loadedNemotronH:
		ForwardNemotronHBodyInto(r.config, r.nemotronH, cache, buf, token, pos)
	case loadedMamba2:
		ForwardMamba2BodyInto(r.config, r.mamba2, cache, buf, token, pos)
	case loadedQwen35:
		ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
	case loadedGptOss:
		ForwardPrefill(r.config, r.gptOss.Standard, cache, buf, token, pos)
	case loadedGemma4:
		if r.gemma4.Native {
			forwardNativeGemma4BodyInto(r.config, r.gemma4, cache, buf, token, pos)
		} else {
			ForwardPrefill(r.config, r.gemma4.Standard, cache, buf, token, pos)
		}
	default:
		ForwardPrefill(r.config, r.standard, cache, buf, token, pos)
	}
}
