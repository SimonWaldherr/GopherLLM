package gopherllm

import (
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"
)

var ErrGenerationCanceled = errors.New("generation canceled")

// repeatPenaltyWindow bounds the prompt and output history considered by the
// sampler's repeat penalty. Keeping the same window from the first sampled
// token avoids a long-prompt allocation and an O(prompt length) first-token
// penalty pass.
const repeatPenaltyWindow = 64

func (r *Runner) Generate(prompt string, options GenerationOptions) (GenerationResult, error) {
	return r.GenerateChat([]ChatMessage{UserMessage(prompt)}, options)
}

func (r *Runner) GenerateChat(messages []ChatMessage, options GenerationOptions) (GenerationResult, error) {
	return r.GenerateChatStream(messages, options, func(string) {})
}

func (r *Runner) GenerateStream(prompt string, options GenerationOptions, onToken func(string)) (GenerationResult, error) {
	return r.GenerateChatStream([]ChatMessage{UserMessage(prompt)}, options, onToken)
}

func (r *Runner) GenerateChatStream(messages []ChatMessage, options GenerationOptions, onToken func(string)) (GenerationResult, error) {
	if onToken == nil {
		return r.GenerateChatStreamUntil(messages, options, nil)
	}
	return r.GenerateChatStreamUntil(messages, options, func(text string) bool {
		onToken(text)
		return true
	})
}

// GenerateChatStreamUntil is the generation entry point everything else wraps
// (Generate, GenerateChat, GenerateStream, ... are thin adapters over it):
// render messages through the model's chat template, prefill the prompt
// (batched when the architecture allows), then decode token by token until
// EOS, a stop sequence, max_tokens, or the context limit. onToken receives
// valid-UTF-8 text increments (bytes are buffered across token boundaries
// until they complete a rune, and the tail is held back while it could still
// be a stop-sequence prefix); returning false cancels generation, yielding
// the partial result with ErrGenerationCanceled. A nil onToken discards
// streamed text and lets generation continue. The final result carries
// content with reasoning and tool calls already extracted (classifyOutput)
// and a FinishReason of "stop", "length", or "tool_calls".
func (r *Runner) GenerateChatStreamUntil(messages []ChatMessage, options GenerationOptions, onToken func(string) bool) (GenerationResult, error) {
	if onToken == nil {
		onToken = func(string) bool { return true }
	}
	if err := r.acquireModelLease(); err != nil {
		return GenerationResult{}, err
	}
	defer r.releaseModelLease()
	r.genLock.Lock()
	defer r.genLock.Unlock()
	// The vision cache is deliberately NOT cleared here. It used to be, on the
	// grounds that a webcam would otherwise grow it without bound -- true of
	// the unbounded map it was then, but it also threw away the encoding of a
	// still image between every turn of a conversation about that image, which
	// is the common case and costs a full tower pass each time. The cache is
	// bounded now (visionCachePut), so live frames evict themselves while a
	// re-sent still stays warm.
	if r.kind == loadedBERT {
		return GenerationResult{}, fmt.Errorf("%s is an embedding model and cannot generate chat completions", r.arch)
	}
	if err := options.Validate(); err != nil {
		return GenerationResult{}, err
	}
	mtpDraftTokens, err := r.mtpDraftTokenCount(options)
	if err != nil {
		return GenerationResult{}, err
	}
	if len(messages) == 0 {
		return GenerationResult{}, fmt.Errorf("no prompt provided")
	}
	totalStart := time.Now()
	var contextWindow *ContextWindowInfo
	// Do this at the shared generation entry point rather than only in the
	// HTTP handler. Agentic requests append assistant tool calls and tool
	// results between iterations; each subsequent model call must be budgeted
	// against those effective messages and active tools as well.
	if normalizedContextWindowMode(options.ContextWindowMode) != ContextWindowFull {
		prepared, info, err := r.PrepareChatContext(messages, options)
		if err != nil {
			return GenerationResult{}, err
		}
		messages = prepared
		contextWindow = &info
	}
	tokens, imageEmbeds, err := r.renderMessagesForGeneration(messages, options.SystemPrompt, options.ActiveTools())
	if err != nil {
		return GenerationResult{}, err
	}
	if len(tokens) == 0 {
		return GenerationResult{}, fmt.Errorf("prompt rendered to zero tokens")
	}
	if r.config.MaxSeqLen <= 0 {
		return GenerationResult{}, fmt.Errorf("model has an invalid context length (%d)", r.config.MaxSeqLen)
	}
	// The KV cache is sized to r.config.MaxSeqLen; without this check a prompt
	// at or beyond that length (easily reached once a request injects a large
	// tool listing) would silently overflow it deeper in the forward pass
	// instead of failing here with a clear error.
	if len(tokens) >= r.config.MaxSeqLen {
		return GenerationResult{}, fmt.Errorf("prompt (%d tokens) leaves no room to generate within the model's context length (%d)", len(tokens), r.config.MaxSeqLen)
	}
	cacheLen := generationCacheLen(r.config.MaxSeqLen, len(tokens), options.MaxTokens)
	cache, buf := r.generationWorkspace(cacheLen, mtpDraftTokens > 0)
	buf.ImageEmbeds = imageEmbeds
	defer func() { buf.ImageEmbeds = nil }()
	cacheInfo := PromptCacheInfo{Mode: "disabled", PromptTokens: len(tokens)}
	reusedTokens := 0
	// Prefix-cache matching is purely token-ID based (sharedTokenPrefix): it
	// cannot distinguish two different images that render to the same
	// placeholder-token count, which would silently reuse a stale KV-cache
	// prefix computed for a different image's pixels. Multimodal turns opt
	// out of prefix-cache reuse entirely rather than risk that — image
	// prefill already dominates a multimodal turn's cost far more than
	// prefix-cache reuse would save.
	cacheEligible := len(imageEmbeds) == 0 && r.prefixCacheSupportedWithMTP(cache, mtpDraftTokens > 0)
	// A grown Qwen workspace can still be retained for the current request
	// while exceeding the stricter K/V-plus-snapshot prefix budget. Drop an
	// old snapshot attached to that workspace immediately rather than keeping
	// memory that can no longer be reused.
	if !cacheEligible && r.kind == loadedQwen35 && r.prefixCache.cache == cache {
		r.clearPrefixCache()
	}
	// A multimodal request overwrites the reusable workspace with image-derived
	// KV rows. It cannot reuse a text prefix itself, but must also invalidate a
	// previously retained text prefix so the next text-only request cannot match
	// placeholder token IDs against stale image state.
	if len(imageEmbeds) > 0 {
		r.clearPrefixCache()
	}
	// The ordinary workspace cache covers the immediately preceding complete
	// prompt. An opt-in Mistral cache additionally keeps immutable K/V snapshots
	// of the static system-and-tools prefix, allowing divergent conversations to
	// share that expensive prefill. Derive it from the renderer and verify it
	// against the already-rendered token stream; images always opt out because
	// token IDs alone cannot identify their substituted embeddings.
	var mistralStaticPrefix []uint32
	var mistralKVPrefixEpoch uint64
	if cacheEligible {
		mistralStaticPrefix = r.mistralStaticPromptPrefix(messages, options.SystemPrompt, options.ActiveTools(), tokens)
		if len(mistralStaticPrefix) > 0 {
			if epoch, enabled := r.mistralKVPrefixCacheEpoch(); enabled {
				mistralKVPrefixEpoch = epoch
			} else {
				mistralStaticPrefix = nil
			}
		}
	}
	cachedResidentTokens := r.prefixCache.tokens
	cachedPromptTokens := r.prefixCache.promptTokens
	cachedPromptLogits := r.prefixCache.promptLogits
	if cacheEligible {
		cacheInfo.Mode = "prefix"
		reusedTokens = r.prefixReuseWithMTP(cache, tokens, mtpDraftTokens > 0)
		// Prefer the full, most-recent prompt cache when it has a longer match.
		// The static Mistral snapshot is only a proper token prefix, so it cannot
		// supply final-prompt logits and never takes the identical-prompt fast
		// path below.
		if len(mistralStaticPrefix) > reusedTokens {
			if copied, hit := r.mistralKVPrefixCacheReuse(cache, mistralStaticPrefix); hit && copied > reusedTokens {
				reusedTokens = copied
			}
		}
		cacheInfo.Hit = reusedTokens > 0
		cacheInfo.ReusedTokens = reusedTokens
	}
	seed := options.Seed
	if seed == 0 {
		seed = uint64(time.Now().UnixNano())
	}
	rng := NewRng(seed)
	ctx := options.generationContext()
	if err := ctx.Err(); err != nil {
		return GenerationResult{}, err
	}
	// Once this request writes into the shared workspace, prior metadata is no
	// longer safe. The deferred replacement records only resident KV rows.
	if cacheEligible {
		r.clearPrefixCache()
	}
	prefillOffset := reusedTokens
	logits := buf.Logits
	if prefillOffset == len(tokens) {
		if cachedPromptTokens == len(tokens) && len(cachedPromptLogits) > 0 {
			ensureLenNoClear(&logits, len(cachedPromptLogits))
			copy(logits, cachedPromptLogits)
		} else {
			prefillOffset = max(0, len(tokens)-1)
			cacheInfo.Hit = prefillOffset > 0
			cacheInfo.ReusedTokens = prefillOffset
		}
	}
	prefillBegan := time.Now()
	residentTokens := cachedResidentTokens[:0]
	promptLogits := cachedPromptLogits[:0]
	prefillComplete := false
	defer func() {
		buf.Logits = logits
		// Clone only after the full prompt prefill succeeded. A canceled or
		// failed prefill may have written a partial set of rows, which must never
		// become a reusable prefix. This deliberately runs after streaming/decode
		// so copying an opt-in snapshot does not add to TTFT.
		if cacheEligible && prefillComplete && len(mistralStaticPrefix) > 0 && mistralKVPrefixEpoch != 0 {
			r.putMistralKVPrefixSnapshot(cache, mistralStaticPrefix, mistralKVPrefixEpoch)
		}
		if !cacheEligible {
			return
		}
		if len(residentTokens) == 0 {
			r.clearPrefixCache()
			return
		}
		var qwen35Snapshot *Qwen35Cache
		var qwen35MTP []float32
		if r.kind == loadedQwen35 {
			qwen35Snapshot = cache.Qwen35.snapshot()
			if qwen35Snapshot == nil {
				r.clearPrefixCache()
				return
			}
			if mtpDraftTokens > 0 {
				qwen35MTP = cache.Qwen35MTP.pendingSnapshot()
				if qwen35MTP == nil {
					r.clearPrefixCache()
					return
				}
			}
		}
		state := prefixCacheState{
			cache:        cache,
			tokens:       residentTokens,
			promptTokens: len(tokens),
			promptLogits: promptLogits,
			qwen35:       qwen35Snapshot,
			qwen35MTP:    qwen35MTP,
		}
		r.prefixCache = state
	}()
	if prefillOffset < len(tokens) {
		if r.canBatchPrefill() {
			if err := r.prefillBatchedAt(ctx, cache, buf, tokens[prefillOffset:], prefillOffset, &logits); err != nil {
				return GenerationResult{}, err
			}
		} else {
			for pos := prefillOffset; pos < len(tokens); pos++ {
				if err := ctx.Err(); err != nil {
					return GenerationResult{}, err
				}
				if pos == len(tokens)-1 {
					r.forwardTokenInto(cache, buf, tokens[pos], pos, &logits)
				} else {
					r.forwardPrefillToken(cache, buf, tokens[pos], pos)
				}
				if mtpDraftTokens > 0 {
					qwen35MTPProcess(r.config, r.qwen35.MTP, cache.Qwen35MTP, tokens[pos], buf.XN, pos)
				}
			}
		}
	}
	if cacheEligible {
		ensureLenNoClear(&promptLogits, len(logits))
		copy(promptLogits, logits)
	}
	residentTokens = append(residentTokens, tokens...)
	prefillComplete = true
	prefillTime := time.Since(prefillBegan)
	decodeStart := time.Now()
	var ttft time.Duration
	output := strings.Builder{}
	maxStopLen := 0
	for _, stop := range options.StopSequences {
		maxStopLen = max(maxStopLen, len(stop))
	}
	streamBuf := buf.StreamBytes[:0]
	flushStream := func(final bool) bool {
		if len(streamBuf) == 0 {
			return true
		}
		n := validUTF8PrefixLen(streamBuf)
		if n == 0 {
			if final {
				streamBuf = streamBuf[:0]
			}
			return true
		}
		if !final && maxStopLen > 1 {
			hold := maxStopLen - 1
			if n <= hold {
				return true
			}
			n -= hold
		}
		text := string(streamBuf[:n])
		if !onToken(text) {
			return false
		}
		copy(streamBuf, streamBuf[n:])
		streamBuf = streamBuf[:len(streamBuf)-n]
		if final && !utf8.Valid(streamBuf) {
			streamBuf = streamBuf[:0]
		}
		return true
	}
	generated := buf.GeneratedTokens[:0]
	recent := recentTokenWindowInto(buf.RecentTokens[:0], tokens)
	defer func() {
		buf.GeneratedTokens = generated[:0]
		buf.RecentTokens = recent[:0]
		buf.StreamBytes = streamBuf[:0]
	}()
	pos := len(tokens)
	finishReason := "length"
	greedyFastPath := r.canGreedyOutputFastPath(options)
	haveNextToken := false
	var nextToken uint32
	buildResult := func() GenerationResult {
		stats := GenerationStats{PromptTokens: len(tokens), GeneratedTokens: len(generated), TTFT: ttft, PrefillTime: prefillTime, DecodeTime: time.Since(decodeStart), TotalTime: time.Since(totalStart)}
		content, reasoning, calls := r.classifyOutput(output.String(), options.ActiveTools(), rng)
		reason := finishReason
		if len(calls) > 0 {
			reason = "tool_calls"
		}
		return GenerationResult{Text: content, ReasoningText: reasoning, ToolCalls: calls, FinishReason: reason, Stats: stats, ContextWindow: contextWindow, PromptCache: &cacheInfo}
	}
	// emitToken owns every user-visible side effect of accepting a token. MTP
	// verification can accept several candidates in one outer decode pass, so
	// keeping this logic in one closure preserves stop-string, UTF-8 streaming,
	// cancellation, repetition-window, and accounting semantics exactly.
	emitToken := func(token uint32) (keepDecoding, canceled bool) {
		if r.isStopToken(token) {
			finishReason = "stop"
			return false, false
		}
		if ttft == 0 {
			ttft = time.Since(totalStart)
		}
		text := r.tok.DecodeToken(token)
		output.WriteString(text)
		current := output.String()
		if maxStopLen > 0 {
			windowStart := max(0, len(current)-maxStopLen-len(text))
			window := current[windowStart:]
			for _, stop := range options.StopSequences {
				if idx := strings.Index(window, stop); idx >= 0 {
					current = current[:windowStart+idx]
					output.Reset()
					output.WriteString(current)
					streamBuf = streamBuf[:0]
					finishReason = "stop"
					return false, false
				}
			}
		}
		streamBuf = append(streamBuf, text...)
		if !flushStream(false) {
			return false, true
		}
		generated = append(generated, token)
		recent = append(recent, token)
		if len(recent) > repeatPenaltyWindow {
			copy(recent, recent[len(recent)-repeatPenaltyWindow:])
			recent = recent[:repeatPenaltyWindow]
		}
		return true, false
	}
	// verifyToken advances the target by an already-emitted token and returns
	// its deterministic next choice. Its MTP catch-up happens only after the
	// target body has produced the real hidden state, which overwrites any
	// speculative MTP row at this position with the authoritative pair.
	verifyToken := func(token uint32, mtpPrepared bool) uint32 {
		var targetNext uint32
		if greedyFastPath {
			var ok bool
			targetNext, ok = r.forwardGreedyToken(cache, buf, token, pos, recent, options.Sampler.RepeatPenalty, &logits)
			if !ok {
				targetNext = SampleWithScratch(logits, options.Sampler, rng, recent, &buf.SamplerCandidates)
			}
		} else {
			r.forwardTokenInto(cache, buf, token, pos, &logits)
			targetNext = SampleWithScratch(logits, options.Sampler, rng, recent, &buf.SamplerCandidates)
		}
		if mtpDraftTokens > 0 {
			if mtpPrepared {
				// The seed MTP row was just computed from this exact token and
				// the same pending target hidden state. Reuse its KV rather than
				// evaluating the full draft block a second time; only the real
				// target boundary needs updating after verification.
				copy(cache.Qwen35MTP.PendingHidden, buf.XN[:r.config.Dim])
			} else {
				qwen35MTPProcess(r.config, r.qwen35.MTP, cache.Qwen35MTP, token, buf.XN, pos)
			}
		}
		if cacheEligible {
			residentTokens = append(residentTokens, token)
		}
		pos++
		return targetNext
	}
decode:
	for range options.MaxTokens {
		if err := ctx.Err(); err != nil {
			return buildResult(), err
		}
		token := nextToken
		if haveNextToken {
			haveNextToken = false
		} else {
			token = SampleWithScratch(logits, options.Sampler, rng, recent, &buf.SamplerCandidates)
		}
		keepDecoding, canceled := emitToken(token)
		if canceled {
			return buildResult(), ErrGenerationCanceled
		}
		if !keepDecoding {
			break
		}
		if len(generated) >= options.MaxTokens || pos >= cacheLen {
			break
		}
		if mtpDraftTokens == 0 {
			if greedyFastPath {
				var ok bool
				nextToken, ok = r.forwardGreedyToken(cache, buf, token, pos, recent, options.Sampler.RepeatPenalty, &logits)
				haveNextToken = ok
				if cacheEligible {
					residentTokens = append(residentTokens, token)
				}
			} else {
				r.forwardTokenInto(cache, buf, token, pos, &logits)
				if cacheEligible {
					residentTokens = append(residentTokens, token)
				}
			}
			pos++
			continue
		}

		// Seed MTP with the just-emitted target token, then verify candidates
		// one by one. Qwen's hybrid target currently has a serial DeltaNet
		// recurrence, so verification remains sequential; keeping it exact here
		// makes the MTP head safe today and lets a future chunked verifier reuse
		// the synchronized cache without changing generation semantics.
		// Never score draft positions that cannot be emitted in this request.
		// Besides avoiding wasted MTP work near max_tokens, this keeps a
		// one-token completion to exactly one draft-head evaluation.
		draftLimit := min(mtpDraftTokens, min(cacheLen-pos, options.MaxTokens-len(generated)))
		drafts := qwen35MTPDraftGreedy(r.config, r.qwen35.MTP, cache.Qwen35MTP, token, pos,
			draftLimit, recent, options.Sampler.RepeatPenalty)
		currentToken := token
		seedPrepared := len(drafts) > 0
		for {
			if err := ctx.Err(); err != nil {
				return buildResult(), err
			}
			targetNext := verifyToken(currentToken, seedPrepared)
			seedPrepared = false
			if len(drafts) == 0 {
				nextToken, haveNextToken = targetNext, true
				break
			}
			candidate := drafts[0]
			drafts = drafts[1:]
			if targetNext != candidate {
				nextToken, haveNextToken = targetNext, true
				break
			}
			keepDecoding, canceled = emitToken(candidate)
			if canceled {
				return buildResult(), ErrGenerationCanceled
			}
			if !keepDecoding || len(generated) >= options.MaxTokens || pos >= cacheLen {
				break decode
			}
			currentToken = candidate
		}
	}
	if !flushStream(true) {
		return buildResult(), ErrGenerationCanceled
	}
	return buildResult(), nil
}

// generationCacheLen returns the cache length for a prompt with a positive
// model context. It performs the context cap before adding MaxTokens so an
// untrusted large max_tokens value cannot overflow an int and turn into a
// negative cache allocation.
func generationCacheLen(maxSeqLen, promptTokens, maxTokens int) int {
	remaining := maxSeqLen - promptTokens
	if maxTokens >= remaining-1 {
		return maxSeqLen
	}
	return promptTokens + maxTokens + 1
}

func recentTokenWindowInto(dst, tokens []uint32) []uint32 {
	start := max(0, len(tokens)-repeatPenaltyWindow)
	return append(dst[:0], tokens[start:]...)
}

func validUTF8PrefixLen(b []byte) int {
	if utf8.Valid(b) {
		return len(b)
	}
	for n := len(b) - 1; n >= 0 && len(b)-n <= utf8.UTFMax; n-- {
		if utf8.Valid(b[:n]) {
			return n
		}
	}
	return 0
}
