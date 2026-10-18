package gopherllm

import "unsafe"

const (
	promptLookupHistoryTokens = 8192
	promptLookupMaxNgram      = 64
)

// promptLookupDraft proposes known continuations of the current history suffix.
// generated already includes the seed token. The target must still verify every
// proposal; matching a repeated passage is not permission to emit it directly.
func promptLookupDraft(prompt, generated []uint32, maxDraft int, dst []uint32) []uint32 {
	dst = dst[:0]
	if maxDraft <= 0 || len(generated) == 0 {
		return dst
	}
	// Work on a virtual prompt+generated suffix. No concatenation or indexing
	// over the unbounded prompt is needed, even after a very long prefill.
	generatedLen := min(len(generated), promptLookupHistoryTokens)
	promptLen := min(len(prompt), promptLookupHistoryTokens-generatedLen)
	n := promptLen + generatedLen
	at := func(i int) uint32 {
		if i < promptLen {
			return prompt[len(prompt)-promptLen+i]
		}
		return generated[len(generated)-generatedLen+i-promptLen]
	}
	maxGram := min(promptLookupMaxNgram, n-1)
	if n-maxDraft >= 4 {
		// An overlapping match in a short periodic history must not win only
		// because its longer suffix leaves one known continuation token. Bound
		// the context to a length that can still have a full earlier window.
		maxGram = min(maxGram, n-maxDraft)
	}
	bestEnd, bestGram, bestFull := -1, 0, false
	for end := n - 2; end >= 3; end-- {
		// Compare each earlier ending once. Extending beyond eight tokens
		// retains distinguishing line numbers and surrounding code, while a
		// backward scan bounds the work to historyLen*maxGram comparisons.
		gram, bound := 0, min(maxGram, end+1)
		for gram < bound && at(end-gram) == at(n-1-gram) {
			gram++
		}
		if gram < 4 {
			continue
		}
		full := n-end-1 >= maxDraft
		if gram > bestGram || gram == bestGram && full && !bestFull {
			bestEnd, bestGram, bestFull = end, gram, full
			if gram == maxGram && full {
				break // Already the youngest full match of maximal length.
			}
		}
	}
	if bestEnd < 0 {
		return dst
	}
	count := min(maxDraft, n-bestEnd-1)
	if cap(dst) < count {
		dst = make([]uint32, count)
	} else {
		dst = dst[:count]
		// A caller may accidentally offer input storage as scratch. Keep
		// the function's read-only history contract even in that case.
		if lookupSlicesOverlap(dst, prompt) || lookupSlicesOverlap(dst, generated) {
			dst = make([]uint32, count)
		}
	}
	for i := range count {
		dst[i] = at(bestEnd + 1 + i)
	}
	return dst
}

func lookupSlicesOverlap(a, b []uint32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	// Compare ranges only; no pointer is reconstructed or used to span slices.
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return aStart < bStart+uintptr(len(b))*4 && bStart < aStart+uintptr(len(a))*4
}
