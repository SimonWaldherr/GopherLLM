package gopherllm

import "unsafe"

const promptLookupHistoryTokens = 8192

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
	for gram := min(8, n-1); gram >= 4; gram-- {
		suffix := n - gram
		match, shortMatch := -1, -1
		for start := suffix - 1; start >= 0; start-- {
			matched := true
			for j := range gram {
				if at(start+j) != at(suffix+j) {
					matched = false
					break
				}
			}
			if !matched {
				continue
			}
			if n-start-gram >= maxDraft {
				match = start
				break
			}
			if shortMatch < 0 {
				shortMatch = start
			}
		}
		// A nearby overlapping match can contain only one known next token.
		// Prefer the youngest full continuation at this same n-gram length;
		// otherwise retain the youngest short match without extrapolating it.
		if match < 0 {
			match = shortMatch
		}
		if match < 0 {
			continue
		}
		count := min(maxDraft, n-match-gram)
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
			dst[i] = at(match + gram + i)
		}
		return dst
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
