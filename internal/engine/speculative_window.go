package engine

import "time"

const (
	speculativeSerialPause = 12
	speculativeDraftCap    = 7
)

// speculativeWindow belongs to one generation. Keeping it off Runner avoids
// carrying one prompt's draft quality or a previous device load into another.
// Three fixed slots also keep adaptation independent of vocabulary/context size.
type speculativeWindow struct {
	maxIndex int
	index    int
	dwell    int
	misses   int
	pause    int
	probing  bool
	serial   float64
	sizes    [3]int
	stats    [3]speculativeWindowStats
}

type speculativeWindowStats struct {
	cost       float64 // elapsed nanoseconds per emitted token
	acceptance float64 // accepted candidates / drafted candidates
	samples    int
}

// speculativeWindowSample covers one complete speculative pass. Emitted counts
// the seed plus accepted candidates actually emitted by the caller; rejected,
// EOS and stop-string candidates contribute no emitted tokens. Rollback must be
// included: a high acceptance rate alone does not make speculation worthwhile.
// SerialPerToken is optional and must describe comparable target-only work.
type speculativeWindowSample struct {
	Drafted, Accepted, Emitted int
	Draft, Verify, Rollback    time.Duration
	SerialPerToken             time.Duration
	// Requested identifies the selected window when history lookup supplies
	// fewer candidates. Zero preserves callers that always fill their window.
	Requested int
}

func newSpeculativeWindow(maxDraft int) speculativeWindow {
	w := speculativeWindow{maxIndex: -1}
	cap := min(maxDraft, speculativeDraftCap)
	for _, size := range [...]int{1, 3, 7} {
		if size > cap {
			break
		}
		w.maxIndex++
		w.sizes[w.maxIndex] = size
	}
	if w.maxIndex >= 0 && w.sizes[w.maxIndex] < cap {
		w.maxIndex++
		w.sizes[w.maxIndex] = cap
	}
	if w.maxIndex >= 1 {
		w.index = 1 // Three drafts plus the seed fill a four-position verifier.
	}
	return w
}

// limit selects 1/3/7 drafts, plus the exact user cap when smaller. Including
// the seed, the regular sizes fill the efficient 2/4/8-position verifier paths.
// Tail windows never increase the user's cap or the available context/output.
// A zero result can also request a bounded interval of target-only decoding.
func (w *speculativeWindow) limit(contextSlots, outputSlots int) int {
	if w.maxIndex < 0 {
		return 0
	}
	n := min(w.sizes[w.index], min(contextSlots, outputSlots))
	if n <= 0 {
		return 0
	}
	if w.pause > 0 {
		w.pause--
		return 0
	}
	for i := w.index; i >= 0; i-- {
		if w.sizes[i] <= n {
			return w.sizes[i]
		}
	}
	return 0
}

func (w *speculativeWindow) observe(s speculativeWindowSample) {
	if w.maxIndex < 0 || s.Drafted <= 0 || s.Accepted < 0 || s.Accepted > s.Drafted ||
		s.Emitted <= 0 || s.Draft < 0 || s.Verify < 0 || s.Rollback < 0 {
		return
	}
	requested := s.Requested
	if requested == 0 {
		requested = s.Drafted
	}
	if requested < s.Drafted || requested > speculativeDraftCap {
		return
	}
	index := -1
	for i := range w.stats {
		if requested == w.sizes[i] {
			index = i
			break
		}
	}
	if index < 0 || index > w.maxIndex {
		return
	}
	// Add as floats so three valid durations cannot overflow time.Duration.
	elapsed := float64(s.Draft) + float64(s.Verify) + float64(s.Rollback)
	stat := &w.stats[index]
	if elapsed > 0 {
		cost := elapsed / float64(s.Emitted)
		acceptance := float64(s.Accepted) / float64(s.Drafted)
		if stat.samples == 0 {
			stat.cost, stat.acceptance = cost, acceptance
		} else {
			// Discard cold timing before making a cost-based decision.
			if stat.samples == 1 {
				stat.cost = cost
			} else {
				stat.cost += (cost - stat.cost) * 0.5
			}
			stat.acceptance += (acceptance - stat.acceptance) * 0.5
		}
		stat.samples++
	}
	if s.SerialPerToken > 0 {
		if w.serial == 0 {
			w.serial = float64(s.SerialPerToken)
		} else {
			w.serial += (float64(s.SerialPerToken) - w.serial) * 0.5
		}
	}
	// A tail clipped by max_tokens/context is not evidence that the chosen
	// larger window became unprofitable, so it cannot change the policy.
	if index != w.index {
		return
	}
	if s.Accepted == 0 {
		w.misses++
	} else {
		w.misses = 0
	}
	if w.misses >= 2 {
		// Repeated complete misses are a draft-quality signal even while
		// kernel setup makes early timing samples unreliable. Do not pay for
		// three wide blocks and three narrow blocks before backing off.
		if w.index > 0 {
			w.index--
		} else {
			w.pause = speculativeSerialPause
		}
		w.misses, w.dwell, w.probing = 0, 0, false
		return
	}
	// Coarse clocks can report zero for real work (notably tiny Windows
	// fixtures). Rejection counts remain useful; zero is not a free-cost
	// sample and must not drive window growth or timing-based backoff.
	if elapsed <= 0 {
		return
	}
	w.dwell++
	if w.probing {
		// A slow, high-acceptance smaller window may simply underfill the
		// verifier. One larger block measures that possibility; if it loses,
		// return immediately rather than spending three expensive probes.
		w.probing = false
		smaller := w.stats[w.index-1]
		if stat.acceptance < 0.50 || stat.cost > smaller.cost*1.10 ||
			(w.serial > 0 && stat.cost > w.serial*1.10) {
			w.index--
			w.dwell = 0
			return
		}
	}
	if w.dwell < 3 {
		return
	}

	slow := w.serial > 0 && stat.cost > w.serial*1.10
	if w.index > 0 {
		smaller := w.stats[w.index-1]
		if smaller.samples >= 3 && stat.cost > smaller.cost*1.10 {
			slow = true
		}
		if slow && stat.acceptance >= 0.85 && w.index < w.maxIndex && w.stats[w.index+1].samples == 0 {
			// P4 can lose to serial while P8 amortizes the same weight reads.
			// Explore an unmeasured larger size once before falling back to P2.
			w.index++
			w.dwell = 0
			w.probing = true
			return
		}
		if slow || stat.acceptance < 0.50 {
			w.index--
			w.dwell = 0
			return
		}
	} else if slow || stat.acceptance < 0.50 {
		// Even one candidate can cost more than target-only decoding. Give
		// the caller twelve serial steps, then measure the smallest size again
		// so changing draft quality or device load can recover within a request.
		w.pause = speculativeSerialPause
		w.dwell = 0
		return
	}
	if w.index >= w.maxIndex || stat.acceptance < 0.75 {
		return
	}
	larger := w.stats[w.index+1]
	knownSlower := larger.samples >= 3 && larger.cost > stat.cost*1.10
	// Revisit a previously slow size after twelve good passes. This matters
	// when the first sample paid setup costs or the device load has changed;
	// shrinking never permanently disables drafting or future exploration.
	if (slow || knownSlower) && w.dwell < 12 {
		return
	}
	w.index++
	w.dwell = 0
}
