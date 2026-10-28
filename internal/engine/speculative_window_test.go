package engine

import (
	"slices"
	"testing"
	"time"
)

func TestSpeculativeWindowBounds(t *testing.T) {
	for _, maxDraft := range []int{-1, 0, 1, 2, 3, 4, 7, 8, 32} {
		w := newSpeculativeWindow(maxDraft)
		for contextSlots := -1; contextSlots <= 10; contextSlots++ {
			for outputSlots := -1; outputSlots <= 10; outputSlots++ {
				n := w.limit(contextSlots, outputSlots)
				if maxDraft <= 0 || contextSlots <= 0 || outputSlots <= 0 {
					if n != 0 {
						t.Fatalf("max=%d context=%d output=%d: got %d", maxDraft, contextSlots, outputSlots, n)
					}
					continue
				}
				if n != 1 && n != 3 && n != 7 && n != maxDraft {
					t.Fatalf("non-kernel size %d", n)
				}
				if n > maxDraft || n > contextSlots || n > outputSlots || n > speculativeDraftCap {
					t.Fatalf("max=%d context=%d output=%d: got %d", maxDraft, contextSlots, outputSlots, n)
				}
				if n > 3 {
					t.Fatalf("initial size %d exceeds conservative bootstrap", n)
				}
			}
		}
	}
}

func TestSpeculativeWindowZeroDurationStillBacksOffRejectedDrafts(t *testing.T) {
	w := newSpeculativeWindow(7)
	for pass, want := range []int{3, 1, 1, 0} {
		n := w.limit(100, 100)
		w.observe(speculativeWindowSample{Requested: n, Drafted: n, Emitted: 1})
		if got := w.limit(100, 100); got != want {
			t.Fatalf("zero-duration rejection %d: window=%d, want %d", pass+1, got, want)
		}
	}
	for _, stat := range w.stats {
		if stat.samples != 0 || stat.cost != 0 {
			t.Fatal("zero-duration work was recorded as a timing sample")
		}
	}
	// The check above consumed one pause step. The remaining pause must stay
	// bounded, then drafting resumes at the smallest window.
	for range speculativeSerialPause - 1 {
		if w.limit(100, 100) != 0 {
			t.Fatal("serial pause ended early")
		}
	}
	if w.limit(100, 100) != 1 {
		t.Fatal("serial pause did not resume drafting")
	}
	for range 8 {
		w.observe(speculativeWindowSample{Drafted: 1, Accepted: 1, Emitted: 2})
	}
	if w.index != 0 || w.stats[0].samples != 0 {
		t.Fatal("zero-duration accepted work caused timing-based growth")
	}
}

func TestSpeculativeWindowIncludesSeedInVerifierSize(t *testing.T) {
	for _, tc := range []struct {
		maxDraft int
		want     []int
	}{
		{1, []int{1}},
		{2, []int{1, 2}},
		{3, []int{1, 3}},
		{4, []int{1, 3, 4}},
		{5, []int{1, 3, 5}},
		{6, []int{1, 3, 6}},
		{7, []int{1, 3, 7}},
		{8, []int{1, 3, 7}},
	} {
		w := newSpeculativeWindow(tc.maxDraft)
		if !slices.Equal(w.sizes[:w.maxIndex+1], tc.want) {
			t.Fatalf("cap=%d sizes=%v, want %v", tc.maxDraft, w.sizes, tc.want)
		}
		if got, want := w.limit(100, 100), tc.want[min(1, len(tc.want)-1)]; got != want {
			t.Fatalf("cap=%d starts at %d, want %d", tc.maxDraft, got, want)
		}
	}
	w := newSpeculativeWindow(7)
	for i, targetRows := range []int{2, 4, 8} {
		if got := w.sizes[i] + 1; got != targetRows {
			t.Fatalf("draft arm %d + seed produces %d target rows, want %d", w.sizes[i], got, targetRows)
		}
	}
	for range 3 {
		w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
	}
	for _, tc := range []struct{ slots, want int }{{1, 1}, {2, 1}, {3, 3}, {4, 3}, {6, 3}, {7, 7}, {8, 7}} {
		if got := w.limit(tc.slots, 100); got != tc.want {
			t.Fatalf("context slots=%d produced %d drafts, want %d", tc.slots, got, tc.want)
		}
		if got := w.limit(100, tc.slots); got != tc.want {
			t.Fatalf("output slots=%d produced %d drafts, want %d", tc.slots, got, tc.want)
		}
	}
}

func TestSpeculativeWindowExactCandidateCap(t *testing.T) {
	for _, maxDraft := range []int{1, 2, 3, 4, 5, 6, 7, 8, 32} {
		w := newSpeculativeWindow(maxDraft)
		for range 20 {
			n := w.limit(100, 100)
			w.observe(speculativeWindowSample{Drafted: n, Accepted: n, Emitted: n + 1, Verify: time.Millisecond})
		}
		if got, want := w.limit(100, 100), min(maxDraft, 7); got != want {
			t.Fatalf("max=%d converged to %d, want %d", maxDraft, got, want)
		}
	}
}

func TestSpeculativeWindowFindsFasterLargerWindow(t *testing.T) {
	w := newSpeculativeWindow(8)
	for range 3 {
		n := w.limit(100, 100)
		if n != 3 {
			t.Fatalf("window = %d, want 3", n)
		}
		w.observe(speculativeWindowSample{
			Drafted: n, Accepted: n, Emitted: n + 1,
			Draft: 100 * time.Microsecond, Verify: time.Millisecond,
			SerialPerToken: time.Millisecond,
		})
	}
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("fast accepted work did not reach seven drafts: %d", got)
	}
	for range 20 {
		w.observe(speculativeWindowSample{Drafted: 7, Accepted: 7, Emitted: 8, Verify: time.Millisecond})
	}
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("size exceeded or fell below useful maximum: %d", got)
	}
}

func TestSpeculativeWindowRollbackCostAndRejection(t *testing.T) {
	for _, sample := range []speculativeWindowSample{
		{Drafted: 3, Accepted: 0, Emitted: 1, Draft: time.Millisecond, Verify: time.Millisecond},
		{Drafted: 3, Accepted: 2, Emitted: 3, Verify: time.Millisecond, Rollback: 7 * time.Millisecond, SerialPerToken: time.Millisecond},
	} {
		w := newSpeculativeWindow(8)
		for range 3 {
			w.observe(sample)
		}
		if got := w.limit(100, 100); got != 1 {
			t.Fatalf("unprofitable sample %+v retained window %d", sample, got)
		}
	}
}

func TestSpeculativeWindowExploresHighAcceptanceLargerSizeOnce(t *testing.T) {
	for _, largerCost := range []time.Duration{6 * time.Millisecond, 16 * time.Millisecond} {
		w := newSpeculativeWindow(7)
		slowerSmall := speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: 6 * time.Millisecond, SerialPerToken: time.Millisecond}
		for range 3 {
			w.observe(slowerSmall)
		}
		if got := w.limit(100, 100); got != 7 || !w.probing {
			t.Fatalf("high-acceptance P4 did not test unmeasured P8: size=%d probing=%t", got, w.probing)
		}
		w.observe(speculativeWindowSample{Drafted: 7, Accepted: 7, Emitted: 8, Verify: largerCost, SerialPerToken: time.Millisecond})
		if w.probing || w.stats[2].samples != 1 {
			t.Fatal("controlled exploration did not finish after one larger block")
		}
		if largerCost == 6*time.Millisecond {
			if got := w.limit(100, 100); got != 7 {
				t.Fatalf("measured faster P8 was discarded: %d", got)
			}
		} else {
			if got := w.limit(100, 100); got != 3 {
				t.Fatalf("measured slower P8 was not abandoned immediately: %d", got)
			}
			for range 3 {
				w.observe(slowerSmall)
			}
			if got := w.limit(100, 100); got != 1 || w.probing {
				t.Fatalf("known slow larger size was repeatedly explored: size=%d probing=%t", got, w.probing)
			}
		}
	}
}

func TestSpeculativeWindowNoTimingWhiplash(t *testing.T) {
	w := newSpeculativeWindow(8)
	w.observe(speculativeWindowSample{
		Drafted: 3, Accepted: 0, Emitted: 1,
		Verify: time.Second, SerialPerToken: time.Millisecond,
	})
	if got := w.limit(100, 100); got != 3 {
		t.Fatalf("one cold sample changed window: %d", got)
	}
	// Setup work in the first pass must eventually be forgotten. Even after
	// shrinking, successful smaller passes can re-explore the faster sizes.
	for range 60 {
		n := w.limit(100, 100)
		w.observe(speculativeWindowSample{
			Drafted: n, Accepted: n, Emitted: n + 1,
			Verify: time.Millisecond, SerialPerToken: time.Millisecond,
		})
	}
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("cold sample prevented recovery: %d", got)
	}
}

func TestSpeculativeWindowRepeatedMissesBackOffBeforeTimingSettles(t *testing.T) {
	w := newSpeculativeWindow(7)
	miss := speculativeWindowSample{Drafted: 3, Emitted: 1, Verify: time.Second}
	w.observe(miss)
	if w.limit(100, 100) != 3 {
		t.Fatal("one miss prematurely changed the draft window")
	}
	w.observe(miss)
	if w.limit(100, 100) != 1 {
		t.Fatal("two complete misses retained the wide window")
	}
	miss.Drafted = 1
	w.observe(miss)
	w.observe(miss)
	for range speculativeSerialPause {
		if w.limit(100, 100) != 0 {
			t.Fatal("repeated smallest-window misses did not pause drafting")
		}
	}
	if w.limit(100, 100) != 1 {
		t.Fatal("drafting did not resume after the bounded pause")
	}
}

func TestSpeculativeWindowRejectsSlowLargerArmAndReprobes(t *testing.T) {
	w := newSpeculativeWindow(8)
	for range 3 {
		w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
	}
	for range 3 {
		w.observe(speculativeWindowSample{Drafted: 7, Accepted: 7, Emitted: 8, Verify: 10 * time.Millisecond})
	}
	if got := w.limit(100, 100); got != 3 {
		t.Fatalf("slower seven-draft arm retained: %d", got)
	}
	for range 11 {
		w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
		if got := w.limit(100, 100); got != 3 {
			t.Fatalf("reprobed before cooldown: %d", got)
		}
	}
	w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("previously slow arm was never revisited: %d", got)
	}
}

func TestSpeculativeWindowTailDoesNotChangeChoice(t *testing.T) {
	w := newSpeculativeWindow(8)
	for range 3 {
		w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
	}
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("bootstrap did not reach seven: %d", got)
	}
	for range 20 {
		w.observe(speculativeWindowSample{Drafted: 1, Accepted: 0, Emitted: 1, Verify: time.Second})
	}
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("clipped tail changed policy: %d", got)
	}
}

func TestSpeculativeWindowInvalidSamplesAndRequestIsolation(t *testing.T) {
	w := newSpeculativeWindow(8)
	before := w
	for _, sample := range []speculativeWindowSample{
		{Drafted: 0, Emitted: 1, Verify: time.Millisecond},
		{Drafted: 2, Accepted: 2, Emitted: 3, Verify: time.Millisecond},
		{Drafted: 3, Accepted: -1, Emitted: 1, Verify: time.Millisecond},
		{Drafted: 3, Accepted: 4, Emitted: 1, Verify: time.Millisecond},
		{Drafted: 3, Accepted: 3, Emitted: 0, Verify: time.Millisecond},
		{Drafted: 3, Accepted: 3, Emitted: 4},
		{Drafted: 3, Accepted: 3, Emitted: 4, Verify: -time.Millisecond},
		{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond, Rollback: -1},
	} {
		w.observe(sample)
	}
	if w != before {
		t.Fatalf("invalid samples changed state: before=%+v after=%+v", before, w)
	}
	for range 3 {
		w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
	}
	fresh := newSpeculativeWindow(8)
	if w.limit(100, 100) != 7 || fresh.limit(100, 100) != 3 {
		t.Fatal("request windows share learned state")
	}
	disabled := newSpeculativeWindow(0)
	disabled.observe(speculativeWindowSample{Drafted: 1, Accepted: 1, Emitted: 2, Verify: time.Millisecond})
	if disabled.limit(100, 100) != 0 {
		t.Fatal("observations enabled unrequested speculation")
	}
}

func TestSpeculativeWindowPausesUnprofitableSmallestSize(t *testing.T) {
	for _, sample := range []speculativeWindowSample{
		{Drafted: 1, Accepted: 0, Emitted: 1, Verify: time.Millisecond},
		{Drafted: 1, Accepted: 1, Emitted: 2, Verify: time.Millisecond, Rollback: 2 * time.Millisecond, SerialPerToken: time.Millisecond},
	} {
		w := newSpeculativeWindow(1)
		observations := 3
		if sample.Accepted == 0 {
			observations = 2
		}
		for range observations - 1 {
			w.observe(sample)
			if got := w.limit(100, 100); got != 1 {
				t.Fatalf("paused before %d observations: %d", observations, got)
			}
		}
		w.observe(sample)
		for step := range speculativeSerialPause {
			if got := w.limit(100, 100); got != 0 {
				t.Fatalf("serial step %d returned %d", step, got)
			}
		}
		if got := w.limit(100, 100); got != 1 {
			t.Fatalf("smallest size was not retried after pause: %d", got)
		}
		for range 3 {
			w.observe(speculativeWindowSample{Drafted: 1, Accepted: 1, Emitted: 2, Verify: time.Millisecond, SerialPerToken: time.Millisecond})
		}
		if got := w.limit(100, 100); got != 1 {
			t.Fatalf("profitable retry remained disabled: %d", got)
		}
	}
}

func TestSpeculativeWindowPauseBoundsAndIsolation(t *testing.T) {
	w := newSpeculativeWindow(1)
	bad := speculativeWindowSample{Drafted: 1, Emitted: 1, Verify: time.Millisecond}
	for range 3 {
		w.observe(bad)
	}
	for range 20 {
		if w.limit(0, 100) != 0 || w.limit(100, 0) != 0 {
			t.Fatal("exhausted request received a draft window")
		}
	}
	if w.pause != speculativeSerialPause {
		t.Fatal("exhausted tail consumed serial-pause steps")
	}
	fresh := newSpeculativeWindow(1)
	if fresh.limit(100, 100) != 1 {
		t.Fatal("pause leaked into another request")
	}
	for range speculativeSerialPause {
		if got := w.limit(100, 100); got != 0 {
			t.Fatalf("pause ended early: %d", got)
		}
	}
	if w.limit(100, 100) != 1 {
		t.Fatal("pause permanently disabled drafting")
	}
	// A second bad probe interval may pause again, without becoming permanent.
	for range 3 {
		w.observe(bad)
	}
	for range speculativeSerialPause {
		if w.limit(100, 100) != 0 {
			t.Fatal("repeated bad probes did not pause again")
		}
	}
	if w.limit(100, 100) != 1 {
		t.Fatal("repeated pause never resumed")
	}
}

func TestSpeculativeWindowPauseRequiresMeasuredSlowdown(t *testing.T) {
	for _, baseline := range []time.Duration{0, time.Millisecond} {
		w := newSpeculativeWindow(1)
		for range 20 {
			// Exactly ten percent overhead remains within the hysteresis.
			w.observe(speculativeWindowSample{
				Drafted: 1, Accepted: 1, Emitted: 2,
				Verify: 2 * time.Millisecond, Rollback: 200 * time.Microsecond,
				SerialPerToken: baseline,
			})
			if got := w.limit(100, 100); got != 1 {
				t.Fatalf("baseline=%s paused without measured >10%% slowdown: %d", baseline, got)
			}
		}
	}
}

func TestSpeculativeWindowCanGrowAfterSerialPause(t *testing.T) {
	w := newSpeculativeWindow(7)
	for range 4 {
		n := w.limit(100, 100)
		w.observe(speculativeWindowSample{Drafted: n, Emitted: 1, Verify: 2 * time.Millisecond, SerialPerToken: time.Millisecond})
	}
	for range speculativeSerialPause {
		if w.limit(100, 100) != 0 {
			t.Fatal("bad smallest arm did not enter serial pause")
		}
	}
	if got := w.limit(100, 100); got != 1 {
		t.Fatalf("retry did not start at smallest size: %d", got)
	}
	for range 30 {
		n := w.limit(100, 100)
		w.observe(speculativeWindowSample{Drafted: n, Accepted: n, Emitted: n + 1, Verify: time.Millisecond, SerialPerToken: time.Millisecond})
	}
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("profitable work did not recover the configured cap: %d", got)
	}
}

func TestSpeculativeWindowShortHistoryTrainsRequestedSize(t *testing.T) {
	for _, drafted := range []int{1, 2, 3} {
		w := newSpeculativeWindow(7)
		for range 3 {
			w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
		}
		if got := w.limit(100, 100); got != 7 {
			t.Fatalf("drafted=%d: did not reach requested window seven: %d", drafted, got)
		}
		for range 3 {
			w.observe(speculativeWindowSample{
				Requested: 7, Drafted: drafted, Accepted: drafted, Emitted: drafted + 1,
				Verify:         time.Duration(drafted+1) * 1500 * time.Microsecond,
				SerialPerToken: time.Millisecond,
			})
		}
		if got := w.limit(100, 100); got != 3 {
			t.Fatalf("drafted=%d: expensive short history retained window %d", drafted, got)
		}
		if stat := w.stats[2]; stat.samples != 3 || stat.acceptance != 1 || stat.cost != float64(1500*time.Microsecond) {
			t.Fatalf("drafted=%d: requested arm did not retain actual acceptance/cost: %+v", drafted, stat)
		}
		// The same short-history behavior must continue shrinking to one and
		// then enter a bounded serial pause even with perfect acceptance.
		for range 6 {
			requested := w.limit(100, 100)
			actual := min(drafted, requested)
			w.observe(speculativeWindowSample{
				Requested: requested, Drafted: actual, Accepted: actual, Emitted: actual + 1,
				Verify:         time.Duration(actual+1) * 1500 * time.Microsecond,
				SerialPerToken: time.Millisecond,
			})
		}
		for step := range speculativeSerialPause {
			if got := w.limit(100, 100); got != 0 {
				t.Fatalf("drafted=%d: did not pause at serial step %d: %d", drafted, step, got)
			}
		}
		if got := w.limit(100, 100); got != 1 {
			t.Fatalf("drafted=%d: did not resume smallest probe: %d", drafted, got)
		}
	}
}

func TestSpeculativeWindowRequestedTailAndValidation(t *testing.T) {
	w := newSpeculativeWindow(7)
	before := w
	for _, sample := range []speculativeWindowSample{
		{Requested: -1, Drafted: 1, Emitted: 1, Verify: time.Millisecond},
		{Requested: 1, Drafted: 2, Emitted: 1, Verify: time.Millisecond},
		{Requested: 2, Drafted: 1, Emitted: 1, Verify: time.Millisecond},
		{Requested: 4, Drafted: 2, Emitted: 1, Verify: time.Millisecond},
		{Requested: 8, Drafted: 2, Emitted: 1, Verify: time.Millisecond},
		{Requested: 9, Drafted: 2, Emitted: 1, Verify: time.Millisecond},
	} {
		w.observe(sample)
	}
	if w != before {
		t.Fatalf("invalid requested sizes changed state: before=%+v after=%+v", before, w)
	}
	for range 3 {
		w.observe(speculativeWindowSample{Drafted: 3, Accepted: 3, Emitted: 4, Verify: time.Millisecond})
	}
	for range 20 {
		// Context/output clipping requested three although the policy selected
		// seven; history supplied just one. That tail cannot change the policy.
		w.observe(speculativeWindowSample{Requested: 3, Drafted: 1, Emitted: 1, Verify: time.Second})
	}
	if got := w.limit(100, 100); got != 7 {
		t.Fatalf("requested tail changed selected window: %d", got)
	}
}
