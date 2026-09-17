package gopherllm

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Record both rows and callback boundaries: counting rows alone would miss a
// change in the original chunk partitions, which some kernels use as units.
type dispatchCoverage struct {
	rows    []atomic.Int32
	mu      sync.Mutex
	ranges  map[[2]int]int
	invalid bool
}

func newDispatchCoverage(rows int) *dispatchCoverage {
	return &dispatchCoverage{rows: make([]atomic.Int32, rows), ranges: make(map[[2]int]int)}
}

func (c *dispatchCoverage) runRows(start, end int) {
	c.mu.Lock()
	c.ranges[[2]int{start, end}]++
	valid := start >= 0 && start <= end && end <= len(c.rows)
	c.invalid = c.invalid || !valid
	c.mu.Unlock()
	if valid {
		for i := start; i < end; i++ {
			c.rows[i].Add(1)
		}
	}
}

func (c *dispatchCoverage) check(t *testing.T, chunks int) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.invalid {
		t.Errorf("dispatch produced an invalid range: %v", c.ranges)
	}
	if len(c.ranges) != chunks {
		t.Errorf("got %d distinct ranges, want %d: %v", len(c.ranges), chunks, c.ranges)
	}
	for chunk := range chunks {
		r := [2]int{len(c.rows) * chunk / chunks, len(c.rows) * (chunk + 1) / chunks}
		if got := c.ranges[r]; got != 1 {
			t.Errorf("range %v ran %d times, want once", r, got)
		}
	}
	for row := range c.rows {
		if got := c.rows[row].Load(); got != 1 {
			t.Errorf("row %d ran %d times, want once", row, got)
			break
		}
	}
}

func startCheckedDispatch(fn func()) <-chan struct{} {
	done := make(chan struct{})
	go func() {
		fn()
		close(done)
	}()
	return done
}

func awaitCheckedDispatch(t *testing.T, done <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func TestDynamicRowDispatchExactCoverage(t *testing.T) {
	previous := oversubscribeDispatch.Load()
	t.Cleanup(func() { oversubscribeDispatch.Store(previous) })
	for _, tc := range []struct {
		name               string
		rows, threads      int
		oversubscribe      bool
		allowOversubscribe bool
		chunks             int
	}{
		{"few_units", 3, 3, true, true, 3},
		{"below_threshold", 383, 3, true, true, 3},
		{"at_threshold", 384, 3, true, true, 24},
		{"uneven", 401, 3, true, true, 24},
		{"two_participants", 513, 2, true, true, 16},
		{"oversubscribe_disabled", 401, 3, false, true, 3},
		{"batched", 401, 3, true, false, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			oversubscribeDispatch.Store(tc.oversubscribe)
			for _, useTask := range []bool{false, true} {
				name := "callback"
				if useTask {
					name = "task"
				}
				t.Run(name, func(t *testing.T) {
					coverage := newDispatchCoverage(tc.rows)
					done := startCheckedDispatch(func() {
						if useTask {
							dispatchParallelTaskMode(8, tc.threads, tc.rows, tc.allowOversubscribe, coverage)
						} else {
							dispatchParallelMode(8, tc.threads, tc.rows, tc.allowOversubscribe, coverage.runRows)
						}
					})
					awaitCheckedDispatch(t, done, "row coverage dispatch")
					coverage.check(t, tc.chunks)
				})
			}
		})
	}
}

func TestDynamicRowDispatchBalancesBlockedChunk(t *testing.T) {
	previous := oversubscribeDispatch.Swap(true)
	t.Cleanup(func() { oversubscribeDispatch.Store(previous) })
	const rows, chunks = 257, 16
	coverage := newDispatchCoverage(rows)
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	remainingDone := make(chan struct{})
	var completed atomic.Int32
	done := startCheckedDispatch(func() {
		dispatchParallelMode(2, 2, rows, true, func(start, end int) {
			if start == 0 {
				<-release
				coverage.runRows(start, end)
				return
			}
			coverage.runRows(start, end)
			if completed.Add(1) == chunks-1 {
				close(remainingDone)
			}
		})
	})
	// Whichever participant claims the first chunk is blocked. Another
	// participant must keep claiming work until every remaining chunk is done.
	awaitCheckedDispatch(t, remainingDone, "unblocked participant to drain remaining chunks")
	select {
	case <-done:
		t.Fatal("dispatch returned before its blocked chunk completed")
	default:
	}
	unblock()
	awaitCheckedDispatch(t, done, "balanced dispatch")
	coverage.check(t, chunks)
}

type gatedDispatchCoverage struct {
	*dispatchCoverage
	started chan struct{}
	release <-chan struct{}
	once    sync.Once
}

func (c *gatedDispatchCoverage) runRows(start, end int) {
	c.once.Do(func() { close(c.started) })
	<-c.release
	c.dispatchCoverage.runRows(start, end)
}

func TestDynamicRowDispatchIsolationAcrossPoolResize(t *testing.T) {
	previous := oversubscribeDispatch.Swap(true)
	t.Cleanup(func() { oversubscribeDispatch.Store(previous) })
	releaseA, releaseB := make(chan struct{}), make(chan struct{})
	var releaseAOnce, releaseBOnce sync.Once
	unblockA := func() { releaseAOnce.Do(func() { close(releaseA) }) }
	unblockB := func() { releaseBOnce.Do(func() { close(releaseB) }) }
	defer unblockA()
	defer unblockB()
	a := &gatedDispatchCoverage{dispatchCoverage: newDispatchCoverage(271), started: make(chan struct{}), release: releaseA}
	b := &gatedDispatchCoverage{dispatchCoverage: newDispatchCoverage(389), started: make(chan struct{}), release: releaseB}
	doneA := startCheckedDispatch(func() { dispatchParallelTaskMode(8, 2, 271, true, a) })
	awaitCheckedDispatch(t, a.started, "first dispatch to start")
	rowPoolMu.Lock()
	oldPool := rowPool
	rowPoolMu.Unlock()
	doneB := startCheckedDispatch(func() { dispatchParallelMode(8, 3, 389, true, b.runRows) })
	awaitCheckedDispatch(t, b.started, "second dispatch to start")
	rowPoolMu.Lock()
	shared := oldPool != nil && rowPool == oldPool && oldPool.active == 2
	rowPoolMu.Unlock()
	if !shared {
		t.Fatal("concurrent dispatches did not hold separate leases on the same pool")
	}

	// Replace the pool while both dispatches are paused. A new dispatch must
	// have independent chunk state and complete without releasing either one.
	c := newDispatchCoverage(257)
	doneC := startCheckedDispatch(func() { dispatchParallelTaskMode(4, 2, 257, true, c) })
	awaitCheckedDispatch(t, doneC, "replacement-pool dispatch")
	c.check(t, 16)
	select {
	case <-oldPool.stop:
		t.Fatal("old pool stopped while two dispatches still held leases")
	default:
	}

	unblockA()
	awaitCheckedDispatch(t, doneA, "first old-pool dispatch")
	a.check(t, 16)
	select {
	case <-oldPool.stop:
		t.Fatal("old pool stopped while the second dispatch still held its lease")
	default:
	}
	unblockB()
	awaitCheckedDispatch(t, doneB, "second old-pool dispatch")
	b.check(t, 24)
	awaitCheckedDispatch(t, oldPool.stop, "retired pool to stop after its last lease")
}
