package gopherllm

import (
	"testing"
	"time"
)

type blockingRowTask struct {
	started chan<- struct{}
	release <-chan struct{}
}

func (t *blockingRowTask) runRows(_, _ int) {
	t.started <- struct{}{}
	<-t.release
}

type noOpRowTask struct{}

func (noOpRowTask) runRows(_, _ int) {}

func TestRowWorkerPoolRetiresOnlyAfterInFlightDispatch(t *testing.T) {
	previousThreads := configuredThreads.Load()
	defer configuredThreads.Store(previousThreads)
	previousOversubscribe := oversubscribeDispatch.Load()
	defer oversubscribeDispatch.Store(previousOversubscribe)

	SetNumThreads(2)
	// Force enough chunks that, after the caller and both workers have started
	// one task each, the old pool still owns queued work. The original bug
	// required precisely that state: stopping a retired pool at this point can
	// strand those jobs before their WaitGroup.Done calls.
	oversubscribeDispatch.Store(true)
	started := make(chan struct{}, 16)
	release := make(chan struct{})
	done := make(chan struct{})
	go func() {
		parallelRowsTask(256, &blockingRowTask{started: started, release: release})
		close(done)
	}()
	for range 3 { // caller + both old-pool workers
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("old worker pool did not start caller and both worker chunks")
		}
	}

	rowPoolMu.Lock()
	oldPool := rowPool
	rowPoolMu.Unlock()
	if oldPool == nil {
		t.Fatal("missing active worker pool")
	}
	if len(oldPool.jobs) == 0 {
		t.Fatal("test setup error: old worker pool has no queued jobs")
	}

	// Trigger a resize while an old-pool dispatch owns queued work. A retiring
	// pool must not stop workers yet: doing so can strand a job before it calls
	// its WaitGroup.Done and hang the original caller.
	SetNumThreads(3)
	parallelChunksTask(3, noOpRowTask{})
	select {
	case <-oldPool.stop:
		t.Fatal("retired worker pool stopped before its dispatch completed")
	default:
	}

	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("in-flight dispatch did not finish after worker-pool resize")
	}
	select {
	case <-oldPool.stop:
	case <-time.After(time.Second):
		t.Fatal("retired worker pool was not stopped after its last lease")
	}
}
