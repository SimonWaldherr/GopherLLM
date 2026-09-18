package gopherllm

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrRunnerBusy means the per-runner bound of 64 active/waiting inference
// operations was reached. Hosts can impose a smaller admission bound.
var ErrRunnerBusy = errors.New("runner inference capacity exceeded")

type contextMutex struct {
	once  sync.Once
	token chan struct{}
}

func (m *contextMutex) init()   { m.once.Do(func() { m.token = make(chan struct{}, 1) }) }
func (m *contextMutex) Lock()   { _ = m.LockContext(context.Background()) }
func (m *contextMutex) Unlock() { <-m.token }
func (m *contextMutex) LockContext(ctx context.Context) error {
	m.init()
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case m.token <- struct{}{}:
		if err := ctx.Err(); err != nil {
			m.Unlock()
			return err
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *Runner) acquireInference(ctx context.Context) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r.inferencePending.Add(1) > 64 {
		r.inferencePending.Add(-1)
		return nil, ErrRunnerBusy
	}
	fail := func(err error) (func(), error) { r.inferencePending.Add(-1); return nil, err }
	// Waiting for a model swap/Close must not create an uncancellable goroutine.
	// The fast path allocates no timer; only a contended lifecycle lock polls.
	if !r.modelMu.TryRLock() {
		timer := time.NewTicker(time.Millisecond)
		defer timer.Stop()
		for {
			select {
			case <-ctx.Done():
				return fail(ctx.Err())
			case <-timer.C:
			}
			if r.modelMu.TryRLock() {
				break
			}
		}
	}
	if r.closed {
		r.modelMu.RUnlock()
		return fail(ErrRunnerClosed)
	}
	if err := r.genLock.LockContext(ctx); err != nil {
		r.modelMu.RUnlock()
		return fail(err)
	}
	return func() { r.genLock.Unlock(); r.modelMu.RUnlock(); r.inferencePending.Add(-1) }, nil
}
