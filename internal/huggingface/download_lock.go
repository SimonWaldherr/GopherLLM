package huggingface

import (
	"context"
	"sync"
)

// Different artifacts can share the same content-addressed blob. Coordinate
// writers within this process so concurrent requests cannot truncate a resume
// file or race its rename. Entries disappear after the last waiter leaves.
var hfBlobLocks = struct {
	sync.Mutex
	entries map[string]*hfBlobLock
}{entries: make(map[string]*hfBlobLock)}

type hfBlobLock struct {
	token chan struct{}
	users int
}

func lockHFBlob(ctx context.Context, path string) (func(), error) {
	hfBlobLocks.Lock()
	entry := hfBlobLocks.entries[path]
	if entry == nil {
		entry = &hfBlobLock{token: make(chan struct{}, 1)}
		hfBlobLocks.entries[path] = entry
	}
	entry.users++
	hfBlobLocks.Unlock()
	leave := func() {
		hfBlobLocks.Lock()
		entry.users--
		if entry.users == 0 {
			delete(hfBlobLocks.entries, path)
		}
		hfBlobLocks.Unlock()
	}
	select {
	case entry.token <- struct{}{}:
		return func() { <-entry.token; leave() }, nil
	case <-ctx.Done():
		leave()
		return nil, ctx.Err()
	}
}
