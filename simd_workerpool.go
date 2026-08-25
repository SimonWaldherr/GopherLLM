package gopherllm

import "sync"

// parallelRows splits [0, rows) across the persistent worker pool, running
// fn(start, end) on each range concurrently and returning when all are done.
// Small row counts (< 8 rows per worker) run inline — the dispatch overhead
// would exceed the work.
func parallelRows(rows int, fn func(start, end int)) {
	threads := min(numThreads(), rows)
	if threads <= 1 || rows < threads*8 {
		fn(0, rows)
		return
	}
	dispatchParallel(threads, rows, fn)
}

// parallelRowsBatched keeps one coarse range per worker. A batch row already
// performs many dot products, so the ARM overdispatch used to balance short
// decode rows adds scheduling overhead instead. The regular parallelRows path
// remains unchanged, providing a local rollback for this prefill-only tuning.
func parallelRowsBatched(rows int, fn func(start, end int)) {
	threads := min(numThreads(), rows)
	if threads <= 1 || rows < threads*8 {
		fn(0, rows)
		return
	}
	dispatchParallelMode(threads, rows, false, fn)
}

// parallelChunks splits n items across the worker pool without the minimum
// per-thread row count required by parallelRows. Intended for coarse-grained
// items (e.g. attention heads) where even a single item is substantial work.
func parallelChunks(n int, fn func(start, end int)) {
	threads := min(numThreads(), n)
	if threads <= 1 {
		fn(0, n)
		return
	}
	dispatchParallel(threads, n, fn)
}

func dispatchParallel(threads, rows int, fn func(start, end int)) {
	dispatchParallelMode(threads, rows, true, fn)
}

func dispatchParallelMode(threads, rows int, allowOversubscribe bool, fn func(start, end int)) {
	// The pool belongs to the configured runtime, not to an individual job.
	// Some kernels expose fewer independent units than numThreads (Ministral
	// has only eight KV-head groups); rebuilding an 8-worker pool for attention
	// and a 12-worker pool for every following matvec was dramatically more
	// expensive than the kernels themselves. Submit fewer jobs to one stable
	// max-sized pool instead.
	pool := getRowWorkerPool(numThreads())
	// Issue more chunks than workers so faster cores naturally pick up the
	// slack of slower ones (e.g. efficiency cores on Apple Silicon). On
	// homogeneous-core amd64 the oversubscription only multiplies channel
	// wakeups, so chunks stay 1:1 with workers there. Small matvecs stay at
	// one chunk per worker to avoid channel wakeup overhead.
	chunks := threads
	if allowOversubscribe && oversubscribeDispatch && rows >= threads*128 {
		chunks = min(threads*8, cap(pool.jobs))
	}
	// Keep the calling goroutine useful: it owns one chunk while the persistent
	// workers consume the rest. This preserves `threads` total compute
	// participants (rather than adding the caller on top), removes one channel
	// send/wakeup/Done per projection, and avoids parking the goroutine that is
	// already running on a performance core on heterogeneous Apple CPUs.
	//
	// A WaitGroup replaces a per-chunk done-channel for the queued work:
	// completion is a single atomic counter drained by Done(), and Wait wakes
	// once instead of receiving one completion message per chunk.
	wg := wgPool.Get().(*sync.WaitGroup)
	wg.Add(chunks - 1)
	for w := 1; w < chunks; w++ {
		start := rows * w / chunks
		end := rows * (w + 1) / chunks
		pool.jobs <- rowJob{start: start, end: end, fn: fn, wg: wg}
	}
	fn(0, rows/chunks)
	wg.Wait()
	wgPool.Put(wg)
}

var wgPool = sync.Pool{New: func() any { return new(sync.WaitGroup) }}

type rowJob struct {
	start int
	end   int
	fn    func(start, end int)
	wg    *sync.WaitGroup
}

// rowWorkerPool is the process-wide pool of matvec worker goroutines. It is
// created lazily at the first parallel dispatch and rebuilt (old workers
// stopped) if SetNumThreads changes the thread count. Keeping the goroutines
// alive across calls avoids per-matvec spawn cost — matvecs run ~30x per
// generated token.
type rowWorkerPool struct {
	threads int
	jobs    chan rowJob
	stop    chan struct{}
}

var (
	rowPoolMu sync.Mutex
	rowPool   *rowWorkerPool
)

func getRowWorkerPool(threads int) *rowWorkerPool {
	rowPoolMu.Lock()
	defer rowPoolMu.Unlock()
	if rowPool != nil && rowPool.threads == threads {
		return rowPool
	}
	if rowPool != nil {
		close(rowPool.stop)
	}
	pool := &rowWorkerPool{
		threads: threads,
		jobs:    make(chan rowJob, threads*8),
		stop:    make(chan struct{}),
	}
	for range threads {
		go rowWorker(pool.jobs, pool.stop)
	}
	rowPool = pool
	return pool
}

func rowWorker(jobs <-chan rowJob, stop <-chan struct{}) {
	for {
		select {
		case job := <-jobs:
			job.fn(job.start, job.end)
			job.wg.Done()
		case <-stop:
			return
		}
	}
}
