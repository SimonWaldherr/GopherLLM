package gopherllm

import "sync"

// parallelRows splits [0, rows) across the persistent worker pool, running
// fn(start, end) on each range concurrently and returning when all are done.
// Small row counts (< 8 rows per worker) run inline — the dispatch overhead
// would exceed the work.
func parallelRows(rows int, fn func(start, end int)) {
	poolThreads := numThreads()
	threads := min(poolThreads, rows)
	if threads <= 1 || rows < threads*8 {
		fn(0, rows)
		return
	}
	dispatchParallel(poolThreads, threads, rows, fn)
}

// rowTask is the allocation-free counterpart of the callback accepted by
// parallelRows. Hot matvec paths keep their per-call state in a pooled task;
// sending that task through the worker pool avoids allocating an escaping
// closure for every projection.
type rowTask interface {
	runRows(start, end int)
}

func parallelRowsTask(rows int, task rowTask) {
	poolThreads := numThreads()
	threads := min(poolThreads, rows)
	if threads <= 1 || rows < threads*8 {
		task.runRows(0, rows)
		return
	}
	dispatchParallelTask(poolThreads, threads, rows, task)
}

// parallelRowsBatched keeps one coarse range per worker. A batch row already
// performs many dot products, so the ARM overdispatch used to balance short
// decode rows adds scheduling overhead instead. The regular parallelRows path
// remains unchanged, providing a local rollback for this prefill-only tuning.
func parallelRowsBatched(rows int, fn func(start, end int)) {
	poolThreads := numThreads()
	threads := min(poolThreads, rows)
	if threads <= 1 || rows < threads*8 {
		fn(0, rows)
		return
	}
	dispatchParallelMode(poolThreads, threads, rows, false, fn)
}

// parallelRowsBatchedTask preserves parallelRowsBatched's one-range-per-
// worker scheduling for callers that carry their work in a reusable rowTask.
func parallelRowsBatchedTask(rows int, task rowTask) {
	poolThreads := numThreads()
	threads := min(poolThreads, rows)
	if threads <= 1 || rows < threads*8 {
		task.runRows(0, rows)
		return
	}
	dispatchParallelTaskMode(poolThreads, threads, rows, false, task)
}

// parallelChunks splits n items across the worker pool without the minimum
// per-thread row count required by parallelRows. Intended for coarse-grained
// items (e.g. attention heads) where even a single item is substantial work.
func parallelChunks(n int, fn func(start, end int)) {
	poolThreads := numThreads()
	threads := min(poolThreads, n)
	if threads <= 1 {
		fn(0, n)
		return
	}
	dispatchParallel(poolThreads, threads, n, fn)
}

// parallelChunksTask is parallelChunks without an escaping callback. It is
// used by attention, where even one head is enough work to justify a worker.
func parallelChunksTask(n int, task rowTask) {
	poolThreads := numThreads()
	threads := min(poolThreads, n)
	if threads <= 1 {
		task.runRows(0, n)
		return
	}
	dispatchParallelTask(poolThreads, threads, n, task)
}

func dispatchParallel(poolThreads, threads, rows int, fn func(start, end int)) {
	dispatchParallelMode(poolThreads, threads, rows, true, fn)
}

func dispatchParallelTask(poolThreads, threads, rows int, task rowTask) {
	dispatchParallelTaskMode(poolThreads, threads, rows, true, task)
}

func dispatchParallelTaskMode(poolThreads, threads, rows int, allowOversubscribe bool, task rowTask) {
	pool := acquireRowWorkerPool(poolThreads)
	chunks := threads
	if allowOversubscribe && oversubscribeDispatch.Load() && rows >= threads*128 {
		chunks = min(threads*8, cap(pool.jobs))
	}
	wg := wgPool.Get().(*sync.WaitGroup)
	wg.Add(chunks - 1)
	for w := 1; w < chunks; w++ {
		start := rows * w / chunks
		end := rows * (w + 1) / chunks
		pool.jobs <- rowJob{start: start, end: end, task: task, wg: wg}
	}
	task.runRows(0, rows/chunks)
	wg.Wait()
	wgPool.Put(wg)
	releaseRowWorkerPool(pool)
}

func dispatchParallelMode(poolThreads, threads, rows int, allowOversubscribe bool, fn func(start, end int)) {
	// The pool belongs to the configured runtime, not to an individual job.
	// Some kernels expose fewer independent units than numThreads (Ministral
	// has only eight KV-head groups); rebuilding an 8-worker pool for attention
	// and a 12-worker pool for every following matvec was dramatically more
	// expensive than the kernels themselves. Submit fewer jobs to one stable
	// max-sized pool instead.
	pool := acquireRowWorkerPool(poolThreads)
	// Issue more chunks than workers so faster cores naturally pick up the
	// slack of slower ones (e.g. efficiency cores on Apple Silicon). On
	// homogeneous-core amd64 the oversubscription only multiplies channel
	// wakeups, so chunks stay 1:1 with workers there. Small matvecs stay at
	// one chunk per worker to avoid channel wakeup overhead.
	chunks := threads
	if allowOversubscribe && oversubscribeDispatch.Load() && rows >= threads*128 {
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
	releaseRowWorkerPool(pool)
}

var wgPool = sync.Pool{New: func() any { return new(sync.WaitGroup) }}

type rowJob struct {
	start int
	end   int
	fn    func(start, end int)
	task  rowTask
	wg    *sync.WaitGroup
}

// rowWorkerPool is the process-wide pool of matvec worker goroutines. It is
// created lazily at the first parallel dispatch and rebuilt (old workers
// stopped) if SetNumThreads changes the thread count. Keeping the goroutines
// alive across calls avoids per-matvec spawn cost — matvecs run ~30x per
// generated token.
type rowWorkerPool struct {
	threads  int
	jobs     chan rowJob
	stop     chan struct{}
	active   int
	retiring bool
	stopped  bool
}

var (
	rowPoolMu sync.Mutex
	rowPool   *rowWorkerPool
)

// acquireRowWorkerPool leases the current worker pool for one dispatch. A
// thread-count change replaces the global pool, but an older pool stays alive
// until every dispatch that already obtained it has completed. Without that
// lease, closing its workers could strand queued jobs and leave their
// WaitGroup waiting forever.
func acquireRowWorkerPool(threads int) *rowWorkerPool {
	rowPoolMu.Lock()
	defer rowPoolMu.Unlock()
	if rowPool != nil && rowPool.threads == threads {
		rowPool.active++
		return rowPool
	}
	if old := rowPool; old != nil {
		retireRowWorkerPoolLocked(old)
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
	pool.active++
	return pool
}

func releaseRowWorkerPool(pool *rowWorkerPool) {
	rowPoolMu.Lock()
	defer rowPoolMu.Unlock()
	if pool == nil || pool.active == 0 {
		return
	}
	pool.active--
	if pool.retiring && pool.active == 0 && !pool.stopped {
		close(pool.stop)
		pool.stopped = true
	}
}

// retireRowWorkerPoolLocked marks a replaced pool for shutdown. rowPoolMu must
// be held by the caller.
func retireRowWorkerPoolLocked(pool *rowWorkerPool) {
	pool.retiring = true
	if pool.active == 0 && !pool.stopped {
		close(pool.stop)
		pool.stopped = true
	}
}

func rowWorker(jobs <-chan rowJob, stop <-chan struct{}) {
	for {
		select {
		case job := <-jobs:
			if job.task != nil {
				job.task.runRows(job.start, job.end)
			} else {
				job.fn(job.start, job.end)
			}
			job.wg.Done()
		case <-stop:
			return
		}
	}
}
