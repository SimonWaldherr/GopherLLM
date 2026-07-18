package gopherllm

import (
	"os"
	"sync"
	"sync/atomic"
)

// prefaultSink forces the compiler to keep every touched byte "used" so the
// page-in reads in prefaultPages can never be optimized away; the value
// itself is meaningless and never read back.
var prefaultSink atomic.Uint64

const prefaultPageSize = 4096

// prefaultPages forces every page of an mmap'd region into the process's
// working set before the model is reported ready, by reading one byte per
// page across all worker threads concurrently.
//
// Bottleneck: a memory-mapped GGUF file only pages in on first touch, and a
// forward pass touches essentially every weight byte — so with a bare mmap,
// the *first* request after loading silently inherits the full page-in cost
// (disk I/O, or on Windows, real-time antivirus scanning of each mapped
// page) inside its prefill/TTFT instead of load time. Measured on the
// throttled Windows dev laptop, this made TTFT for a 12-token prompt 4-5s
// even with the model file already OS-cached — decode-per-token cost is
// ~2ms, so that was almost entirely first-touch overhead.
//
// Change: touch every page once, in parallel, right after mmap'ing and
// before the Runner is handed back, so "Loaded ... in Xs" honestly reports
// full readiness and every subsequent request (the first one included) sees
// consistent, already-warm latency. Effect: for a one-shot CLI run the total
// wall-clock is roughly unchanged (the bytes still have to be paged in from
// somewhere); the win is for the HTTP server and REPL cases, where this cost
// would otherwise land unpredictably on whichever request happens to run
// first. Risk: none beyond the already-necessary page-in cost happening
// eagerly instead of lazily. Rollback: set GOPHERLLM_NO_PREFAULT=1.
func prefaultPages(data []byte) {
	if len(data) == 0 || os.Getenv("GOPHERLLM_NO_PREFAULT") != "" {
		return
	}
	threads := numThreads()
	if threads < 1 {
		threads = 1
	}
	if threads > len(data) {
		threads = 1
	}
	chunk := (len(data) + threads - 1) / threads
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		start := t * chunk
		end := min(start+chunk, len(data))
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			var sink byte
			for i := lo; i < hi; i += prefaultPageSize {
				sink += data[i]
			}
			prefaultSink.Add(uint64(sink))
		}(start, end)
	}
	wg.Wait()
}
