package mmapfile

import (
	"os"
	"sort"
	"sync"
	"sync/atomic"
)

// prefaultSink forces the compiler to keep every touched byte "used" so the
// page-in reads in PrefaultPages can never be optimized away; the value
// itself is meaningless and never read back.
var prefaultSink atomic.Uint64

const prefaultPageSize = 4096

type ByteRange struct {
	Start int
	End   int
}

// PrefaultPages forces every page of an mmap'd region into the process's
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
func PrefaultPages(data []byte, workers int) {
	PrefaultRanges(data, []ByteRange{{Start: 0, End: len(data)}}, workers)
}

// PrefaultRanges touches just the given mmap ranges. Ranges are normalized and
// split into moderately sized jobs so one large tensor cannot leave most worker
// threads idle. This is deliberately a best-effort warm-up: mmap remains the
// owner of residency and the operating system may evict any page later.
func PrefaultRanges(data []byte, ranges []ByteRange, workers int) {
	if len(data) == 0 || len(ranges) == 0 || os.Getenv("GOPHERLLM_NO_PREFAULT") != "" {
		return
	}
	ranges = NormalizeRanges(len(data), ranges)
	if len(ranges) == 0 {
		return
	}

	// 16 MiB keeps the job list compact even for large GGUFs, yet gives the
	// worker pool enough pieces to balance differently sized tensors.
	const chunkBytes = 16 << 20
	jobs := make([]ByteRange, 0, len(ranges))
	for _, r := range ranges {
		for start := r.Start; start < r.End; start += chunkBytes {
			jobs = append(jobs, ByteRange{Start: start, End: min(start+chunkBytes, r.End)})
		}
	}
	threads := min(max(1, workers), len(jobs))
	work := make(chan ByteRange)
	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := range work {
				var sink byte
				for i := r.Start; i < r.End; i += prefaultPageSize {
					sink += data[i]
				}
				prefaultSink.Add(uint64(sink))
			}
		}()
	}
	for _, job := range jobs {
		work <- job
	}
	close(work)
	wg.Wait()
}

func NormalizeRanges(dataLen int, ranges []ByteRange) []ByteRange {
	if dataLen <= 0 || len(ranges) == 0 {
		return nil
	}
	// Tensor descriptors are in file order for normal GGUFs, but sorting keeps
	// this safe for producers that do not preserve that convention.
	ranges = append([]ByteRange(nil), ranges...)
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].Start < ranges[j].Start })
	out := make([]ByteRange, 0, len(ranges))
	for _, r := range ranges {
		r.Start = max(0, r.Start)
		r.End = min(dataLen, r.End)
		if r.Start >= r.End {
			continue
		}
		if n := len(out); n > 0 && r.Start <= out[n-1].End {
			out[n-1].End = max(out[n-1].End, r.End)
			continue
		}
		out = append(out, r)
	}
	return out
}
