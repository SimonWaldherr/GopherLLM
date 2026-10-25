package yolo

import (
	"sync"

	"github.com/SimonWaldherr/GopherLLM/internal/threads"
)

// parallelChunks splits independent YOLO work without depending on the root
// inference runtime.
func parallelChunks(n int, fn func(start, end int)) {
	if n <= 0 {
		fn(0, n)
		return
	}
	workers := min(threads.Count(), n)
	if workers <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for worker := 1; worker < workers; worker++ {
		start, end := n*worker/workers, n*(worker+1)/workers
		go func() {
			defer wg.Done()
			fn(start, end)
		}()
	}
	fn(0, n/workers)
	wg.Wait()
}

func yoloAxpy(out []float32, alpha float32, x []float32) {
	for i := 0; i < min(len(out), len(x)); i++ {
		out[i] += alpha * x[i]
	}
}
