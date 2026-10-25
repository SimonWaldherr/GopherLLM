//go:build darwin && cgo

package voxtralblas

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

func TestAttentionWorkerBudget(t *testing.T) {
	for _, tc := range [][5]int{
		{1, 751, 32, 8, 1}, {372, 372, 32, 8, 7}, {2048, 2048, 32, 8, 1},
		{17, 145, 7, 8, 7}, {3, 19, 2, 8, 1}, {16, 766, 32, 1, 1},
	} {
		n, total, heads, threads, want := tc[0], tc[1], tc[2], tc[3], tc[4]
		got := AttentionWorkers(n, total, heads, threads)
		if got != want || got*n*total > max(1<<20, n*total) {
			t.Fatalf("%v: workers=%d", tc, got)
		}
	}
}

func TestParallelAttentionMatchesSerial(t *testing.T) {
	for _, shape := range [][3]int{{1, 750, 32}, {17, 128, 7}, {128, 0, 12}, {372, 0, 32}} {
		for _, window := range []int{0, 1, 17, 750} {
			t.Run(fmt.Sprintf("%v/window%d", shape, window), func(t *testing.T) {
				n, past, heads := shape[0], shape[1], shape[2]
				const dim = 64
				total, stride := past+n, past+n+11
				q := make([]float32, n*heads*dim)
				k, v := make([]float32, heads*stride*dim), make([]float32, heads*stride*dim)
				for i := range q {
					q[i] = float32(i%41-20) * .03
				}
				for i := range k {
					k[i], v[i] = float32(i%47-23)*.01, float32(i%53-26)*.02
				}
				want := make([]float32, len(q))
				Attention(n, total, heads, dim, past, window, stride, .125, q, k, v, make([]float32, n*total), want)
				var wg sync.WaitGroup
				for _, threads := range []int{1, 3, 8} {
					wg.Add(1)
					go func() {
						defer wg.Done()
						workers := AttentionWorkers(n, total, heads, threads)
						storage := make([]float32, len(q)+2)
						storage[0], storage[len(storage)-1] = 123, 456
						out := storage[1 : len(storage)-1]
						scores := make([]float32, workers*n*total)
						for range 2 {
							for i := range out {
								out[i] = float32(math.NaN())
							}
							AttentionWithWorkers(n, total, heads, dim, past, window, stride, .125, q, k, v, scores, out, workers)
							for i, value := range want {
								if math.Float32bits(out[i]) != math.Float32bits(value) {
									t.Errorf("workers%d element%d: %g, want %g", workers, i, out[i], value)
									return
								}
							}
							if storage[0] != 123 || storage[len(storage)-1] != 456 {
								t.Error("output guard overwritten")
							}
						}
					}()
				}
				wg.Wait()
			})
		}
	}
}
