package engine

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// Keep the former callback dispatch as a differential benchmark. The dot
// kernels and their accumulation order are unchanged by task reuse.
func callbackQ6KMatvec(data []byte, x []float32, rows, cols int, out *[]float32) {
	ensureLenNoClear(out, rows)
	scratch := xsumsScratchPool.Get().(*[]float32)
	xs := fillQ6KXSums16(x, cols, scratch)
	ScaleF32(xs, 32)
	rowBytes := (cols / 256) * 210
	if useQ8Activations.Load() {
		q8, scales, lease := acquireQ8(x, cols)
		parallelRows(rows, func(start, end int) {
			dotQ6KRowsQ8(data, q8, scales, xs, cols, rowBytes, start, end, *out)
		})
		releaseQ8(q8, scales, lease)
	} else {
		parallelRows(rows, func(start, end int) {
			dotQ6KRowsWithXSums(data, x, xs, cols, rowBytes, start, end, *out)
		})
	}
	*scratch = xs
	xsumsScratchPool.Put(scratch)
}

func TestQuantizedDecodeTasksConcurrent(t *testing.T) {
	const cols = 512
	previousThreads := configuredThreads.Load()
	previousOversubscribe := oversubscribeDispatch.Load()
	t.Cleanup(func() {
		configuredThreads.Store(previousThreads)
		oversubscribeDispatch.Store(previousOversubscribe)
	})
	configuredThreads.Store(2)
	oversubscribeDispatch.Store(true)
	// Every projection uses real workers, including dynamically claimed chunks.
	// Uneven row counts also force chunks to straddle projection boundaries.
	rowCounts := [3]int{257, 61, 43}
	rng := rand.New(rand.NewSource(82))
	var q4, q6 [3][]byte
	for m, rows := range rowCounts {
		for range rows {
			q4[m] = append(q4[m], randomQ4KRow(rng, cols)...)
			q6[m] = append(q6[m], randomQ6KRow(rng, cols)...)
		}
	}
	for _, q8 := range []bool{false, true} {
		t.Run(fmt.Sprintf("q8=%v", q8), func(t *testing.T) {
			withQ8Activations(q8, func() {
				var workers sync.WaitGroup
				for worker := range 4 {
					x := randomVec(rng, cols)
					var want4, want6 [3][]float32
					for m, rows := range rowCounts {
						MatvecQ4KInto(q4[m], x, rows, cols, &want4[m])
						callbackQ6KMatvec(q6[m], x, rows, cols, &want6[m])
					}
					workers.Add(1)
					go func() {
						defer workers.Done()
						var got4, got6 [3][]float32
						var sums []float32
						for range 6 {
							MatvecQ6KInto(q6[0], x, rowCounts[0], cols, &got6[0])
							checkDecodeTaskRows(t, worker, got6[:1], want6[:1])
							if !MatvecQ6K2Into(q6[0], rowCounts[0], cols, q6[1], rowCounts[1], cols, x, &got6[0], &got6[1]) {
								t.Error("Q6_K pair rejected valid dimensions")
								return
							}
							checkDecodeTaskRows(t, worker, got6[:2], want6[:2])
							if !MatvecQ6K3Into(q6[0], rowCounts[0], cols, q6[1], rowCounts[1], cols, q6[2], rowCounts[2], cols, x, &got6[0], &got6[1], &got6[2]) {
								t.Error("Q6_K triple rejected valid dimensions")
								return
							}
							checkDecodeTaskRows(t, worker, got6[:], want6[:])
							if !MatvecQ4K2IntoWithXSums(q4[0], rowCounts[0], cols, q4[1], rowCounts[1], cols, x, &sums, &got4[0], &got4[1]) {
								t.Error("Q4_K pair rejected valid dimensions")
								return
							}
							checkDecodeTaskRows(t, worker, got4[:2], want4[:2])
						}
					}()
				}
				workers.Wait()
			})
		})
	}
}

func checkDecodeTaskRows(t *testing.T, worker int, got, want [][]float32) {
	t.Helper()
	for m := range want {
		if len(got[m]) != len(want[m]) {
			t.Errorf("worker %d projection %d: output length %d, want %d", worker, m, len(got[m]), len(want[m]))
			continue
		}
		for row := range want[m] {
			if math.Float32bits(got[m][row]) != math.Float32bits(want[m][row]) {
				t.Errorf("worker %d projection %d row %d: %.9g, want %.9g", worker, m, row, got[m][row], want[m][row])
				break
			}
		}
	}
}

func BenchmarkQuantizedDecodeDispatch(b *testing.B) {
	const rows, cols = 1024, 1024
	rng := rand.New(rand.NewSource(82))
	var data []byte
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)
	for _, q8 := range []bool{false, true} {
		b.Run(fmt.Sprintf("q8=%v", q8), func(b *testing.B) {
			withQ8Activations(q8, func() {
				for _, mode := range []string{"callback", "task"} {
					b.Run(mode, func(b *testing.B) {
						out := make([]float32, rows)
						fn := callbackQ6KMatvec
						if mode == "task" {
							fn = MatvecQ6KInto
						}
						fn(data, x, rows, cols, &out)
						b.ReportAllocs()
						b.ResetTimer()
						for b.Loop() {
							fn(data, x, rows, cols, &out)
						}
					})
				}
			})
		})
	}
}
