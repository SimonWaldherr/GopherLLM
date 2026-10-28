package engine

import (
	"math/rand"
	"sync"
	"testing"
)

// This is the previous Q8 dispatch, retained to compare its numerical output
// and allocation cost against the pooled task under the same worker schedule.
func callbackQ8_0Group(weights []Weight, x []float32, outputs []*[]float32) bool {
	cols, total := len(x), 0
	for i, w := range weights {
		ensureLenNoClear(outputs[i], w.Rows)
		total += w.Rows
	}
	q8, scales, lease := acquireQ8(x, cols)
	parallelRows(total, func(start, end int) {
		base := 0
		for i, w := range weights {
			if s, e := clippedRange(start, end, base, base+w.Rows); s < e {
				dotQ8_0RowsQ8(w.Raw, q8, scales, cols, cols/32*34, s-base, e-base, *outputs[i])
			}
			base += w.Rows
		}
	})
	releaseQ8(q8, scales, lease)
	return true
}

func TestQ8_0DecodeTaskReuseConcurrent(t *testing.T) {
	previousThreads := configuredThreads.Load()
	previousOversubscribe := oversubscribeDispatch.Load()
	t.Cleanup(func() {
		configuredThreads.Store(previousThreads)
		oversubscribeDispatch.Store(previousOversubscribe)
	})
	configuredThreads.Store(2)
	oversubscribeDispatch.Store(true)
	withQ8Activations(true, func() {
		if !useQ8Activations.Load() {
			// The internal grouped callback assumes the caller has selected
			// Q8 activations. The public single-matrix entry point correctly
			// uses float activations when this backend is unavailable.
			t.Skip("Q8 activation backend unavailable")
		}
		const cols = 512
		rng := rand.New(rand.NewSource(8401))
		weights := make([]Weight, 4)
		for i, rows := range []int{257, 0, 39, 17} {
			weights[i] = Weight{Type: GGMLTypeQ8_0, Rows: rows, Cols: cols}
			for range rows {
				weights[i].Raw = append(weights[i].Raw, randomQ8_0Row(rng, cols)...)
			}
		}
		var workers sync.WaitGroup
		for worker := range 4 {
			x := randomVec(rng, cols)
			want := make([][]float32, len(weights))
			wantPointers := make([]*[]float32, len(weights))
			for i := range want {
				wantPointers[i] = &want[i]
			}
			callbackQ8_0Group(weights, x, wantPointers)
			workers.Add(1)
			go func() {
				defer workers.Done()
				got := make([][]float32, len(weights))
				pointers := make([]*[]float32, len(weights))
				for i := range got {
					pointers[i] = &got[i]
				}
				for range 6 {
					// Alternate descriptor counts, including a zero-row matrix,
					// and separate calls while other callers lease pooled tasks.
					for _, count := range []int{4, 1, 3, 2} {
						if !matvecQ8_0GroupInto(weights[:count], x, pointers[:count]) {
							t.Error("valid Q8 group rejected")
							return
						}
						checkDecodeTaskRows(t, worker, got[:count], want[:count])
					}
					MatvecQ8_0Into(weights[0].Raw, x, weights[0].Rows, cols, &got[0])
					checkDecodeTaskRows(t, worker, got[:1], want[:1])
				}
			}()
		}
		workers.Wait()
	})
}

func TestQ8_0DecodeFloatFallbackConcurrent(t *testing.T) {
	previousThreads := configuredThreads.Load()
	previousOversubscribe := oversubscribeDispatch.Load()
	t.Cleanup(func() {
		configuredThreads.Store(previousThreads)
		oversubscribeDispatch.Store(previousOversubscribe)
	})
	configuredThreads.Store(2)
	oversubscribeDispatch.Store(true)
	withQ8Activations(false, func() {
		// A Q8-compatible shape ensures that the activation-mode gate, not
		// an unsupported dimension, selects the unquantized float fallback.
		const rows, cols = 257, 512
		const rowBytes = cols / 32 * 34
		rng := rand.New(rand.NewSource(8401))
		data := make([]byte, 0, rows*rowBytes)
		for range rows {
			data = append(data, randomQ8_0Row(rng, cols)...)
		}
		var workers sync.WaitGroup
		for worker := range 4 {
			x := randomVec(rng, cols)
			want := make([]float32, rows)
			for row := range rows {
				want[row] = DotQ8_0F32(data[row*rowBytes:(row+1)*rowBytes], x, cols)
			}
			workers.Add(1)
			go func() {
				defer workers.Done()
				var got []float32
				for range 6 {
					MatvecQ8_0Into(data, x, rows, cols, &got)
					checkDecodeTaskRows(t, worker, [][]float32{got}, [][]float32{want})
				}
			}()
		}
		workers.Wait()
	})
}

func BenchmarkQ8_0DecodeDispatch(b *testing.B) {
	const rows, cols = 1024, 1024
	rng := rand.New(rand.NewSource(8401))
	var raw []byte
	for range rows {
		raw = append(raw, randomQ8_0Row(rng, cols)...)
	}
	x := randomVec(rng, cols)
	weights := []Weight{{Type: GGMLTypeQ8_0, Raw: raw, Rows: rows, Cols: cols}}
	withQ8Activations(true, func() {
		for _, mode := range []string{"callback", "singleTask", "groupTask"} {
			b.Run(mode, func(b *testing.B) {
				out := make([]float32, rows)
				outputs := []*[]float32{&out}
				run := func() {
					switch mode {
					case "callback":
						callbackQ8_0Group(weights, x, outputs)
					case "singleTask":
						MatvecQ8_0Into(raw, x, rows, cols, &out)
					default:
						matvecQ8_0GroupInto(weights, x, outputs)
					}
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					run()
				}
			})
		}
	})
}
