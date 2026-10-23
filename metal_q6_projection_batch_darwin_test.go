//go:build darwin && cgo && metal

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func metalQ6ProjectionFixture(t testing.TB, rows, cols int) *MetalWeight {
	t.Helper()
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	rng := rand.New(rand.NewSource(381027))
	rowBytes := cols / 256 * 210
	data := make([]byte, rows*rowBytes)
	patterns := make([][]byte, min(rows, 32))
	for i := range patterns {
		patterns[i] = randomQ6KRow(rng, cols)
	}
	for row := range rows {
		copy(data[row*rowBytes:], patterns[row%len(patterns)])
	}
	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil || !metalWeightUsesDirect(w) {
		t.Fatalf("prepare direct Q6_K %dx%d: %s", rows, cols, MetalError())
	}
	t.Cleanup(func() { releaseMetalWeight(w) })
	return w
}

func metalQ6ProjectionInputs(batch, cols, seed int) []float32 {
	x := make([]float32, batch*cols)
	for token := range batch {
		for col := range cols {
			x[token*cols+col] = float32(((token+seed)*(col%53)-71)%101) / 37
		}
	}
	return x
}

func metalQ6ProjectionSerial(t testing.TB, w *MetalWeight, x []float32, batch int) []float32 {
	t.Helper()
	want := make([]float32, batch*w.rows)
	for token := range batch {
		row := want[token*w.rows : (token+1)*w.rows]
		if !matvecMetalQ6KInto(w, x[token*w.cols:(token+1)*w.cols], w.rows, w.cols, &row) {
			t.Fatalf("scalar Q6_K projection: %s", MetalError())
		}
	}
	return want
}

func TestMetalQ6KDecodeBatchMatchesScalarBits(t *testing.T) {
	// An odd row count covers the final partial GPU threadgroup, while the
	// real direct-offload threshold prevents accidental CPU-only coverage.
	w := metalQ6ProjectionFixture(t, metalQ6KDirectMinRows+1, 512)
	const guard = float32(-12345.25)
	for _, batch := range []int{32, 1, 2, 4, 8, 17, 32} {
		t.Run(fmt.Sprintf("positions=%d", batch), func(t *testing.T) {
			x := metalQ6ProjectionInputs(batch, w.cols, batch+3)
			xBefore := slices.Clone(x)
			want := metalQ6ProjectionSerial(t, w, x, batch)
			storage := make([]float32, len(want)+2)
			storage[0], storage[len(storage)-1] = guard, guard
			got := storage[1 : len(storage)-1 : len(storage)-1]
			if !matvecMetalQ6KDecodeBatchInto(w, x, batch, &got) {
				t.Fatal(MetalError())
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("position=%d row=%d got=%08x want=%08x", i/w.rows, i%w.rows,
						math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			if storage[0] != guard || storage[len(storage)-1] != guard || !slices.Equal(x, xBefore) {
				t.Fatal("projection changed its input or output guard")
			}
		})
	}
}

func TestMetalQ6KDecodeBatchRejectsInvalidInputs(t *testing.T) {
	w := metalQ6ProjectionFixture(t, metalQ6KDirectMinRows, 256)
	x := metalQ6ProjectionInputs(2, w.cols, 5)
	for _, tc := range []struct {
		name  string
		edit  func(*MetalWeight)
		x     []float32
		batch int
		nilW  bool
	}{
		{name: "nil-weight", x: x, batch: 2, nilW: true},
		{name: "zero-batch", x: x},
		{name: "negative-batch", x: x, batch: -1},
		{name: "large-batch", x: x, batch: 33},
		{name: "short-input", x: x[:len(x)-1], batch: 2},
		{name: "empty-input", batch: 1},
		{name: "wrong-type", edit: func(w *MetalWeight) { w.typ = GGMLTypeQ4_K }, x: x, batch: 2},
		{name: "missing-handle", edit: func(w *MetalWeight) { w.q6 = nil }, x: x, batch: 2},
		{name: "below-direct-threshold", edit: func(w *MetalWeight) { w.rows = metalQ6KDirectMinRows - 1 }, x: x, batch: 2},
		{name: "zero-cols", edit: func(w *MetalWeight) { w.cols = 0 }, x: x, batch: 2},
		{name: "unaligned-cols", edit: func(w *MetalWeight) { w.cols-- }, x: x, batch: 2},
		{name: "overflow-rows", edit: func(w *MetalWeight) { w.rows = int(^uint(0) >> 1) }, x: x, batch: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			copyWeight := *w
			if tc.edit != nil {
				tc.edit(&copyWeight)
			}
			inputWeight := &copyWeight
			if tc.nilW {
				inputWeight = nil
			}
			out := []float32{123, -71, .25}
			before := slices.Clone(out)
			if matvecMetalQ6KDecodeBatchInto(inputWeight, tc.x, tc.batch, &out) {
				t.Fatal("accepted invalid projection")
			}
			if !slices.Equal(out, before) {
				t.Fatal("rejected projection changed output")
			}
		})
	}
	if matvecMetalQ6KDecodeBatchInto(w, x, 2, nil) {
		t.Fatal("accepted nil output pointer")
	}
	// Unlike the resizing root adapter, the backend receives a fixed slab and
	// must reject a short output without writing its valid prefix.
	for _, n := range []int{0, 1, 2*w.rows - 1} {
		out := make([]float32, n)
		for i := range out {
			out[i] = -31
		}
		before := slices.Clone(out)
		if metalbackend.MatvecQ6KDecodeBatch(w.q6, x, out, 2) || !slices.Equal(out, before) {
			t.Fatalf("backend accepted/changed short output of %d elements", n)
		}
	}
	var out []float32
	if !matvecMetalQ6KDecodeBatchInto(w, x, 2, &out) || len(out) != 2*w.rows {
		t.Fatal("valid projection did not allocate its output")
	}
}

func TestMetalQ6KDecodeBatchSerializesWorkspace(t *testing.T) {
	w := metalQ6ProjectionFixture(t, metalQ6KDirectMinRows, 256)
	const requests = 3
	xs, wants := make([][]float32, requests), make([][]float32, requests)
	for request := range requests {
		xs[request] = metalQ6ProjectionInputs(2, w.cols, request+11)
		wants[request] = metalQ6ProjectionSerial(t, w, xs[request], 2)
	}
	var wg sync.WaitGroup
	errs := make(chan string, requests)
	for request := range requests {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var out []float32
			for range 4 {
				if !matvecMetalQ6KDecodeBatchInto(w, xs[request], 2, &out) || !slices.Equal(out, wants[request]) {
					errs <- fmt.Sprintf("request %d changed under concurrent workspace use", request)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// The Qwen 27B QKV shape streams about 43 MB of Q6_K weights. Repeated row
// patterns keep fixture construction cheap; this benchmark measures only
// projection submission, arithmetic and readback, without recurrent updates.
func BenchmarkMetalQ6KDecodeBatchQwen27B(b *testing.B) {
	const rows, cols = 10240, 5120
	w := metalQ6ProjectionFixture(b, rows, cols)
	for _, batch := range []int{2, 8, 32} {
		x := metalQ6ProjectionInputs(batch, cols, 17)
		want := metalQ6ProjectionSerial(b, w, x, batch)
		var got []float32
		if !matvecMetalQ6KDecodeBatchInto(w, x, batch, &got) || !slices.Equal(got, want) {
			b.Fatal("batched projection differs from scalar Metal")
		}
		b.Run(fmt.Sprintf("positions=%d", batch), func(b *testing.B) {
			for _, bulk := range []bool{false, true} {
				b.Run(fmt.Sprintf("bulk=%t", bulk), func(b *testing.B) {
					out := make([]float32, batch*rows)
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						if bulk {
							if !matvecMetalQ6KDecodeBatchInto(w, x, batch, &out) {
								b.Fatal(MetalError())
							}
						} else {
							for token := range batch {
								row := out[token*rows : (token+1)*rows]
								if !matvecMetalQ6KInto(w, x[token*cols:(token+1)*cols], rows, cols, &row) {
									b.Fatal(MetalError())
								}
							}
						}
					}
				})
			}
		})
	}
}
