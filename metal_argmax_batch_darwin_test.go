//go:build darwin && cgo && metal

package gopherllm

import (
	"math/rand"
	"os"
	"strconv"
	"sync"
	"testing"
)

func TestMetalQ6KBatchArgmaxMatchesCPUAndReusesWorkspace(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ6KDirectMinRows, 256
	rng := rand.New(rand.NewSource(721))
	data := make([]byte, 0, rows*(cols/256)*210)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatalf("prepare Q6_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)

	// Start at the largest supported draft window, then shrink it. The shared
	// workspace must retain its allocation without retaining stale winners.
	for _, batch := range []int{metalBatchArgmaxMaxTokens, 2, metalBatchArgmaxMaxTokens} {
		xs := make([]float32, batch*cols)
		views := make([][]float32, batch)
		for position := range batch {
			views[position] = xs[position*cols : (position+1)*cols]
			for i := range cols {
				views[position][i] = float32(((position+3)*(i%31)-47)%29) / 13
			}
		}
		got := make([]uint32, batch)
		if !argmaxMetalQ6KBatch(w, xs, got, batch) {
			t.Fatalf("batch=%d Metal Q6_K argmax: %s", batch, MetalError())
		}
		for position := range batch {
			logits := []float32{}
			MatvecQ6KInto(data, views[position], rows, cols, &logits)
			want := argmaxFiniteToken(logits)
			if got[position] != want || got[position] >= rows {
				t.Fatalf("batch=%d position=%d token=%d, want valid CPU argmax %d", batch, position, got[position], want)
			}
		}

		// Exercise the public Weight-level form too: speculative verification
		// naturally has individual hidden-state views, not a flattened slice.
		weight := Weight{Raw: data, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols, Metal: w}
		viaWeight := []uint32{}
		if !weight.ArgmaxMatvecBatch(views, &viaWeight) {
			t.Fatalf("batch=%d Weight.ArgmaxMatvecBatch rejected valid inputs", batch)
		}
		for position := range batch {
			if viaWeight[position] != got[position] {
				t.Fatalf("batch=%d position=%d Weight token=%d, direct Metal token=%d", batch, position, viaWeight[position], got[position])
			}
		}
	}
}

func TestMetalQ6KBatchArgmaxTieAndValidation(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols, batch = metalQ6KDirectMinRows, 256, metalBatchArgmaxMaxTokens
	w := prepareMetalWeight(make([]byte, rows*(cols/256)*210), GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatalf("prepare zero Q6_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)
	xs := make([]float32, batch*cols)
	for position := range batch {
		for i := range cols {
			xs[position*cols+i] = float32(position-i%7) / 5
		}
	}
	tokens := make([]uint32, batch)
	if !argmaxMetalQ6KBatch(w, xs, tokens, batch) {
		t.Fatalf("tied Q6_K batch argmax: %s", MetalError())
	}
	for position, token := range tokens {
		if token != 0 {
			t.Fatalf("tied position %d token=%d, want lowest valid token 0", position, token)
		}
	}
	if argmaxMetalQ6KBatch(w, xs, tokens, 0) {
		t.Fatal("accepted zero batch")
	}
	if argmaxMetalQ6KBatch(w, xs, tokens, metalBatchArgmaxMaxTokens+1) {
		t.Fatal("accepted oversized batch")
	}
	if argmaxMetalQ6KBatch(w, xs[:cols], tokens, 2) {
		t.Fatal("accepted short input")
	}
	if argmaxMetalQ6KBatch(w, xs, tokens[:1], 2) {
		t.Fatal("accepted short token output")
	}
}

func TestMetalQ6KBatchArgmaxSerializesSharedWorkspace(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols, batch = metalQ6KDirectMinRows, 256, 2
	rng := rand.New(rand.NewSource(724))
	data := make([]byte, 0, rows*(cols/256)*210)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatalf("prepare Q6_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)

	inputs := [2][]float32{make([]float32, batch*cols), make([]float32, batch*cols)}
	for caller := range inputs {
		for i := range inputs[caller] {
			inputs[caller][i] = float32((caller+2)*(i%41)-31) / 17
		}
	}
	var want [2][]uint32
	for caller := range inputs {
		want[caller] = make([]uint32, batch)
		if !argmaxMetalQ6KBatch(w, inputs[caller], want[caller], batch) {
			t.Fatalf("warmup caller=%d: %s", caller, MetalError())
		}
	}

	// Both goroutines share the global x/out/result workspace. The backend's
	// queue lock must make every call see only its own contiguous input and
	// result records; this would otherwise intermittently return the other
	// caller's winners under an ordinary server workload.
	errs := make(chan string, 2)
	var wg sync.WaitGroup
	for caller := range inputs {
		wg.Add(1)
		go func(caller int) {
			defer wg.Done()
			got := make([]uint32, batch)
			for iteration := 0; iteration < 12; iteration++ {
				if !argmaxMetalQ6KBatch(w, inputs[caller], got, batch) {
					errs <- "concurrent batch argmax: " + MetalError()
					return
				}
				for position := range batch {
					if got[position] != want[caller][position] {
						errs <- "concurrent batch argmax returned a stale workspace winner"
						return
					}
				}
			}
		}(caller)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

// BenchmarkMetalQ6KBatchArgmaxMinistral3BOutput compares a short verifier
// window against serial Metal calls at the actual 3B vocabulary shape. It is
// deliberately greedy-only: neither variant reads logits back to the CPU.
func BenchmarkMetalQ6KBatchArgmaxMinistral3BOutput(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	const rows, cols = 131072, 3072
	rng := rand.New(rand.NewSource(722))
	row := randomQ6KRow(rng, cols)
	data := make([]byte, rows*len(row))
	for r := range rows {
		copy(data[r*len(row):], row)
	}
	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		b.Fatalf("prepare Q6_K Metal output: %s", MetalError())
	}
	b.Cleanup(func() { releaseMetalWeight(w) })

	for _, batch := range []int{2, 4, metalBatchArgmaxMaxTokens} {
		xs := make([]float32, batch*cols)
		for i := range xs {
			xs[i] = float32((i%53)-26) / 19
		}
		tokens := make([]uint32, batch)
		b.Run("P"+strconv.Itoa(batch)+"/serial", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for position := range batch {
					if _, ok := argmaxMetalQ6K(w, xs[position*cols:(position+1)*cols]); !ok {
						b.Fatal(MetalError())
					}
				}
			}
		})
		b.Run("P"+strconv.Itoa(batch)+"/batch", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if !argmaxMetalQ6KBatch(w, xs, tokens, batch) {
					b.Fatal(MetalError())
				}
			}
		})
	}
}

// BenchmarkMetalQ6KBatchArgmaxConfiguredModel makes the short verifier-window
// crossover reproducible on a locally installed model. For example:
// GOPHERLLM_BENCH_MODEL=/path/to/Ministral-Q4_K_M.gguf go test -tags metal
// -run '^$' -bench BenchmarkMetalQ6KBatchArgmaxConfiguredModel -benchtime=5x
func BenchmarkMetalQ6KBatchArgmaxConfiguredModel(b *testing.B) {
	modelPath := os.Getenv("GOPHERLLM_BENCH_MODEL")
	if modelPath == "" {
		b.Skip("set GOPHERLLM_BENCH_MODEL=/path/to/Q6_K-output-model.gguf")
	}
	runner, _, err := RunnerFromPathWithOptions(modelPath, LoadOptions{UseMetal: true})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = runner.Close() })
	if runner.kind != loadedStandard || runner.standard.Output.Type != GGMLTypeQ6_K || runner.standard.Output.Metal == nil {
		b.Skip("configured model has no eligible Q6_K Metal output projection")
	}
	weight := runner.standard.Output
	for _, batch := range []int{2, 4, metalBatchArgmaxMaxTokens} {
		xs := make([]float32, batch*weight.Cols)
		for i := range xs {
			xs[i] = float32((i%97)-48) / 31
		}
		tokens := make([]uint32, batch)
		b.Run("P"+strconv.Itoa(batch)+"/serial", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				for position := range batch {
					if _, ok := argmaxMetalQ6K(weight.Metal, xs[position*weight.Cols:(position+1)*weight.Cols]); !ok {
						b.Fatal(MetalError())
					}
				}
			}
		})
		b.Run("P"+strconv.Itoa(batch)+"/batch", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if !argmaxMetalQ6KBatch(weight.Metal, xs, tokens, batch) {
					b.Fatal(MetalError())
				}
			}
		})
	}
}
