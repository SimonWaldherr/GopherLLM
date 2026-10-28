//go:build darwin && cgo && metal

package engine

import (
	"math"
	"math/rand"
	"slices"
	"sync"
	"testing"
)

func TestMetalQ6KBatchArgmaxIndependentPenalties(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ6KDirectMinRows, 256
	rng := rand.New(rand.NewSource(20261019))
	data := make([]byte, 0, rows*210)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatal(MetalError())
	}
	defer releaseMetalWeight(w)
	for _, n := range []int{8, 2, 4, 8} {
		xs := make([]float32, n*cols)
		histories := make([][]uint32, n)
		for position := range n {
			for i := range cols {
				xs[position*cols+i] = float32((position+3)*(i%41)-53) / 47
			}
			winner, ok := argmaxMetalQ6K(w, xs[position*cols:(position+1)*cols])
			if !ok {
				t.Fatal(MetalError())
			}
			// Each history targets its own row's winner. Repeated IDs must
			// apply repeatedly, while an out-of-range token is ignored.
			histories[position] = []uint32{winner, uint32(rows + 7), winner, uint32(position)}
			if position%3 == 1 {
				histories[position] = nil
			} else if position%3 == 2 {
				histories[position] = make([]uint32, 64)
				for j := range histories[position] {
					histories[position][j] = winner
				}
			}
		}
		for _, penalty := range []float32{1, 1.1, 2, .5} {
			got := make([]uint32, n)
			if !argmaxMetalQ6KBatchPenalized(w, xs, got, histories, penalty) {
				t.Fatal(MetalError())
			}
			for position := range n {
				x := xs[position*cols : (position+1)*cols]
				want, ok := argmaxMetalQ6KPenalized(w, x, histories[position], penalty)
				if !ok || got[position] != want {
					t.Fatalf("P%d position %d penalty %g: batch=%d serial=%d/%v", n, position, penalty, got[position], want, ok)
				}
				var logits []float32
				MatvecQ6KInto(data, x, rows, cols, &logits)
				applyRepeatPenalty(logits, histories[position], penalty)
				if want := argmaxFiniteToken(logits); got[position] != want {
					t.Fatalf("P%d position %d penalty %g: batch=%d CPU=%d", n, position, penalty, got[position], want)
				}
			}
		}
	}
}

func TestMetalQ6KBatchArgmaxPenaltyValidationAndTies(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols = metalQ6KDirectMinRows, 256
	w := prepareMetalWeight(make([]byte, rows*210), GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatal(MetalError())
	}
	defer releaseMetalWeight(w)
	xs, got := make([]float32, 2*cols), []uint32{9, 9}
	histories := [][]uint32{{0, 0, 1, rows + 1}, {1, 0}}
	if !argmaxMetalQ6KBatchPenalized(w, xs, got, histories, 1.1) || !slices.Equal(got, []uint32{0, 0}) {
		t.Fatalf("ties = %v: %s", got, MetalError())
	}
	for _, penalty := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if argmaxMetalQ6KBatchPenalized(w, xs, got, histories, penalty) {
			t.Fatalf("accepted penalty %g", penalty)
		}
	}
	for _, invalid := range [][][]uint32{nil, make([][]uint32, 9), {make([]uint32, 65), nil}} {
		if argmaxMetalQ6KBatchPenalized(w, xs, got, invalid, 1.1) {
			t.Fatal("accepted invalid histories")
		}
	}
	if argmaxMetalQ6KBatchPenalized(w, xs[:cols], got, histories, 1.1) || argmaxMetalQ6KBatchPenalized(w, xs, got[:1], histories, 1.1) {
		t.Fatal("accepted short input/output")
	}
}

func TestMetalQ6KBatchArgmaxPenaltiesConcurrent(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols, n = metalQ6KDirectMinRows, 256, 4
	rng := rand.New(rand.NewSource(20261020))
	data := make([]byte, 0, rows*210)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatal(MetalError())
	}
	defer releaseMetalWeight(w)
	var xs [2][]float32
	var histories [2][][]uint32
	var want [2][]uint32
	for caller := range xs {
		xs[caller], histories[caller], want[caller] = make([]float32, n*cols), make([][]uint32, n), make([]uint32, n)
		for i := range xs[caller] {
			xs[caller][i] = float32((i%37)*(caller+3)-41) / 29
		}
		for position := range n {
			winner, ok := argmaxMetalQ6K(w, xs[caller][position*cols:(position+1)*cols])
			if !ok {
				t.Fatal(MetalError())
			}
			histories[caller][position] = []uint32{winner, winner, uint32(position + caller)}
		}
		if !argmaxMetalQ6KBatchPenalized(w, xs[caller], want[caller], histories[caller], 1.5) {
			t.Fatal(MetalError())
		}
	}
	var wg sync.WaitGroup
	errs := make(chan string, 2)
	for caller := range xs {
		wg.Add(1)
		go func(caller int) {
			defer wg.Done()
			got := make([]uint32, n)
			for range 12 {
				if !argmaxMetalQ6KBatchPenalized(w, xs[caller], got, histories[caller], 1.5) || !slices.Equal(got, want[caller]) {
					errs <- "concurrent penalty batch returned another caller's result"
					return
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
