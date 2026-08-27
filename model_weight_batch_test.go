package gopherllm

import (
	"math/rand"
	"testing"
)

func TestWeightArgmaxMatvecBatchCPUFallbackMatchesSingle(t *testing.T) {
	const rows, cols = 17, 256
	rng := rand.New(rand.NewSource(723))
	data := make([]byte, 0, rows*(cols/256)*210)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	w := Weight{Raw: data, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols}
	xs := [][]float32{randomVec(rng, cols), randomVec(rng, cols), randomVec(rng, cols)}
	got := []uint32{}
	if !w.ArgmaxMatvecBatch(xs, &got) {
		t.Fatal("ArgmaxMatvecBatch rejected valid CPU inputs")
	}
	if len(got) != len(xs) {
		t.Fatalf("token count = %d, want %d", len(got), len(xs))
	}
	for position := range xs {
		want, ok := w.ArgmaxMatvec(xs[position])
		if !ok {
			t.Fatalf("single position %d rejected", position)
		}
		if got[position] != want {
			t.Fatalf("position %d token=%d, want single argmax %d", position, got[position], want)
		}
	}
	if w.ArgmaxMatvecBatch([][]float32{xs[0], xs[1][:cols-1]}, &got) {
		t.Fatal("accepted mismatched activation widths")
	}
	if w.ArgmaxMatvecBatch(nil, &got) {
		t.Fatal("accepted empty batch")
	}
	if w.ArgmaxMatvecBatch(xs, nil) {
		t.Fatal("accepted nil output")
	}
}
