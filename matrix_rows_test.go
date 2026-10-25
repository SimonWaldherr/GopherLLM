package gopherllm

import "testing"

func TestMatrixRowsReusePreservesContiguousStorage(t *testing.T) {
	var data []float32
	var rows [][]float32
	resizeMatrixRows(&data, &rows, 8, 16, false)
	backing, headers := &data[0], &rows[0]
	for i := range data {
		data[i] = float32(i + 1)
	}
	for _, shape := range [][2]int{{2, 32}, {4, 8}, {0, 8}, {8, 16}} {
		resizeMatrixRows(&data, &rows, shape[0], shape[1], false)
		if len(rows) == 0 {
			continue
		}
		flat, ok := contiguousBatchRows(rows, shape[1])
		if !ok || &flat[0] != backing || &rows[0] != headers {
			t.Fatalf("shape %v lost contiguous storage or reusable headers", shape)
		}
		for i, value := range flat {
			if value != float32(i+1) {
				t.Fatalf("shape %v overwrote element %d", shape, i)
			}
		}
	}
	if allocations := testing.AllocsPerRun(20, func() {
		resizeMatrixRows(&data, &rows, 8, 16, true)
	}); allocations != 0 {
		t.Fatalf("warm workspace allocated %g times", allocations)
	}
	for _, value := range data {
		if value != 0 {
			t.Fatal("zeroed workspace retained data")
		}
	}
}

func BenchmarkMatrixRowsReuse(b *testing.B) {
	var data []float32
	var rows [][]float32
	resizeMatrixRows(&data, &rows, 512, 768, false)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		resizeMatrixRows(&data, &rows, 512, 768, false)
	}
}
