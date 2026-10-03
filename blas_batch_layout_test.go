package gopherllm

import "testing"

func TestContiguousBatchRows(t *testing.T) {
	slab := make([]float32, 12)
	contiguous := [][]float32{slab[:4], slab[4:8], slab[8:12]}
	for _, tc := range []struct {
		name  string
		rows  [][]float32
		width int
		want  bool
	}{
		{"contiguous", contiguous, 4, true},
		{"empty", nil, 4, false},
		{"zero width", [][]float32{{}}, 0, false},
		{"short row", [][]float32{slab[:4], slab[4:7]}, 4, false},
		{"bounded first capacity", [][]float32{slab[:4:4], slab[4:8]}, 4, false},
		{"reordered", [][]float32{slab[:4], slab[8:12], slab[4:8]}, 4, false},
		{"separate allocations", [][]float32{make([]float32, 4), make([]float32, 4)}, 4, false},
		{"empty first row", [][]float32{slab[:0], slab[4:8]}, 4, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			flat, ok := contiguousBatchRows(tc.rows, tc.width)
			if ok != tc.want {
				t.Fatalf("contiguous = %v, want %v", ok, tc.want)
			}
			if ok && (len(flat) != len(tc.rows)*tc.width || &flat[0] != &slab[0]) {
				t.Fatal("wrong backing slab")
			}
		})
	}
	if !batchSlicesOverlap(slab[:4], slab[3:8]) || batchSlicesOverlap(slab[:4], slab[4:8]) || batchSlicesOverlap(nil, slab) {
		t.Fatal("incorrect overlap classification")
	}
}
