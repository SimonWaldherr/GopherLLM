package gopherllm

import "unsafe"

// Packed projections share these bounds with callers that split large token
// batches. Keeping a tail above the minimum avoids dropping out of BLAS.
const prefillBLASInputElements = 2 << 20
const prefillBLASPackedMinTokens = 128

// contiguousBatchRows returns the slab behind equal-width row views. Requiring
// the first row's capacity to cover the whole slab proves it is one allocation;
// adjacent but independently allocated rows must never be spanned by a slice.
func contiguousBatchRows(rows [][]float32, width int) ([]float32, bool) {
	if len(rows) == 0 || width <= 0 || len(rows) > cap(rows[0])/width {
		return nil, false
	}
	flat := rows[0][:len(rows)*width]
	for i, row := range rows {
		if len(row) != width || &row[0] != &flat[i*width] {
			return nil, false
		}
	}
	return flat, true
}

// Comparing addresses does not dereference them or construct new pointers.
// BLAS must read a stable input when caller-owned output views overlap it.
func batchSlicesOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	startA, startB := uintptr(unsafe.Pointer(&a[0])), uintptr(unsafe.Pointer(&b[0]))
	if startA <= startB {
		return startB-startA < uintptr(len(a))*unsafe.Sizeof(a[0])
	}
	return startA-startB < uintptr(len(b))*unsafe.Sizeof(b[0])
}
