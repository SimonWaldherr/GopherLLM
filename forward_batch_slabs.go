package gopherllm

// batchRowsInSlab verifies row provenance against an explicitly supplied Go
// slice. Unlike inferring a slab from row capacity, this also accepts clipped
// views without widening their bounds or spanning separate allocations.
func batchRowsInSlab(rows [][]float32, width int, slab []float32) ([]float32, bool) {
	if len(rows) == 0 || width <= 0 || len(rows) > len(slab)/width {
		return nil, false
	}
	flat := slab[:len(rows)*width]
	for i, row := range rows {
		if len(row) != width || &row[0] != &flat[i*width] {
			return nil, false
		}
	}
	return flat, true
}

// The batch workspace owns the complete input/output slabs. Pass them through
// separately so BLAS can skip packing even though every public row view keeps
// its capacity clipped to one token. Unsupported backends retain normal dispatch.
func matvecBatchSlabs(w Weight, xs, outs [][]float32, input, output []float32) {
	matvecBatchWithQ8AndSlabs(w, xs, outs, useQ8Activations.Load(), input, output)
}

func matvecBatch2Slabs(a, b Weight, xs, aOut, bOut [][]float32, input, aFlat, bFlat []float32) {
	if matvecBatchQ8Fused2(a, b, xs, aOut, bOut) {
		return
	}
	matvecBatchSlabs(a, xs, aOut, input, aFlat)
	matvecBatchSlabs(b, xs, bOut, input, bFlat)
}

func matvecBatch3Slabs(a, b, c Weight, xs, aOut, bOut, cOut [][]float32, input, aFlat, bFlat, cFlat []float32) {
	if matvecBatchQ8Fused3(a, b, c, xs, aOut, bOut, cOut) {
		return
	}
	matvecBatchSlabs(a, xs, aOut, input, aFlat)
	matvecBatchSlabs(b, xs, bOut, input, bFlat)
	matvecBatchSlabs(c, xs, cOut, input, cFlat)
}
