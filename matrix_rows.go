package gopherllm

// resizeMatrixRows reuses a contiguous float32 workspace and its row views.
// Rows retain the remaining slab capacity so BLAS can use them without packing.
func resizeMatrixRows(data *[]float32, rows *[][]float32, count, width int, zero bool) [][]float32 {
	if cap(*data) < count*width {
		// Inactive views must not retain a slab replaced by fewer, wider rows.
		clear((*rows)[:cap(*rows)])
		*data = make([]float32, count*width)
	} else {
		*data = (*data)[:count*width]
		if zero {
			clear(*data)
		}
	}
	ensureLenNoClear(rows, count)
	for i := range *rows {
		(*rows)[i] = (*data)[i*width : (i+1)*width]
	}
	return *rows
}
