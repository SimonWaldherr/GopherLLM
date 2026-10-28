//go:build arm64

package engine

//go:noescape
func gqa4WeightedValuesF32Asm(out, values, scores *float32, dim, stride, span, rows int)

// gqa4WeightedValuesF32 keeps one 16-channel tile from each head in registers
// across the chronological V-row traversal. Each channel retains the original
// ordered FMA sequence, while the output tile is loaded and stored only once.
func gqa4WeightedValuesF32(out, values, scores []float32, dim, stride, span, startT, rows int) bool {
	if dim != 128 || stride < dim || rows <= 0 || span < rows || startT < 0 || dim > len(out)/4 || span > len(scores)/4 || startT > len(values)/stride {
		return false
	}
	offset := startT * stride
	if len(values)-offset < dim {
		return true // The original row traversal stops before an incomplete row.
	}
	rows = min(rows, 1+(len(values)-offset-dim)/stride)
	// Keep very short, unmeasured spans on the existing single-row kernel.
	if rows < 16 {
		return false
	}
	// At long contexts, a compact contiguous V stream already makes good use
	// of hardware prefetching. Keep that path and other unmeasured layouts on
	// the existing row kernel; the 1024-float stride is Ministral's eight KV
	// heads and retains the measured benefit through 32k positions.
	if rows > 4096 && stride != 1024 {
		return false
	}
	// Bound each repeated channel traversal to a small position tile. Walking
	// an entire long context once per channel tile would evict earlier V rows
	// from cache, especially when the KV heads use a large row stride.
	const positionsPerTile = 256
	for first := 0; first < rows; first += positionsPerTile {
		count := min(rows-first, positionsPerTile)
		gqa4WeightedValuesF32Asm(&out[0], &values[offset+first*stride], &scores[first], dim, stride, span, count)
	}
	return true
}
