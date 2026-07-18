//go:build amd64

package gopherllm

// q4_0DotQ8KRow and q4_1DotQ8KRow bring the legacy 32-element block formats
// onto the int8-activation full-row path (see q4kDotQ8KRow). blocks counts
// 256-element superchunks; xsums are the per-32-element float sums of the
// original activations (fillQ4KXSums), which carry Q4_0's -8 offset term and
// Q4_1's +m offset term exactly.
//
//go:noescape
func q4_0DotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

//go:noescape
func q4_1DotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

func dotQ4_0RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q4_0DotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

func dotQ4_1RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q4_1DotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}
