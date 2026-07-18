//go:build amd64

package gopherllm

// q5kDotQ8KRow is the Q5_K analogue of q4kDotQ8KRow: a full-row
// int8-activation dot product over 176-byte Q5_K superblocks (the Q4_K
// layout plus a 32-byte fifth-bit plane). xsums are the per-32-element float
// sums of the ORIGINAL activations (fillQ4KXSums — Q5_K shares Q4_K's
// per-sub-block dmin/min structure exactly).
//
//go:noescape
func q5kDotQ8KRow(q *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

// dotQ5KRowsQ8 fills out[start:end] with Q5_K row dots against Q8K-quantized
// activations.
func dotQ5KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q5kDotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}
