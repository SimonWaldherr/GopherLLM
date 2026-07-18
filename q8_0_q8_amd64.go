//go:build amd64

package gopherllm

// q8_0DotQ8KRow is the Q8_0 analogue of q4kDotQ8KRow/q6kDotQ8KRow: a full-row
// int8-activation dot product against Q8_0-quantized weights (34-byte
// blocks: one f16 scale + 32 signed int8 values, no min/offset term). row
// must point at 8*34=272 bytes per "block" (one Q8K activation super-block
// of 256 elements); q8/xscales are as in the K-quant kernels.
//
//go:noescape
func q8_0DotQ8KRow(row *byte, q8 *int8, xscales *float32, blocks int) float32

// dotQ8_0RowsQ8 fills out[start:end] with Q8_0 row dots against
// Q8K-quantized activations.
func dotQ8_0RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q8_0DotQ8KRow(&data[off], &q8[0], &xscale[0], blocks)
	}
}
