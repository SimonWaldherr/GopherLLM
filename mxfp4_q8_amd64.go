//go:build amd64

package gopherllm

// mxfp4DotQ8KRow is the MXFP4 (gpt-oss) member of the int8-activation
// full-row kernel family: FP4 e2m1 magnitudes are doubled into exact small
// integers via a VPSHUFB lookup, signs ride VPSIGNB on the activation
// operand, and the E8M0 block scale (x0.5 to undo the doubling) folds into
// the float accumulation. Symmetric format — no xsums offset term.
//
//go:noescape
func mxfp4DotQ8KRow(row *byte, q8 *int8, xscales *float32, blocks int) float32

func dotMXFP4RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = mxfp4DotQ8KRow(&data[off], &q8[0], &xscale[0], blocks)
	}
}
