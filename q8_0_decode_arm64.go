//go:build arm64

package gopherllm

import "math"

// The full-row kernel keeps block scales and reductions in registers instead
// of returning eight integer dots to Go for every 256 input elements. SDOT
// must be probed before the self-check executes any assembly.
var q8_0DecodeAsmOK = hasDotProd && validateQ8_0DecodeAsm()

//go:noescape
func q8_0RowDecodeAsm(row *byte, q8 *int8, scales, lut *float32, blocks int) float32

func q8_0RowDecode(row []byte, q8 []int8, scales []float32, blocks int) float32 {
	if blocks <= 0 {
		return 0
	}
	// Check element counts without multiplying blocks: a malformed count must
	// not overflow an index calculation and reach the unchecked assembly.
	if blocks > len(row)/272 || blocks > len(q8)/256 || blocks > len(scales) {
		panic("gopherllm: short Q8_0 decode buffers")
	}
	return q8_0RowDecodeAsm(&row[0], &q8[0], &scales[0], &f16LUT[0], blocks)
}

func validateQ8_0DecodeAsm() bool {
	const blocks = 2
	row := q8kSelfCheckBytes(272*blocks, 37, 11)
	// Package variable initialization precedes the init function that fills
	// f16LUT, so the self-check supplies an independent scale table.
	lut := [...]float32{0, 0.25, -0.125, 0.0078125}
	for j := range 8 * blocks {
		row[j*34] = byte(j % len(lut))
		row[j*34+1] = 0
	}
	var q8 [blocks * 256]int8
	for i := range q8 {
		q8[i] = int8(i*17 + i/256*13)
	}
	scales := [...]float32{0.013, 0.027}
	got := q8_0RowDecodeAsm(&row[0], &q8[0], &scales[0], &lut[0], blocks)
	var want float32
	for b := range blocks {
		var blockSum float32
		for j := range 8 {
			var dot int32
			for i := range 32 {
				dot += int32(int8(row[b*272+j*34+2+i])) * int32(q8[b*256+j*32+i])
			}
			blockSum += lut[int(row[b*272+j*34])] * float32(dot)
		}
		want += scales[b] * blockSum
	}
	return q8kSelfCheck("Q8_0 decode", []int32{int32(math.Float32bits(got))}, []int32{int32(math.Float32bits(want))})
}
