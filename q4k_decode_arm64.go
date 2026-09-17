//go:build arm64

package gopherllm

import "math"

// Keep the integer dots, packed scales and row reduction in registers across
// all blocks. The existing block kernel remains the fallback and reference.
var q4kDecodeAsmOK = hasDotProd && validateQ4KDecodeAsm()

//go:noescape
func q4kRowDecodeAsm(row *byte, q8 *int8, scales, sums, lut *float32, blocks int) float32

func q4kRowDecode(row []byte, q8 []int8, scales, sums []float32, blocks int) float32 {
	if blocks <= 0 {
		return 0
	}
	// Validate complete ranges without overflowing when blocks is malformed.
	if blocks > len(row)/144 || blocks > len(q8)/256 || blocks > len(scales) || blocks > len(sums)/8 {
		panic("q4kRowDecode: input shorter than block count")
	}
	return q4kRowDecodeAsm(&row[0], &q8[0], &scales[0], &sums[0], &f16LUT[0], blocks)
}

func validateQ4KDecodeAsm() bool {
	const blocks = 3
	row := q8kSelfCheckBytes(blocks*144, 37, 11)
	// Package variables initialize before f16LUT is filled in init(). Use a
	// local lookup table with positive, negative and zero block multipliers.
	lut := [...]float32{0.25, -0.125, 0, 0.03125}
	var q8 [blocks * 256]int8
	var sums [blocks * 8]float32
	scales := [...]float32{0.013, 0.027, 0.019}
	for i := range q8 {
		q8[i] = int8(i*17 + i/256*13)
	}
	for i := range sums {
		sums[i] = float32(i%13-6) * 0.117
	}
	var want float32
	for b := range blocks {
		block := row[b*144 : (b+1)*144]
		block[0], block[1] = byte(b), 0
		block[2], block[3] = byte(b+1), 0
		// Independent scalar dots exercise every packed nibble and signed
		// activation without relying on another assembly self-check.
		var dots [8]int32
		for group := range 4 {
			for i := range 32 {
				v := block[16+group*32+i]
				dots[2*group] += int32(v&15) * int32(q8[b*256+group*64+i])
				dots[2*group+1] += int32(v>>4) * int32(q8[b*256+group*64+32+i])
			}
		}
		want += combineQ4KStyle(block[4:16], &dots, sums[b*8:], lut[b], lut[b+1], scales[b])
	}
	got := q4kRowDecodeAsm(&row[0], &q8[0], &scales[0], &sums[0], &lut[0], blocks)
	return q8kSelfCheck("Q4_K full row", []int32{int32(math.Float32bits(got))}, []int32{int32(math.Float32bits(want))})
}
