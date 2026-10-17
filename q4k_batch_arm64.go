//go:build arm64

package gopherllm

import "math"

//go:noescape
func q4kQ8Dots8x4Asm(q *byte, q8 *int8, stride int, out *int32)

//go:noescape
func q4kRows4Asm(row *byte, q8 *int8, scales, sums, lut *float32, cols, blocks int, out *float32)

var q4kBatchAsmOK = hasDotProd && validateQ4KBatchAsm() && validateQ4KRows4Asm()

func validateQ4KBatchAsm() bool {
	q := q8kSelfCheckBytes(128, 7, 0)
	var x [4 * 256]int8
	for i := range x {
		x[i] = int8(i*37 + i/256*11)
	}
	var got [32]int32
	q4kQ8Dots8x4Asm(&q[0], &x[0], 256, &got[0])
	var want [32]int32
	for token := range 4 {
		for group := range 4 {
			for i := range 32 {
				v := q[group*32+i]
				want[token*8+2*group] += int32(v&15) * int32(x[token*256+group*64+i])
				want[token*8+2*group+1] += int32(v>>4) * int32(x[token*256+group*64+32+i])
			}
		}
	}
	return q8kSelfCheck("Q4_K batch4", got[:], want[:])
}

// Ministral's Q4_K attention and FFN projections dominate prompt prefill.
// Unpack each block once for four positions without changing quantization or
// float accumulation order. Incomplete groups retain the single-token kernel.
func batchQ4KRows4(w Weight, outs [][]float32, q8 []int8, scales, sums []float32, start, end int) bool {
	if !q4kBatchAsmOK || !q4kDotAsmOK || w.Type != GGMLTypeQ4_K || len(outs) < 4 {
		return false
	}
	cols, blocks := w.Cols, w.Cols/256
	rowBytes := blocks * 144
	for row := start; row < end; row++ {
		weights := w.Raw[row*rowBytes : (row+1)*rowBytes]
		token := 0
		for ; token+4 <= len(outs); token += 4 {
			values := q4kRowBatch4(weights, q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], cols, blocks)
			for i := range 4 {
				outs[token+i][row] = values[i]
			}
		}
		for ; token < len(outs); token++ {
			outs[token][row] = q4kDotQ8KRow(weights, q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
		}
	}
	return true
}

func q4kRowBatch4(row []byte, q8 []int8, scales, sums []float32, cols, blocks int) (result [4]float32) {
	if blocks <= 0 {
		return
	}
	// Check all four token ranges before entering assembly. Division keeps
	// malformed block/stride counts from overflowing an index calculation.
	if cols <= 0 || cols%256 != 0 || blocks != cols/256 || blocks > len(row)/144 ||
		len(q8)/cols < 4 || blocks > len(scales)/4 || blocks > len(sums)/32 {
		panic("gopherllm: short Q4_K batch buffers")
	}
	q4kRows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &f16LUT[0], cols, blocks, &result[0])
	return
}

func validateQ4KRows4Asm() bool {
	const blocks, cols = 3, 3 * 256
	row := q8kSelfCheckBytes(blocks*144, 37, 11)
	// Package variables initialize before f16LUT is populated. A local table
	// exercises positive, negative and zero block multipliers independently.
	lut := [...]float32{0.25, -0.125, 0, 0.03125}
	var q8 [4 * cols]int8
	var scales [4 * blocks]float32
	var sums [4 * blocks * 8]float32
	for i := range q8 {
		q8[i] = int8(i*17 + i/cols*13)
	}
	for i := range scales {
		scales[i] = float32(i+1) * 0.017
	}
	for i := range sums {
		sums[i] = float32((i*17)%29-14) * 0.117
	}
	var got, want [4]float32
	for b := range blocks {
		block := row[b*144 : (b+1)*144]
		block[0], block[1] = byte(b), 0
		block[2], block[3] = byte(b+1), 0
		for token := range 4 {
			// Compute every integer dot independently of the assembly kernels.
			var dots [8]int32
			for group := range 4 {
				for i := range 32 {
					v := block[16+group*32+i]
					dots[2*group] += int32(v&15) * int32(q8[token*cols+b*256+group*64+i])
					dots[2*group+1] += int32(v>>4) * int32(q8[token*cols+b*256+group*64+32+i])
				}
			}
			want[token] += combineQ4KStyle(block[4:16], &dots, sums[(token*blocks+b)*8:], lut[b], lut[b+1], scales[token*blocks+b])
		}
	}
	q4kRows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &lut[0], cols, blocks, &got[0])
	var gotBits, wantBits [4]int32
	for token := range 4 {
		gotBits[token] = int32(math.Float32bits(got[token]))
		wantBits[token] = int32(math.Float32bits(want[token]))
	}
	return q8kSelfCheck("Q4_K full row batch4", gotBits[:], wantBits[:])
}
