//go:build arm64

package gopherllm

import "math"

// Four prompt positions share the same Q6_K weight unpack. Each SIMD lane
// keeps one token's original ordered float accumulation independently.
var q6kBatchAsmOK = hasDotProd && validateQ6KBatchAsm()

//go:noescape
func q6kRows4Asm(row *byte, q8 *int8, scales, sums, lut *float32, cols, blocks int, out *float32)

func q6kRowBatch4(row []byte, q8 []int8, scales, sums []float32, cols, blocks int) (out [4]float32) {
	if blocks <= 0 {
		return
	}
	// Check the complete four-token ranges before assembly; division prevents
	// malformed block/stride counts from wrapping an index calculation.
	if cols <= 0 || cols%256 != 0 || blocks != cols/256 || blocks > len(row)/210 ||
		len(q8)/cols < 4 || blocks > len(scales)/4 || blocks > len(sums)/64 {
		panic("gopherllm: short Q6_K batch buffers")
	}
	q6kRows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &f16LUT[0], cols, blocks, &out[0])
	return
}

func batchQ6KRows4(w Weight, outs [][]float32, q8 []int8, scales, sums []float32, start, end int) bool {
	if !q6kBatchAsmOK || !q6kDotAsmOK || w.Type != GGMLTypeQ6_K || len(outs) < 4 {
		return false
	}
	cols, blocks := w.Cols, w.Cols/256
	for row := start; row < end; row++ {
		raw := w.Raw[row*blocks*210 : (row+1)*blocks*210]
		token := 0
		for ; token+4 <= len(outs); token += 4 {
			values := q6kRowBatch4(raw, q8[token*cols:], scales[token*blocks:], sums[token*blocks*16:], cols, blocks)
			for i := range 4 {
				outs[token+i][row] = values[i]
			}
		}
		for ; token < len(outs); token++ {
			outs[token][row] = q6kDotQ8KRow(raw, q8[token*cols:], scales[token*blocks:], sums[token*blocks*16:], blocks)
		}
	}
	return true
}

func validateQ6KBatchAsm() bool {
	const blocks, cols = 3, 3 * 256
	row := q8kSelfCheckBytes(blocks*210, 37, 11)
	// f16LUT is not initialized yet. A local table tests signed and zero scales.
	lut := [...]float32{0.25, -0.125, 0}
	var q8 [4 * cols]int8
	var scales [4 * blocks]float32
	var sums [4 * blocks * 16]float32
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
		block := row[b*210 : (b+1)*210]
		block[208], block[209] = byte(b), 0
		for token := range 4 {
			var dots [16]int32
			for half := range 2 {
				for l := range 32 {
					lo, lo2, hi := block[half*64+l], block[half*64+l+32], block[128+half*32+l]
					quants := [4]byte{lo&15 | (hi&3)<<4, lo2&15 | ((hi>>2)&3)<<4,
						lo>>4 | ((hi>>4)&3)<<4, lo2>>4 | (hi>>6)<<4}
					for g, q := range quants {
						dots[half*8+g*2+l/16] += int32(q) * int32(q8[token*cols+b*256+half*128+g*32+l])
					}
				}
			}
			var dot int32
			var offset float32
			for i := range 16 {
				s := int32(int8(block[192+i]))
				dot += s * dots[i]
				offset += float32(s) * sums[(token*blocks+b)*16+i]
			}
			want[token] += lut[b] * (scales[token*blocks+b]*float32(dot) - offset)
		}
	}
	q6kRows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &lut[0], cols, blocks, &got[0])
	var gotBits, wantBits [4]int32
	for token := range 4 {
		gotBits[token] = int32(math.Float32bits(got[token]))
		wantBits[token] = int32(math.Float32bits(want[token]))
	}
	return q8kSelfCheck("Q6_K batch4", gotBits[:], wantBits[:])
}
