//go:build arm64

package gopherllm

//go:noescape
func q4kQ8Dots8x4Asm(q *byte, q8 *int8, stride int, out *int32)

var q4kBatchAsmOK = hasDotProd && validateQ4KBatchAsm()

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
	var dots [4][8]int32
	for block := range blocks {
		weights := row[block*144 : (block+1)*144]
		q4kQ8Dots8x4Asm(&weights[16], &q8[block*256], cols, &dots[0][0])
		d, dmin := F16ToF32(binaryLE16(weights)), F16ToF32(binaryLE16(weights[2:]))
		packed := weights[4:16]
		s0, m0 := int32(packed[0]&63), int32(packed[4]&63)
		s1, m1 := int32(packed[1]&63), int32(packed[5]&63)
		s2, m2 := int32(packed[2]&63), int32(packed[6]&63)
		s3, m3 := int32(packed[3]&63), int32(packed[7]&63)
		s4 := int32((packed[8] & 0x0f) | ((packed[0] >> 6) << 4))
		m4 := int32((packed[8] >> 4) | ((packed[4] >> 6) << 4))
		s5 := int32((packed[9] & 0x0f) | ((packed[1] >> 6) << 4))
		m5 := int32((packed[9] >> 4) | ((packed[5] >> 6) << 4))
		s6 := int32((packed[10] & 0x0f) | ((packed[2] >> 6) << 4))
		m6 := int32((packed[10] >> 4) | ((packed[6] >> 6) << 4))
		s7 := int32((packed[11] & 0x0f) | ((packed[3] >> 6) << 4))
		m7 := int32((packed[11] >> 4) | ((packed[7] >> 6) << 4))
		for token := range 4 {
			xsums := sums[(token*blocks+block)*8:]
			_ = xsums[7]
			var blockInt int32
			blockInt += s0 * dots[token][0]
			blockInt += s1 * dots[token][1]
			blockInt += s2 * dots[token][2]
			blockInt += s3 * dots[token][3]
			blockInt += s4 * dots[token][4]
			blockInt += s5 * dots[token][5]
			blockInt += s6 * dots[token][6]
			blockInt += s7 * dots[token][7]
			var minTerm float32
			minTerm += float32(m0) * xsums[0]
			minTerm += float32(m1) * xsums[1]
			minTerm += float32(m2) * xsums[2]
			minTerm += float32(m3) * xsums[3]
			minTerm += float32(m4) * xsums[4]
			minTerm += float32(m5) * xsums[5]
			minTerm += float32(m6) * xsums[6]
			minTerm += float32(m7) * xsums[7]
			result[token] += d*scales[token*blocks+block]*float32(blockInt) - dmin*minTerm
		}
	}
	return
}
