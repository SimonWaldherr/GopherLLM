//go:build arm64

package gopherllm

import "math"

var q8_0BatchAsmOK = hasDotProd && validateQ8_0BatchAsm()

func validateQ8_0BatchAsm() bool {
	const cols, blocks = 512, 2
	row := q8kSelfCheckBytes(272*blocks, 37, 11)
	lut := []float32{1, 0.25, 0.125}
	for j := range 8 * blocks {
		row[j*34] = byte(j % 3)
		row[j*34+1] = 0
	}
	var x [4 * cols]int8
	var scales [4 * blocks]float32
	for i := range x {
		x[i] = int8(i*17 + i/cols*13)
	}
	for i := range scales {
		scales[i] = float32(i+1) * 0.01
	}
	var got, want [4]float32
	q8_0Rows4Asm(&row[0], &x[0], &scales[0], &lut[0], cols, blocks, &got[0])
	for t := range 4 {
		for b := range blocks {
			var sum float32
			for j := range 8 {
				var dot int32
				for i := range 32 {
					dot += int32(int8(row[b*272+j*34+2+i])) * int32(x[t*cols+b*256+j*32+i])
				}
				sum += lut[int(row[b*272+j*34])] * float32(dot)
			}
			want[t] += scales[t*blocks+b] * sum
		}
	}
	var gotBits, wantBits [4]int32
	for i := range 4 {
		gotBits[i] = int32(math.Float32bits(got[i]))
		wantBits[i] = int32(math.Float32bits(want[i]))
	}
	return q8kSelfCheck("Q8_0 batch4", gotBits[:], wantBits[:])
}

func batchQ8_0Rows4(w Weight, outs [][]float32, q8 []int8, scales []float32, start, end int) bool {
	if !q8_0BatchAsmOK || !q8_0DotAsmOK || w.Type != GGMLTypeQ8_0 || len(outs) < 4 {
		return false
	}
	cols, blocks := w.Cols, w.Cols/256
	for row := start; row < end; row++ {
		raw := w.Raw[row*blocks*272 : (row+1)*blocks*272]
		token := 0
		for ; token+4 <= len(outs); token += 4 {
			values := q8_0RowBatch4(raw, q8[token*cols:], scales[token*blocks:], cols, blocks)
			for i := range 4 {
				outs[token+i][row] = values[i]
			}
		}
		for ; token < len(outs); token++ {
			outs[token][row] = q8_0DotQ8KRow(raw, q8[token*cols:], scales[token*blocks:], blocks)
		}
	}
	return true
}

// Preserve each token's scalar block and scale accumulation order exactly.
//
//go:noescape
func q8_0Rows4Asm(row *byte, q8 *int8, scales, lut *float32, stride, blocks int, out *float32)

func q8_0RowBatch4(row []byte, q8 []int8, scales []float32, cols, blocks int) (out [4]float32) {
	q8_0Rows4Asm(&row[0], &q8[0], &scales[0], &f16LUT[0], cols, blocks, &out[0])
	return
}
