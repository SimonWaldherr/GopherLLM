//go:build arm64

package gopherllm

import "math"

var q8_0PackedAsmOK = hasDotProd && validateQ8_0PackedAsm()

//go:noescape
func q8_0Pack4Asm(src, dst *int8, cols int)

//go:noescape
func q8_0Rows4PackedAsm(row *byte, packed *int8, scales, lut *float32, blocks int, out *float32)

// prepareQ8_0PackedBatch interleaves four already-quantized token vectors in
// groups of four bytes. A SIMD lane then holds one token's dot, avoiding the
// horizontal reductions and lane packing in every weight row. Complete token
// groups occupy contiguous 4*cols regions; trailing tokens stay in q8.
func prepareQ8_0PackedBatch(dst *[]int8, q8 []int8, cols, tokens int) bool {
	if !q8_0PackedAsmOK || !q8_0DotAsmOK || cols <= 0 || cols%256 != 0 || tokens < 4 || tokens > len(q8)/cols {
		return false
	}
	complete := tokens &^ 3
	ensureLenNoClear(dst, complete*cols)
	for token := 0; token < complete; token += 4 {
		off := token * cols
		q8_0Pack4Asm(&q8[off], &(*dst)[off], cols)
	}
	return true
}

func q8_0RowBatch4Packed(row []byte, packed []int8, scales []float32, blocks int) (out [4]float32) {
	if blocks <= 0 {
		return
	}
	if blocks > len(row)/272 || blocks > len(packed)/1024 || blocks > len(scales)/4 {
		panic("gopherllm: short packed Q8_0 batch buffers")
	}
	q8_0Rows4PackedAsm(&row[0], &packed[0], &scales[0], &f16LUT[0], blocks, &out[0])
	return
}

func batchQ8_0PackedRows4(w Weight, outs [][]float32, q8, packed []int8, scales []float32, start, end int) bool {
	if !q8_0PackedAsmOK || !q8_0DotAsmOK || w.Type != GGMLTypeQ8_0 || w.Cols <= 0 || w.Cols%256 != 0 || len(outs) < 4 {
		return false
	}
	cols, blocks := w.Cols, w.Cols/256
	if len(packed)/cols < len(outs)&^3 || len(q8)/cols < len(outs) || len(scales)/blocks < len(outs) {
		return false
	}
	// Establish complete input/output ranges before writing any results. The
	// q8 size check above also bounds cols, so blocks*272 cannot overflow.
	rowBytes := blocks * 272
	if w.Rows <= 0 || start < 0 || end < start || end > w.Rows || w.Rows > len(w.Raw)/rowBytes {
		return false
	}
	for _, out := range outs {
		if len(out) < w.Rows {
			return false
		}
	}
	for row := start; row < end; row++ {
		raw := w.Raw[row*blocks*272 : (row+1)*blocks*272]
		token := 0
		for ; token+4 <= len(outs); token += 4 {
			values := q8_0RowBatch4Packed(raw, packed[token*cols:], scales[token*blocks:], blocks)
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

func validateQ8_0PackedAsm() bool {
	const blocks, cols = 2, 512
	row := q8kSelfCheckBytes(272*blocks, 37, 11)
	lut := [...]float32{0, 0.25, -0.125, 0.0078125}
	for j := range 8 * blocks {
		row[j*34], row[j*34+1] = byte(j%len(lut)), 0
	}
	var q8, packed [4 * cols]int8
	var scales [4 * blocks]float32
	for i := range q8 {
		q8[i] = int8(i*17 + i/cols*13)
	}
	for i := range scales {
		scales[i] = float32(i-3) * 0.013
	}
	q8_0Pack4Asm(&q8[0], &packed[0], cols)
	var got, want [4]float32
	q8_0Rows4PackedAsm(&row[0], &packed[0], &scales[0], &lut[0], blocks, &got[0])
	for token := range 4 {
		for b := range blocks {
			var blockSum float32
			for j := range 8 {
				var dot int32
				for i := range 32 {
					dot += int32(int8(row[b*272+j*34+2+i])) * int32(q8[token*cols+b*256+j*32+i])
				}
				blockSum += lut[int(row[b*272+j*34])] * float32(dot)
			}
			want[token] += scales[token*blocks+b] * blockSum
		}
	}
	var gotBits, wantBits [4]int32
	for i := range 4 {
		gotBits[i], wantBits[i] = int32(math.Float32bits(got[i])), int32(math.Float32bits(want[i]))
	}
	return q8kSelfCheck("Q8_0 packed batch4", gotBits[:], wantBits[:])
}
