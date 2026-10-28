//go:build arm64

package engine

import "math"

//go:noescape
func q4_0Rows4Asm(row *byte, q8 *int8, scales, sums, lut *float32, cols, blocks int, out *float32)

//go:noescape
func q4_1Rows4Asm(row *byte, q8 *int8, scales, sums, lut *float32, cols, blocks int, out *float32)

// Never execute SDOT during startup on an older ARM CPU. A failed numerical
// check also leaves the established per-token kernels in charge.
var q4_01BatchAsmOK = hasDotProd && validateQ4_01Rows4Asm()

// batchQ4_01Rows4 shares each legacy Q4 block's nibble unpack and scale loads
// across four prompt tokens. Every lane retains its original ordered float
// reduction; trailing tokens use the same single-token kernel as before.
func batchQ4_01Rows4(w Weight, outs [][]float32, q8 []int8, scales, sums []float32, start, end int) bool {
	if !q4_01BatchAsmOK || len(outs) < 4 || w.F32 != nil || w.Rows <= 0 || w.Cols <= 0 || w.Cols%256 != 0 || start < 0 || end < start || end > w.Rows {
		return false
	}
	var stride int
	var dot func([]byte, []int8, []float32, []float32, int) float32
	switch w.Type {
	case GGMLTypeQ4_0:
		if !q4_0DotAsmOK {
			return false
		}
		stride, dot = 18, q4_0DotQ8KRow
	case GGMLTypeQ4_1:
		if !q4_1DotAsmOK {
			return false
		}
		stride, dot = 20, q4_1DotQ8KRow
	default:
		return false
	}
	cols, blocks := w.Cols, w.Cols/256
	if cols/32 > len(w.Raw)/stride {
		return false
	}
	rowBytes := cols / 32 * stride
	if w.Rows > len(w.Raw)/rowBytes || len(outs) > len(q8)/cols || len(outs) > len(scales)/blocks || len(outs) > len(sums)/blocks/8 {
		return false
	}
	for _, out := range outs {
		if len(out) < end {
			return false
		}
	}
	for row := start; row < end; row++ {
		raw := w.Raw[row*rowBytes : (row+1)*rowBytes]
		token := 0
		for ; token+4 <= len(outs); token += 4 {
			values := q4_01RowBatch4(w.Type, raw, q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], cols, blocks)
			for i := range 4 {
				outs[token+i][row] = values[i]
			}
		}
		for ; token < len(outs); token++ {
			outs[token][row] = dot(raw, q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
		}
	}
	return true
}

func q4_01RowBatch4(typ GGMLType, row []byte, q8 []int8, scales, sums []float32, cols, blocks int) (out [4]float32) {
	if blocks <= 0 {
		return
	}
	blockBytes := 144
	if typ == GGMLTypeQ4_1 {
		blockBytes = 160
	} else if typ != GGMLTypeQ4_0 {
		panic("gopherllm: invalid legacy Q4 batch type")
	}
	if cols <= 0 || cols%256 != 0 || blocks != cols/256 || blocks > len(row)/blockBytes || len(q8)/cols < 4 || blocks > len(scales)/4 || blocks > len(sums)/32 {
		panic("gopherllm: short legacy Q4 batch buffers")
	}
	if typ == GGMLTypeQ4_0 {
		q4_0Rows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &f16LUT[0], cols, blocks, &out[0])
	} else {
		q4_1Rows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &f16LUT[0], cols, blocks, &out[0])
	}
	return
}

func validateQ4_01Rows4Asm() bool {
	const blocks, cols = 3, 3 * 256
	// Package initialization cannot rely on the shared half lookup table.
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
		sums[i] = float32(i%29-14) * 0.117
	}
	for _, typ := range []GGMLType{GGMLTypeQ4_0, GGMLTypeQ4_1} {
		stride, header := 18, 2
		if typ == GGMLTypeQ4_1 {
			stride, header = 20, 4
		}
		row := q8kSelfCheckBytes(blocks*8*stride, 37, 11)
		var got, want [4]float32
		for g := range blocks * 8 {
			row[g*stride], row[g*stride+1] = byte(g%len(lut)), 0
			if header == 4 {
				row[g*stride+2], row[g*stride+3] = byte((g+1)%len(lut)), 0
			}
			for token := range 4 {
				var dot int32
				for i := range 16 {
					v := row[g*stride+header+i]
					dot += int32(v&15)*int32(q8[token*cols+g*32+i]) + int32(v>>4)*int32(q8[token*cols+g*32+16+i])
				}
				d, scale, sum := lut[g%len(lut)], scales[token*blocks+g/8], sums[token*blocks*8+g]
				if typ == GGMLTypeQ4_0 {
					want[token] += d * (scale*float32(dot) - 8*sum)
				} else {
					want[token] += d*scale*float32(dot) + lut[(g+1)%len(lut)]*sum
				}
			}
		}
		if typ == GGMLTypeQ4_0 {
			q4_0Rows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &lut[0], cols, blocks, &got[0])
		} else {
			q4_1Rows4Asm(&row[0], &q8[0], &scales[0], &sums[0], &lut[0], cols, blocks, &got[0])
		}
		var gotBits, wantBits [4]int32
		for token := range 4 {
			gotBits[token], wantBits[token] = int32(math.Float32bits(got[token])), int32(math.Float32bits(want[token]))
		}
		if !q8kSelfCheck(typ.String()+" full row batch4", gotBits[:], wantBits[:]) {
			return false
		}
	}
	return true
}
