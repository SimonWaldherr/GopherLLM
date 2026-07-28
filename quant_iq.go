package gopherllm

// Scalar IQ2_S and IQ3_S dequantization kernels. The block codebooks are
// embedded in iq_codebooks_generated.go from the corresponding MIT-licensed
// llama.cpp reference tables. These formats trade a compact per-block index
// stream for indirect lookup, so the portable path intentionally favors exact
// decoding and row parallelism over a premature SIMD approximation.

func iqSign(signs byte, index int) float32 {
	if signs&(1<<index) != 0 {
		return -1
	}
	return 1
}

func iq2SGridValue(index, component int) float32 {
	return float32(byte(iq2SGrid[index] >> (8 * component)))
}

func iq3SGridValue(index, component int) float32 {
	return float32(byte(iq3SGrid[index] >> (8 * component)))
}

// DequantRowIQ2SInto decodes GGML's IQ2_S blocks. A 256-value block contains
// a base f16 scale, 32 codebook bytes, 32 sign bytes, 8 high-index bytes, and
// 8 packed sub-block scales (82 bytes total).
func DequantRowIQ2SInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 82
		if base+82 > len(row) {
			break
		}
		block := row[base : base+82]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+32]
		signs := block[2+32 : 2+64]
		qh := block[2+64 : 2+72]
		scales := block[2+72 : 2+80]
		dst := out[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			db0 := d * (0.5 + float32(scales[ib32]&0x0f)) * 0.25
			db1 := d * (0.5 + float32(scales[ib32]>>4)) * 0.25
			for l := 0; l < 4; l++ {
				scale := db0
				if l >= 2 {
					scale = db1
				}
				index := int(qs[l]) | ((int(qh[ib32]) << (8 - 2*l)) & 0x300)
				for j := 0; j < 8; j++ {
					dst[at+j] = scale * iq2SGridValue(index, j) * iqSign(signs[l], j)
				}
				at += 8
			}
			qs = qs[4:]
			signs = signs[4:]
		}
	}
}

func DequantRowIQ2S(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowIQ2SInto(row, cols, out)
	return out
}

func DotIQ2SF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 82
		if base+82 > len(row) {
			break
		}
		block := row[base : base+82]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+32]
		signs := block[2+32 : 2+64]
		qh := block[2+64 : 2+72]
		scales := block[2+72 : 2+80]
		xBlock := x[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			db0 := d * (0.5 + float32(scales[ib32]&0x0f)) * 0.25
			db1 := d * (0.5 + float32(scales[ib32]>>4)) * 0.25
			for l := 0; l < 4; l++ {
				scale := db0
				if l >= 2 {
					scale = db1
				}
				index := int(qs[l]) | ((int(qh[ib32]) << (8 - 2*l)) & 0x300)
				for j := 0; j < 8; j++ {
					sum += scale * iq2SGridValue(index, j) * iqSign(signs[l], j) * xBlock[at+j]
				}
				at += 8
			}
			qs = qs[4:]
			signs = signs[4:]
		}
	}
	return sum
}

func MatvecIQ2SInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 82
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotIQ2SF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

// DequantRowIQ3SInto decodes GGML's IQ3_S blocks. Each 256-value block uses
// 64 low codebook bytes, 8 high-index bytes, 32 sign bytes, four packed
// scales, and one f16 base scale (110 bytes total).
func DequantRowIQ3SInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 110
		if base+110 > len(row) {
			break
		}
		block := row[base : base+110]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		qh := block[2+64 : 2+72]
		signs := block[2+72 : 2+104]
		scales := block[2+104 : 2+108]
		dst := out[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32 += 2 {
			db := [2]float32{
				d * (1 + 2*float32(scales[ib32/2]&0x0f)),
				d * (1 + 2*float32(scales[ib32/2]>>4)),
			}
			for half := 0; half < 2; half++ {
				for l := 0; l < 4; l++ {
					index1 := int(qs[2*l]) | ((int(qh[half]) << (8 - 2*l)) & 0x100)
					index2 := int(qs[2*l+1]) | ((int(qh[half]) << (7 - 2*l)) & 0x100)
					for j := 0; j < 4; j++ {
						dst[at+j] = db[half] * iq3SGridValue(index1, j) * iqSign(signs[l], j)
						dst[at+4+j] = db[half] * iq3SGridValue(index2, j) * iqSign(signs[l], j+4)
					}
					at += 8
				}
				qs = qs[8:]
				signs = signs[4:]
			}
			qh = qh[2:]
		}
	}
}

func DequantRowIQ3S(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowIQ3SInto(row, cols, out)
	return out
}

func DotIQ3SF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 110
		if base+110 > len(row) {
			break
		}
		block := row[base : base+110]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		qh := block[2+64 : 2+72]
		signs := block[2+72 : 2+104]
		scales := block[2+104 : 2+108]
		xBlock := x[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32 += 2 {
			db := [2]float32{
				d * (1 + 2*float32(scales[ib32/2]&0x0f)),
				d * (1 + 2*float32(scales[ib32/2]>>4)),
			}
			for half := 0; half < 2; half++ {
				for l := 0; l < 4; l++ {
					index1 := int(qs[2*l]) | ((int(qh[half]) << (8 - 2*l)) & 0x100)
					index2 := int(qs[2*l+1]) | ((int(qh[half]) << (7 - 2*l)) & 0x100)
					for j := 0; j < 4; j++ {
						sum += db[half] * iq3SGridValue(index1, j) * iqSign(signs[l], j) * xBlock[at+j]
						sum += db[half] * iq3SGridValue(index2, j) * iqSign(signs[l], j+4) * xBlock[at+4+j]
					}
					at += 8
				}
				qs = qs[8:]
				signs = signs[4:]
			}
			qh = qh[2:]
		}
	}
	return sum
}

func MatvecIQ3SInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 110
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotIQ3SF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}
