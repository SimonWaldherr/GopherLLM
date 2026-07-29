package gopherllm

// Portable kernels for GGML's ternary TQ1_0/TQ2_0 and codebook-free Q1_0/Q2_0
// formats. All four use compact blocks whose values can be consumed directly
// during matvec without allocating a dequantized copy.

var tq1Digits = func() (table [256][5]int8) {
	pow3 := [...]int{1, 3, 9, 27, 81}
	for packed := range 256 {
		for digit := range 5 {
			table[packed][digit] = int8(((int(byte(packed*pow3[digit])) * 3) >> 8) - 1)
		}
	}
	return table
}()

// DequantRowTQ1_0Into decodes 256 ternary weights from each 54-byte block.
// The first 48 bytes hold five scaled base-3 digits apiece, the next four hold
// four digits apiece, and the final two bytes are the f16 scale.
func DequantRowTQ1_0Into(row []byte, cols int, out []float32) {
	blocks := min(cols/256, len(row)/54, len(out)/256)
	for b := 0; b < blocks; b++ {
		block := row[b*54 : b*54+54]
		d := F16ToF32(binaryLE16(block[52:]))
		dst := out[b*256 : b*256+256]
		at := 0
		for group := 0; group < 2; group++ {
			base, width := group*32, 32
			if group == 1 {
				width = 16
			}
			for digit := 0; digit < 5; digit++ {
				for j := 0; j < width; j++ {
					dst[at] = float32(tq1Digits[block[base+j]][digit]) * d
					at++
				}
			}
		}
		for digit := 0; digit < 4; digit++ {
			for j := 0; j < 4; j++ {
				dst[at] = float32(tq1Digits[block[48+j]][digit]) * d
				at++
			}
		}
	}
}

func DequantRowTQ1_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowTQ1_0Into(row, cols, out)
	return out
}

func DotTQ1_0F32(row []byte, x []float32, cols int) float32 {
	blocks := min(cols/256, len(row)/54, len(x)/256)
	var sum float32
	for b := 0; b < blocks; b++ {
		block := row[b*54 : b*54+54]
		d := F16ToF32(binaryLE16(block[52:]))
		xb := x[b*256 : b*256+256]
		at := 0
		var acc float32
		for group := 0; group < 2; group++ {
			base, width := group*32, 32
			if group == 1 {
				width = 16
			}
			for digit := 0; digit < 5; digit++ {
				for j := 0; j < width; j++ {
					acc += float32(tq1Digits[block[base+j]][digit]) * xb[at]
					at++
				}
			}
		}
		for digit := 0; digit < 4; digit++ {
			for j := 0; j < 4; j++ {
				acc += float32(tq1Digits[block[48+j]][digit]) * xb[at]
				at++
			}
		}
		sum += d * acc
	}
	return sum
}

func MatvecTQ1_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 54
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotTQ1_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

// DequantRowTQ2_0Into decodes 256 ternary weights from each 66-byte block.
// Each 32-byte group stores four 2-bit planes in output order; the final two
// bytes hold the f16 scale.
func DequantRowTQ2_0Into(row []byte, cols int, out []float32) {
	blocks := min(cols/256, len(row)/66, len(out)/256)
	for b := 0; b < blocks; b++ {
		block := row[b*66 : b*66+66]
		d := F16ToF32(binaryLE16(block[64:]))
		dst := out[b*256 : b*256+256]
		at := 0
		for base := 0; base < 64; base += 32 {
			for plane := 0; plane < 4; plane++ {
				shift := plane * 2
				for j := 0; j < 32; j++ {
					q := int((block[base+j]>>shift)&3) - 1
					dst[at] = float32(q) * d
					at++
				}
			}
		}
	}
}

func DequantRowTQ2_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowTQ2_0Into(row, cols, out)
	return out
}

func DotTQ2_0F32(row []byte, x []float32, cols int) float32 {
	blocks := min(cols/256, len(row)/66, len(x)/256)
	var sum float32
	for b := 0; b < blocks; b++ {
		block := row[b*66 : b*66+66]
		d := F16ToF32(binaryLE16(block[64:]))
		xb := x[b*256 : b*256+256]
		at := 0
		var acc float32
		for base := 0; base < 64; base += 32 {
			for plane := 0; plane < 4; plane++ {
				shift := plane * 2
				for j := 0; j < 32; j++ {
					q := int((block[base+j]>>shift)&3) - 1
					acc += float32(q) * xb[at]
					at++
				}
			}
		}
		sum += d * acc
	}
	return sum
}

func MatvecTQ2_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 66
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotTQ2_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

// DequantRowQ1_0Into decodes 128 weights from each 18-byte block: one f16
// scale followed by 128 sign bits. A set bit is +scale and a clear bit is
// -scale.
func DequantRowQ1_0Into(row []byte, cols int, out []float32) {
	blocks := min(cols/128, len(row)/18, len(out)/128)
	for b := 0; b < blocks; b++ {
		block := row[b*18 : b*18+18]
		d := F16ToF32(binaryLE16(block))
		dst := out[b*128 : b*128+128]
		for j, bits := range block[2:] {
			base := j * 8
			for bit := range 8 {
				if bits&(1<<bit) != 0 {
					dst[base+bit] = d
				} else {
					dst[base+bit] = -d
				}
			}
		}
	}
}

func DequantRowQ1_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ1_0Into(row, cols, out)
	return out
}

func DotQ1_0F32(row []byte, x []float32, cols int) float32 {
	blocks := min(cols/128, len(row)/18, len(x)/128)
	var sum float32
	for b := 0; b < blocks; b++ {
		block := row[b*18 : b*18+18]
		d := F16ToF32(binaryLE16(block))
		xb := x[b*128 : b*128+128]
		var acc float32
		for j, bits := range block[2:] {
			base := j * 8
			for bit := range 8 {
				if bits&(1<<bit) != 0 {
					acc += xb[base+bit]
				} else {
					acc -= xb[base+bit]
				}
			}
		}
		sum += d * acc
	}
	return sum
}

func MatvecQ1_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 128) * 18
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ1_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

// DequantRowQ2_0Into decodes 64 weights from each 18-byte block. Four
// consecutive weights occupy each byte and map 00/01/10/11 to -1/0/+1/+2
// times the block's f16 scale.
func DequantRowQ2_0Into(row []byte, cols int, out []float32) {
	blocks := min(cols/64, len(row)/18, len(out)/64)
	for b := 0; b < blocks; b++ {
		block := row[b*18 : b*18+18]
		d := F16ToF32(binaryLE16(block))
		dst := out[b*64 : b*64+64]
		for j, packed := range block[2:] {
			base := j * 4
			dst[base] = float32(int(packed&3)-1) * d
			dst[base+1] = float32(int((packed>>2)&3)-1) * d
			dst[base+2] = float32(int((packed>>4)&3)-1) * d
			dst[base+3] = float32(int(packed>>6)-1) * d
		}
	}
}

func DequantRowQ2_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ2_0Into(row, cols, out)
	return out
}

func DotQ2_0F32(row []byte, x []float32, cols int) float32 {
	blocks := min(cols/64, len(row)/18, len(x)/64)
	var sum float32
	for b := 0; b < blocks; b++ {
		block := row[b*18 : b*18+18]
		d := F16ToF32(binaryLE16(block))
		xb := x[b*64 : b*64+64]
		var acc float32
		for j, packed := range block[2:] {
			base := j * 4
			acc += float32(int(packed&3)-1)*xb[base] +
				float32(int((packed>>2)&3)-1)*xb[base+1] +
				float32(int((packed>>4)&3)-1)*xb[base+2] +
				float32(int(packed>>6)-1)*xb[base+3]
		}
		sum += d * acc
	}
	return sum
}

func MatvecQ2_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 64) * 18
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ2_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}
