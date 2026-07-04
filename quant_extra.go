package main

// Compatibility kernels for the less-common GGUF quantization formats:
// the legacy simple quants Q4_1/Q5_0/Q5_1/Q8_1 and the small K-quants
// Q2_K/Q3_K. Block layouts are documented on GGMLType.DataSize and follow
// llama.cpp's ggml-quants reference implementations. These are portable
// scalar kernels only — the formats appear in older or extremely
// memory-constrained files, not in the mainline Q4_K_M/Q6_K models the
// SIMD fast paths target — but they still parallelize across rows through
// the worker pool, and each is verified by a dot-vs-dequantized-dot
// differential test in quant_extra_test.go.

// --- Q4_1: val = d*q + m, q an unsigned nibble (low 16 then high 16) ---

func DequantRowQ4_1Into(row []byte, cols int, out []float32) {
	for b := 0; b < cols/32; b++ {
		base := b * 20
		if base+20 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		m := F16ToF32(binaryLE16(row[base+2:]))
		for i := 0; i < 16; i++ {
			packed := row[base+4+i]
			out[b*32+i] = d*float32(packed&0x0f) + m
			out[b*32+16+i] = d*float32(packed>>4) + m
		}
	}
}

func DequantRowQ4_1(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ4_1Into(row, cols, out)
	return out
}

func DotQ4_1F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/32; b++ {
		base := b * 20
		if base+20 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		m := F16ToF32(binaryLE16(row[base+2:]))
		xBlock := x[b*32 : b*32+32]
		_ = xBlock[31]
		var qdot, xsum float32
		for i := 0; i < 16; i++ {
			packed := row[base+4+i]
			qdot += float32(packed&0x0f)*xBlock[i] + float32(packed>>4)*xBlock[16+i]
			xsum += xBlock[i] + xBlock[16+i]
		}
		sum += d*qdot + m*xsum
	}
	return sum
}

// --- Q5_0: val = d*((q | 5thBit<<4) - 16) ---

// q5Bits reconstructs the two 5-bit values of element j (low half and high
// half) from the packed nibble byte and the 32-bit high-bit plane.
func q5Bits(packed byte, qh uint32, j int) (lo, hi byte) {
	xh0 := byte((qh>>j)<<4) & 0x10
	xh1 := byte(qh>>(j+12)) & 0x10
	return (packed & 0x0f) | xh0, (packed >> 4) | xh1
}

func DequantRowQ5_0Into(row []byte, cols int, out []float32) {
	for b := 0; b < cols/32; b++ {
		base := b * 22
		if base+22 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		qh := uint32(row[base+2]) | uint32(row[base+3])<<8 | uint32(row[base+4])<<16 | uint32(row[base+5])<<24
		for j := 0; j < 16; j++ {
			lo, hi := q5Bits(row[base+6+j], qh, j)
			out[b*32+j] = d * float32(int(lo)-16)
			out[b*32+16+j] = d * float32(int(hi)-16)
		}
	}
}

func DequantRowQ5_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ5_0Into(row, cols, out)
	return out
}

func DotQ5_0F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/32; b++ {
		base := b * 22
		if base+22 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		qh := uint32(row[base+2]) | uint32(row[base+3])<<8 | uint32(row[base+4])<<16 | uint32(row[base+5])<<24
		xBlock := x[b*32 : b*32+32]
		_ = xBlock[31]
		var acc float32
		for j := 0; j < 16; j++ {
			lo, hi := q5Bits(row[base+6+j], qh, j)
			acc += float32(int(lo)-16)*xBlock[j] + float32(int(hi)-16)*xBlock[16+j]
		}
		sum += d * acc
	}
	return sum
}

// --- Q5_1: val = d*(q | 5thBit<<4) + m ---

func DequantRowQ5_1Into(row []byte, cols int, out []float32) {
	for b := 0; b < cols/32; b++ {
		base := b * 24
		if base+24 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		m := F16ToF32(binaryLE16(row[base+2:]))
		qh := uint32(row[base+4]) | uint32(row[base+5])<<8 | uint32(row[base+6])<<16 | uint32(row[base+7])<<24
		for j := 0; j < 16; j++ {
			lo, hi := q5Bits(row[base+8+j], qh, j)
			out[b*32+j] = d*float32(lo) + m
			out[b*32+16+j] = d*float32(hi) + m
		}
	}
}

func DequantRowQ5_1(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ5_1Into(row, cols, out)
	return out
}

func DotQ5_1F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/32; b++ {
		base := b * 24
		if base+24 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		m := F16ToF32(binaryLE16(row[base+2:]))
		qh := uint32(row[base+4]) | uint32(row[base+5])<<8 | uint32(row[base+6])<<16 | uint32(row[base+7])<<24
		xBlock := x[b*32 : b*32+32]
		_ = xBlock[31]
		var qdot, xsum float32
		for j := 0; j < 16; j++ {
			lo, hi := q5Bits(row[base+8+j], qh, j)
			qdot += float32(lo)*xBlock[j] + float32(hi)*xBlock[16+j]
			xsum += xBlock[j] + xBlock[16+j]
		}
		sum += d*qdot + m*xsum
	}
	return sum
}

// --- Q8_1: val = d*q; the second f16 (d*sum(q)) is a dot-product aid this
// scalar path doesn't need ---

func DequantRowQ8_1Into(row []byte, cols int, out []float32) {
	for b := 0; b < cols/32; b++ {
		base := b * 36
		if base+36 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		for i := 0; i < 32; i++ {
			out[b*32+i] = d * float32(int8(row[base+4+i]))
		}
	}
}

func DequantRowQ8_1(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ8_1Into(row, cols, out)
	return out
}

func DotQ8_1F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/32; b++ {
		base := b * 36
		if base+36 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		xBlock := x[b*32 : b*32+32]
		_ = xBlock[31]
		var acc float32
		for i := 0; i < 32; i++ {
			acc += float32(int8(row[base+4+i])) * xBlock[i]
		}
		sum += d * acc
	}
	return sum
}

// --- Q2_K: 16 sub-blocks of 16 elems; val = d*(sc&0xF)*q - dmin*(sc>>4),
// q a 2-bit quant. Layout: scales[16] | qs[64] | d f16 | dmin f16 ---

func DequantRowQ2KInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 84
		if base+84 > len(row) {
			break
		}
		scales := row[base : base+16]
		qs := row[base+16 : base+80]
		d := F16ToF32(binaryLE16(row[base+80:]))
		dmin := F16ToF32(binaryLE16(row[base+82:]))
		y := b * 256
		is := 0
		// Two 128-element halves, each drawing 2-bit quants from 32 bytes at
		// shifts 0/2/4/6, two 16-element sub-blocks per shift.
		for n := 0; n < 256; n += 128 {
			q := qs[(n/128)*32:]
			for shift := 0; shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					sc := scales[is]
					is++
					dl := d * float32(sc&0x0f)
					ml := dmin * float32(sc>>4)
					for l := 0; l < 16; l++ {
						out[y] = dl*float32((q[half*16+l]>>shift)&3) - ml
						y++
					}
				}
			}
		}
	}
}

func DequantRowQ2K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ2KInto(row, cols, out)
	return out
}

func DotQ2KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 84
		if base+84 > len(row) {
			break
		}
		scales := row[base : base+16]
		qs := row[base+16 : base+80]
		d := F16ToF32(binaryLE16(row[base+80:]))
		dmin := F16ToF32(binaryLE16(row[base+82:]))
		xBlock := x[b*256 : b*256+256]
		_ = xBlock[255]
		y := 0
		is := 0
		for n := 0; n < 256; n += 128 {
			q := qs[(n/128)*32:]
			for shift := 0; shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					sc := scales[is]
					is++
					dl := d * float32(sc&0x0f)
					ml := dmin * float32(sc>>4)
					var qdot, xsum float32
					for l := 0; l < 16; l++ {
						xv := xBlock[y+l]
						qdot += float32((q[half*16+l]>>shift)&3) * xv
						xsum += xv
					}
					sum += dl*qdot - ml*xsum
					y += 16
				}
			}
		}
	}
	return sum
}

// --- Q3_K: 16 sub-blocks of 16 elems; val = d*(sc-32)*(q - hi*4) where q is
// 2 low bits and a CLEAR hmask bit subtracts 4. Layout: hmask[32] | qs[64] |
// scales[12 packed 6-bit] | d f16 ---

// q3KScales unpacks the 12 packed bytes into 16 6-bit scale values
// (biased by 32), following ggml's kmask shuffle.
func q3KScales(packed []byte) [16]int8 {
	var aux [4]uint32
	for i := 0; i < 3; i++ {
		aux[i] = uint32(packed[i*4]) | uint32(packed[i*4+1])<<8 | uint32(packed[i*4+2])<<16 | uint32(packed[i*4+3])<<24
	}
	const kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
	var out [16]int8
	for i := 0; i < 4; i++ {
		out[i*4] = int8(aux[i])
		out[i*4+1] = int8(aux[i] >> 8)
		out[i*4+2] = int8(aux[i] >> 16)
		out[i*4+3] = int8(aux[i] >> 24)
	}
	return out
}

func DequantRowQ3KInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 110
		if base+110 > len(row) {
			break
		}
		hmask := row[base : base+32]
		qs := row[base+32 : base+96]
		scales := q3KScales(row[base+96 : base+108])
		d := F16ToF32(binaryLE16(row[base+108:]))
		y := b * 256
		is := 0
		m := byte(1)
		for n := 0; n < 256; n += 128 {
			q := qs[(n/128)*32:]
			for shift := 0; shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					dl := d * float32(int(scales[is])-32)
					is++
					for l := 0; l < 16; l++ {
						v := int((q[half*16+l] >> shift) & 3)
						if hmask[half*16+l]&m == 0 {
							v -= 4
						}
						out[y] = dl * float32(v)
						y++
					}
				}
				m <<= 1
			}
		}
	}
}

func DequantRowQ3K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ3KInto(row, cols, out)
	return out
}

func DotQ3KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 110
		if base+110 > len(row) {
			break
		}
		hmask := row[base : base+32]
		qs := row[base+32 : base+96]
		scales := q3KScales(row[base+96 : base+108])
		d := F16ToF32(binaryLE16(row[base+108:]))
		xBlock := x[b*256 : b*256+256]
		_ = xBlock[255]
		y := 0
		is := 0
		m := byte(1)
		for n := 0; n < 256; n += 128 {
			q := qs[(n/128)*32:]
			for shift := 0; shift < 8; shift += 2 {
				for half := 0; half < 2; half++ {
					dl := d * float32(int(scales[is])-32)
					is++
					var acc float32
					for l := 0; l < 16; l++ {
						v := int((q[half*16+l] >> shift) & 3)
						if hmask[half*16+l]&m == 0 {
							v -= 4
						}
						acc += float32(v) * xBlock[y+l]
					}
					sum += dl * acc
					y += 16
				}
				m <<= 1
			}
		}
	}
	return sum
}

// --- Matvec wrappers (parallel over rows through the shared worker pool) ---

func matvecScalarRows(rowBytes int, dot func(row []byte, x []float32, cols int) float32) func(data []byte, x []float32, rows, cols int, out *[]float32) {
	return func(data []byte, x []float32, rows, cols int, out *[]float32) {
		ensureLenNoClear(out, rows)
		parallelRows(rows, func(start, end int) {
			for r := start; r < end; r++ {
				off := r * rowBytes
				(*out)[r] = dot(data[off:min(off+rowBytes, len(data))], x, cols)
			}
		})
	}
}

func MatvecQ4_1Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/32)*20, DotQ4_1F32)(data, x, rows, cols, out)
}

func MatvecQ5_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/32)*22, DotQ5_0F32)(data, x, rows, cols, out)
}

func MatvecQ5_1Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/32)*24, DotQ5_1F32)(data, x, rows, cols, out)
}

func MatvecQ8_1Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/32)*36, DotQ8_1F32)(data, x, rows, cols, out)
}

func MatvecQ2KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/256)*84, DotQ2KF32)(data, x, rows, cols, out)
}

func MatvecQ3KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/256)*110, DotQ3KF32)(data, x, rows, cols, out)
}
