package gopherllm

import (
	"encoding/binary"

	"github.com/SimonWaldherr/GopherLLM/internal/iqcodebook"
)

// Scalar kernels for the sub-2-bit-to-3-bit "importance quantization"
// formats: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ3_XXS. These follow the same
// codebook-lookup + sign-table shape as IQ2_S/IQ3_S in quant_iq.go, just with
// their own grids (see internal/iqcodebook) and, for the 1-bit formats, a
// signed ternary grid plus a shared per-group delta instead of a sign table.
// Block layouts and bit-extraction match llama.cpp's ggml-quants.c
// dequantize_row_iq{1,2,3}* reference implementations verbatim.

const iq1Delta = 0.125

// iqGridByteU treats a codebook entry as 8 packed unsigned magnitude bytes
// (IQ2_XXS/IQ2_XS reuse IQ2_S's convention; the sign comes from a separate
// bitmask, not the byte's own sign).
func iqGridByteU(grid uint64, j int) float32 {
	return float32(byte(grid >> (8 * j)))
}

// iq3xxsGridByteU is the same convention as iqGridByteU but for the
// 32-bit (4-byte) IQ3_XXS grid entries.
func iq3xxsGridByteU(grid uint32, j int) float32 {
	return float32(byte(grid >> (8 * j)))
}

// iq1GridByteS treats a codebook entry as 8 packed signed int8 values in
// {-1, 0, 1}; IQ1_S/IQ1_M encode the sign directly in the grid.
func iq1GridByteS(grid uint64, j int) float32 {
	return float32(int8(byte(grid >> (8 * j))))
}

// ====================== IQ2_XXS: 66 B / 256 elems ======================

func DequantRowIQ2XXSInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 66
		if base+66 > len(row) {
			break
		}
		block := row[base : base+66]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		dst := out[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			chunk := qs[8*ib32 : 8*ib32+8]
			aux1 := binary.LittleEndian.Uint32(chunk[4:8])
			db := d * (0.5 + float32(aux1>>28)) * 0.25
			for l := 0; l < 4; l++ {
				grid := iqcodebook.IQ2XXSGrid[chunk[l]]
				signs := iqcodebook.KSignsIQ2XS[(aux1>>(7*uint(l)))&127]
				for j := 0; j < 8; j++ {
					dst[at+j] = db * iqGridByteU(grid, j) * iqSign(signs, j)
				}
				at += 8
			}
		}
	}
}

func DequantRowIQ2XXS(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowIQ2XXSInto(row, cols, out)
	return out
}

func DotIQ2XXSF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 66
		if base+66 > len(row) {
			break
		}
		block := row[base : base+66]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		xBlock := x[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			chunk := qs[8*ib32 : 8*ib32+8]
			aux1 := binary.LittleEndian.Uint32(chunk[4:8])
			db := d * (0.5 + float32(aux1>>28)) * 0.25
			for l := 0; l < 4; l++ {
				grid := iqcodebook.IQ2XXSGrid[chunk[l]]
				signs := iqcodebook.KSignsIQ2XS[(aux1>>(7*uint(l)))&127]
				for j := 0; j < 8; j++ {
					sum += db * iqGridByteU(grid, j) * iqSign(signs, j) * xBlock[at+j]
				}
				at += 8
			}
		}
	}
	return sum
}

func MatvecIQ2XXSInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/256)*66, DotIQ2XXSF32)(data, x, rows, cols, out)
}

// ====================== IQ2_XS: 74 B / 256 elems ======================

func DequantRowIQ2XSInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 74
		if base+74 > len(row) {
			break
		}
		block := row[base : base+74]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		scales := block[2+64 : 2+72]
		dst := out[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			db0 := d * (0.5 + float32(scales[ib32]&0x0f)) * 0.25
			db1 := d * (0.5 + float32(scales[ib32]>>4)) * 0.25
			for l := 0; l < 4; l++ {
				q16 := binary.LittleEndian.Uint16(qs[2*(4*ib32+l):])
				grid := iqcodebook.IQ2XSGrid[q16&511]
				signs := iqcodebook.KSignsIQ2XS[q16>>9]
				db := db0
				if l >= 2 {
					db = db1
				}
				for j := 0; j < 8; j++ {
					dst[at+j] = db * iqGridByteU(grid, j) * iqSign(signs, j)
				}
				at += 8
			}
		}
	}
}

func DequantRowIQ2XS(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowIQ2XSInto(row, cols, out)
	return out
}

func DotIQ2XSF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 74
		if base+74 > len(row) {
			break
		}
		block := row[base : base+74]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		scales := block[2+64 : 2+72]
		xBlock := x[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			db0 := d * (0.5 + float32(scales[ib32]&0x0f)) * 0.25
			db1 := d * (0.5 + float32(scales[ib32]>>4)) * 0.25
			for l := 0; l < 4; l++ {
				q16 := binary.LittleEndian.Uint16(qs[2*(4*ib32+l):])
				grid := iqcodebook.IQ2XSGrid[q16&511]
				signs := iqcodebook.KSignsIQ2XS[q16>>9]
				db := db0
				if l >= 2 {
					db = db1
				}
				for j := 0; j < 8; j++ {
					sum += db * iqGridByteU(grid, j) * iqSign(signs, j) * xBlock[at+j]
				}
				at += 8
			}
		}
	}
	return sum
}

func MatvecIQ2XSInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/256)*74, DotIQ2XSF32)(data, x, rows, cols, out)
}

// ====================== IQ3_XXS: 98 B / 256 elems ======================

func DequantRowIQ3XXSInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 98
		if base+98 > len(row) {
			break
		}
		block := row[base : base+98]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		scalesAndSigns := block[2+64 : 2+96]
		dst := out[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			aux32 := binary.LittleEndian.Uint32(scalesAndSigns[4*ib32 : 4*ib32+4])
			db := d * (0.5 + float32(aux32>>28)) * 0.5
			for l := 0; l < 4; l++ {
				signs := iqcodebook.KSignsIQ2XS[(aux32>>(7*uint(l)))&127]
				grid1 := iqcodebook.IQ3XXSGrid[qs[8*ib32+2*l+0]]
				grid2 := iqcodebook.IQ3XXSGrid[qs[8*ib32+2*l+1]]
				for j := 0; j < 4; j++ {
					dst[at+j] = db * iq3xxsGridByteU(grid1, j) * iqSign(signs, j)
					dst[at+4+j] = db * iq3xxsGridByteU(grid2, j) * iqSign(signs, j+4)
				}
				at += 8
			}
		}
	}
}

func DequantRowIQ3XXS(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowIQ3XXSInto(row, cols, out)
	return out
}

func DotIQ3XXSF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 98
		if base+98 > len(row) {
			break
		}
		block := row[base : base+98]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+64]
		scalesAndSigns := block[2+64 : 2+96]
		xBlock := x[b*256 : b*256+256]
		at := 0
		for ib32 := 0; ib32 < 8; ib32++ {
			aux32 := binary.LittleEndian.Uint32(scalesAndSigns[4*ib32 : 4*ib32+4])
			db := d * (0.5 + float32(aux32>>28)) * 0.5
			for l := 0; l < 4; l++ {
				signs := iqcodebook.KSignsIQ2XS[(aux32>>(7*uint(l)))&127]
				grid1 := iqcodebook.IQ3XXSGrid[qs[8*ib32+2*l+0]]
				grid2 := iqcodebook.IQ3XXSGrid[qs[8*ib32+2*l+1]]
				for j := 0; j < 4; j++ {
					sum += db * iq3xxsGridByteU(grid1, j) * iqSign(signs, j) * xBlock[at+j]
					sum += db * iq3xxsGridByteU(grid2, j) * iqSign(signs, j+4) * xBlock[at+4+j]
				}
				at += 8
			}
		}
	}
	return sum
}

func MatvecIQ3XXSInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/256)*98, DotIQ3XXSF32)(data, x, rows, cols, out)
}

// ====================== IQ1_S: 50 B / 256 elems ======================

func DequantRowIQ1SInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 50
		if base+50 > len(row) {
			break
		}
		block := row[base : base+50]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+32]
		qh := block[2+32 : 2+48]
		dst := out[b*256 : b*256+256]
		at := 0
		for ib := 0; ib < 8; ib++ {
			qhv := binary.LittleEndian.Uint16(qh[2*ib:])
			dl := d * (2*float32((qhv>>12)&7) + 1)
			delta := float32(iq1Delta)
			if qhv&0x8000 != 0 {
				delta = -iq1Delta
			}
			for l := 0; l < 4; l++ {
				idx := uint32(qs[4*ib+l]) | ((uint32(qhv>>(3*uint(l))) & 7) << 8)
				grid := iqcodebook.IQ1SGrid[idx]
				for j := 0; j < 8; j++ {
					dst[at+j] = dl * (iq1GridByteS(grid, j) + delta)
				}
				at += 8
			}
		}
	}
}

func DequantRowIQ1S(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowIQ1SInto(row, cols, out)
	return out
}

func DotIQ1SF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 50
		if base+50 > len(row) {
			break
		}
		block := row[base : base+50]
		d := F16ToF32(binaryLE16(block))
		qs := block[2 : 2+32]
		qh := block[2+32 : 2+48]
		xBlock := x[b*256 : b*256+256]
		at := 0
		for ib := 0; ib < 8; ib++ {
			qhv := binary.LittleEndian.Uint16(qh[2*ib:])
			dl := d * (2*float32((qhv>>12)&7) + 1)
			delta := float32(iq1Delta)
			if qhv&0x8000 != 0 {
				delta = -iq1Delta
			}
			for l := 0; l < 4; l++ {
				idx := uint32(qs[4*ib+l]) | ((uint32(qhv>>(3*uint(l))) & 7) << 8)
				grid := iqcodebook.IQ1SGrid[idx]
				for j := 0; j < 8; j++ {
					sum += dl * (iq1GridByteS(grid, j) + delta) * xBlock[at+j]
				}
				at += 8
			}
		}
	}
	return sum
}

func MatvecIQ1SInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/256)*50, DotIQ1SF32)(data, x, rows, cols, out)
}

// ====================== IQ1_M: 56 B / 256 elems ======================
//
// IQ1_M has no standalone f16 block scale: it is reassembled from the top 4
// bits of each of the 4 uint16 sub-block-scale words (see scaleU16 below),
// matching ggml's iq1m_scale_t union trick.

func iq1mScale(scalesBytes []byte) (d float32, sc [4]uint16) {
	sc[0] = binary.LittleEndian.Uint16(scalesBytes[0:])
	sc[1] = binary.LittleEndian.Uint16(scalesBytes[2:])
	sc[2] = binary.LittleEndian.Uint16(scalesBytes[4:])
	sc[3] = binary.LittleEndian.Uint16(scalesBytes[6:])
	scaleU16 := (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
	return F16ToF32(scaleU16), sc
}

func iq1mBlockDeltasAndIdx(qs, qh []byte, ib int, sc [4]uint16) (dl1, dl2 float32, idx [4]uint32, delta [4]float32) {
	shift := uint(6 * (ib % 2))
	dl1 = 2*float32((sc[ib/2]>>(shift+0))&7) + 1
	dl2 = 2*float32((sc[ib/2]>>(shift+3))&7) + 1
	qb := qs[4*ib : 4*ib+4]
	qhb0 := qh[2*ib]
	qhb1 := qh[2*ib+1]
	idx = [4]uint32{
		uint32(qb[0]) | ((uint32(qhb0) << 8) & 0x700),
		uint32(qb[1]) | ((uint32(qhb0) << 4) & 0x700),
		uint32(qb[2]) | ((uint32(qhb1) << 8) & 0x700),
		uint32(qb[3]) | ((uint32(qhb1) << 4) & 0x700),
	}
	delta = [4]float32{iq1Delta, iq1Delta, iq1Delta, iq1Delta}
	if qhb0&0x08 != 0 {
		delta[0] = -iq1Delta
	}
	if qhb0&0x80 != 0 {
		delta[1] = -iq1Delta
	}
	if qhb1&0x08 != 0 {
		delta[2] = -iq1Delta
	}
	if qhb1&0x80 != 0 {
		delta[3] = -iq1Delta
	}
	return
}

func DequantRowIQ1MInto(row []byte, cols int, out []float32) {
	for b := 0; b < cols/256; b++ {
		base := b * 56
		if base+56 > len(row) {
			break
		}
		block := row[base : base+56]
		qs := block[0:32]
		qh := block[32:48]
		d, sc := iq1mScale(block[48:56])
		dst := out[b*256 : b*256+256]
		at := 0
		for ib := 0; ib < 8; ib++ {
			dl1, dl2, idx, delta := iq1mBlockDeltasAndIdx(qs, qh, ib, sc)
			dl1 *= d
			dl2 *= d
			for l := 0; l < 2; l++ {
				grid := iqcodebook.IQ1SGrid[idx[l]]
				for j := 0; j < 8; j++ {
					dst[at+j] = dl1 * (iq1GridByteS(grid, j) + delta[l])
				}
				at += 8
			}
			for l := 2; l < 4; l++ {
				grid := iqcodebook.IQ1SGrid[idx[l]]
				for j := 0; j < 8; j++ {
					dst[at+j] = dl2 * (iq1GridByteS(grid, j) + delta[l])
				}
				at += 8
			}
		}
	}
}

func DequantRowIQ1M(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowIQ1MInto(row, cols, out)
	return out
}

func DotIQ1MF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	for b := 0; b < cols/256; b++ {
		base := b * 56
		if base+56 > len(row) {
			break
		}
		block := row[base : base+56]
		qs := block[0:32]
		qh := block[32:48]
		d, sc := iq1mScale(block[48:56])
		xBlock := x[b*256 : b*256+256]
		at := 0
		for ib := 0; ib < 8; ib++ {
			dl1, dl2, idx, delta := iq1mBlockDeltasAndIdx(qs, qh, ib, sc)
			dl1 *= d
			dl2 *= d
			for l := 0; l < 2; l++ {
				grid := iqcodebook.IQ1SGrid[idx[l]]
				for j := 0; j < 8; j++ {
					sum += dl1 * (iq1GridByteS(grid, j) + delta[l]) * xBlock[at+j]
				}
				at += 8
			}
			for l := 2; l < 4; l++ {
				grid := iqcodebook.IQ1SGrid[idx[l]]
				for j := 0; j < 8; j++ {
					sum += dl2 * (iq1GridByteS(grid, j) + delta[l]) * xBlock[at+j]
				}
				at += 8
			}
		}
	}
	return sum
}

func MatvecIQ1MInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	matvecScalarRows((cols/256)*56, DotIQ1MF32)(data, x, rows, cols, out)
}
