package gopherllm

import "math"

var mxfp4LUT = [...]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}

func MatvecMXFP4Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 32) * 17
	ensureLenNoClear(out, rows)
	// MXFP4 is symmetric (no offset term), so the int8 path needs only the
	// per-256-block activation scales, no xsums (see mxfp4DotQ8KRow).
	if useQ8Activations.Load() && cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		q8, xsc, lease := acquireQ8(x, cols)
		parallelRows(rows, func(start, end int) {
			dotMXFP4RowsQ8(data, q8, xsc, cols, rowBytes, start, end, *out)
		})
		releaseQ8(q8, xsc, lease)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotMXFP4F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func DotMXFP4F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 32
	for b := 0; b < blocks; b++ {
		base := b * 17
		if base+17 > len(row) {
			break
		}
		scale := float32(math.Pow(2, float64(int(row[base+16])-127)))
		rBlock := row[base : base+16]
		xBlock := x[b*32 : b*32+32]
		_ = rBlock[15]
		_ = xBlock[31]
		for i := 0; i < 16; i++ {
			v := rBlock[i]
			sum += mxfp4LUT[v&0x0f] * scale * xBlock[i*2]
			sum += mxfp4LUT[v>>4] * scale * xBlock[i*2+1]
		}
	}
	return sum
}

func DequantRowMXFP4(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowMXFP4Into(row, cols, out)
	return out
}

// DequantRowMXFP4Into is the allocation-free MXFP4 row decoder used by
// RowInto and prompt batching.
func DequantRowMXFP4Into(row []byte, cols int, out []float32) {
	for b := range cols / 32 {
		base := b * 17
		if base+17 > len(row) {
			break
		}
		scale := float32(math.Pow(2, float64(int(row[base+16])-127)))
		rBlock := row[base : base+16]
		outBlock := out[b*32 : b*32+32]
		_ = rBlock[15]
		_ = outBlock[31]
		for i := range 16 {
			v := rBlock[i]
			outBlock[i*2] = scale * mxfp4LUT[v&0x0f]
			outBlock[i*2+1] = scale * mxfp4LUT[v>>4]
		}
	}
}
