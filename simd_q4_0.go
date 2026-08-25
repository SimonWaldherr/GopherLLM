package gopherllm

func MatvecQ4_0(data []byte, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	MatvecQ4_0Into(data, x, rows, cols, &out)
	return out
}

func MatvecQ4_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 32) * 18
	ensureLenNoClear(out, rows)
	// The per-32-element activation sums carry Q4_0's -8 offset term exactly,
	// like Q4_K's dmin term (see q4_0DotQ8KRow).
	if useQ8Activations.Load() && cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, cols, scratch)
		q8, xsc, lease := acquireQ8(x, cols)
		parallelRows(rows, func(start, end int) {
			dotQ4_0RowsQ8(data, q8, xsc, xs, cols, rowBytes, start, end, *out)
		})
		releaseQ8(q8, xsc, lease)
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ4_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func DotQ4_0F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 32
	for b := 0; b < blocks; b++ {
		base := b * 18
		if base+18 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		rBlock := row[base+2 : base+18]
		xBlock := x[b*32 : b*32+32]
		_ = rBlock[15]
		_ = xBlock[31]

		var s0, s1, s2, s3 float32
		for i := 0; i < 16; i += 4 {
			p0 := rBlock[i]
			p1 := rBlock[i+1]
			p2 := rBlock[i+2]
			p3 := rBlock[i+3]

			lo0 := float32(int(p0&0x0f) - 8)
			hi0 := float32(int((p0>>4)&0x0f) - 8)
			lo1 := float32(int(p1&0x0f) - 8)
			hi1 := float32(int((p1>>4)&0x0f) - 8)
			lo2 := float32(int(p2&0x0f) - 8)
			hi2 := float32(int((p2>>4)&0x0f) - 8)
			lo3 := float32(int(p3&0x0f) - 8)
			hi3 := float32(int((p3>>4)&0x0f) - 8)

			s0 += lo0*xBlock[i] + hi0*xBlock[16+i]
			s1 += lo1*xBlock[i+1] + hi1*xBlock[17+i]
			s2 += lo2*xBlock[i+2] + hi2*xBlock[18+i]
			s3 += lo3*xBlock[i+3] + hi3*xBlock[19+i]
		}
		sum += scale * ((s0 + s1) + (s2 + s3))
	}
	return sum
}

func DequantRowQ4_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ4_0Into(row, cols, out)
	return out
}

// DequantRowQ4_0Into is the allocation-free Q4_0 decoder used by batched
// prefill. Stable Code GGUFs commonly use this legacy format.
func DequantRowQ4_0Into(row []byte, cols int, out []float32) {
	for b := range cols / 32 {
		base := b * 18
		if base+18 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		for i := range 16 {
			packed := row[base+2+i]
			out[b*32+i] = scale * float32(int(packed&0x0f)-8)
			out[b*32+16+i] = scale * float32(int((packed>>4)&0x0f)-8)
		}
	}
}
