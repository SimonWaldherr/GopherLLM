package gopherllm

func MatvecQ8_0(data []byte, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	MatvecQ8_0Into(data, x, rows, cols, &out)
	return out
}

func MatvecQ8_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 32) * 34
	ensureLenNoClear(out, rows)
	// (cols/32)*34 == (cols/256)*272 exactly when cols%256==0, so rowBytes
	// already matches the int8-activation path's 8-block-per-256 grouping.
	if useQ8Activations.Load() && cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		q8, xsc, lease := acquireQ8(x, cols)
		parallelRows(rows, func(start, end int) {
			dotQ8_0RowsQ8(data, q8, xsc, cols, rowBytes, start, end, *out)
		})
		releaseQ8(q8, xsc, lease)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ8_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func DotQ8_0F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 32
	for b := 0; b < blocks; b++ {
		base := b * 34
		if base+34 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		rBlock := row[base+2 : base+34]
		xBlock := x[b*32 : b*32+32]
		_ = rBlock[31]
		_ = xBlock[31]

		var s0, s1, s2, s3 float32
		for i := 0; i < 32; i += 8 {
			s0 += float32(int8(rBlock[i])) * xBlock[i]
			s1 += float32(int8(rBlock[i+1])) * xBlock[i+1]
			s2 += float32(int8(rBlock[i+2])) * xBlock[i+2]
			s3 += float32(int8(rBlock[i+3])) * xBlock[i+3]

			s0 += float32(int8(rBlock[i+4])) * xBlock[i+4]
			s1 += float32(int8(rBlock[i+5])) * xBlock[i+5]
			s2 += float32(int8(rBlock[i+6])) * xBlock[i+6]
			s3 += float32(int8(rBlock[i+7])) * xBlock[i+7]
		}
		sum += scale * ((s0 + s1) + (s2 + s3))
	}
	return sum
}

func DequantRowQ8_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ8_0Into(row, cols, out)
	return out
}

// DequantRowQ8_0Into is the allocation-free Q8_0 decoder used by batched
// prefill, where one decoded weight row is shared across many prompt tokens.
func DequantRowQ8_0Into(row []byte, cols int, out []float32) {
	for b := range cols / 32 {
		base := b * 34
		if base+34 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		for i := range 32 {
			out[b*32+i] = scale * float32(int8(row[base+2+i]))
		}
	}
}
