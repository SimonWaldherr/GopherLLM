package gopherllm

func MatvecQ6KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 210
	ensureLenNoClear(out, rows)
	if cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ6KXSums16(x, cols, scratch)
		ScaleF32(xs, 32)
		if useQ8Activations.Load() {
			q8, xsc, lease := acquireQ8(x, cols)
			parallelRows(rows, func(start, end int) {
				dotQ6KRowsQ8(data, q8, xsc, xs, cols, rowBytes, start, end, *out)
			})
			releaseQ8(q8, xsc, lease)
		} else {
			parallelRows(rows, func(start, end int) {
				dotQ6KRowsWithXSums(data, x, xs, cols, rowBytes, start, end, *out)
			})
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ6KF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func dotQ6KRows(data []byte, x []float32, cols, rowBytes, start, end int, out []float32) {
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = DotQ6KF32(data[off:off+rowBytes], x, cols)
	}
}

func dotQ6KRowsWithXSums(data []byte, x, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	if !hasQuantSIMD {
		dotQ6KRows(data, x, cols, rowBytes, start, end, out)
		return
	}
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = dotQ6KF32SIMDWithXSums(data[off:off+rowBytes], x, xsums, cols)
	}
}

// fillQ6KXSums16 computes per-16-element sums of x, used to fold the
// constant -32 offset of Q6_K quants out of the inner dot product.
func fillQ6KXSums16(x []float32, cols int, scratch *[]float32) []float32 {
	groups := cols / 16
	ensureLenNoClear(scratch, groups)
	out := *scratch
	if hasQuantSIMD && groups > 0 && len(x) >= groups*16 {
		sumF32Groups16(&x[0], &out[0], groups)
		return out
	}
	for g := range groups {
		base := g * 16
		if base+16 > len(x) {
			out[g] = 0
			continue
		}
		xBlock := x[base : base+16]
		_ = xBlock[15]
		var s0, s1, s2, s3 float32
		for i := 0; i < 16; i += 4 {
			s0 += xBlock[i]
			s1 += xBlock[i+1]
			s2 += xBlock[i+2]
			s3 += xBlock[i+3]
		}
		out[g] = (s0 + s1) + (s2 + s3)
	}
	return out
}

// dotQ6KF32SIMDWithXSums computes a Q6_K row dot product using the SIMD block
// kernel. xsums must hold 32x per-16-element sums of x, folding the Q6_K -32
// offset out of each row.
func dotQ6KF32SIMDWithXSums(row []byte, x, xsums []float32, cols int) float32 {
	var qdots [16]float32
	var sum float32
	blocks := cols / 256
	blocks = min(blocks, len(row)/210)
	blocks = min(blocks, len(x)/256)
	blocks = min(blocks, len(xsums)/16)
	if blocks <= 0 {
		return 0
	}
	_ = row[blocks*210-1]
	_ = x[blocks*256-1]
	_ = xsums[blocks*16-1]
	for b := 0; b < blocks; b++ {
		base := b * 210
		block := row[base : base+210]
		_ = block[209]
		d := F16ToF32(uint16(block[208]) | uint16(block[209])<<8)
		q6kQDots16(&block[0], &block[128], &x[b*256], &qdots[0])
		xs := xsums[b*16 : b*16+16]
		_ = xs[15]
		blockSum :=
			int8ToFloat32LUT[block[192]]*(qdots[0]-xs[0]) +
				int8ToFloat32LUT[block[193]]*(qdots[1]-xs[1]) +
				int8ToFloat32LUT[block[194]]*(qdots[2]-xs[2]) +
				int8ToFloat32LUT[block[195]]*(qdots[3]-xs[3]) +
				int8ToFloat32LUT[block[196]]*(qdots[4]-xs[4]) +
				int8ToFloat32LUT[block[197]]*(qdots[5]-xs[5]) +
				int8ToFloat32LUT[block[198]]*(qdots[6]-xs[6]) +
				int8ToFloat32LUT[block[199]]*(qdots[7]-xs[7]) +
				int8ToFloat32LUT[block[200]]*(qdots[8]-xs[8]) +
				int8ToFloat32LUT[block[201]]*(qdots[9]-xs[9]) +
				int8ToFloat32LUT[block[202]]*(qdots[10]-xs[10]) +
				int8ToFloat32LUT[block[203]]*(qdots[11]-xs[11]) +
				int8ToFloat32LUT[block[204]]*(qdots[12]-xs[12]) +
				int8ToFloat32LUT[block[205]]*(qdots[13]-xs[13]) +
				int8ToFloat32LUT[block[206]]*(qdots[14]-xs[14]) +
				int8ToFloat32LUT[block[207]]*(qdots[15]-xs[15])
		sum += d * blockSum
	}
	return sum
}

func DotQ6KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 210
		if base+210 > len(row) {
			break
		}
		block := row[base : base+210]
		ql := block[0:128]
		qh := block[128:192]
		sc := block[192:208]
		d := F16ToF32(binaryLE16(block[208:]))
		xBlock := x[b*256 : b*256+256]

		_ = ql[127]
		_ = qh[63]
		_ = sc[15]
		_ = xBlock[255]

		for step := 0; step < 2; step++ {
			n := step * 128
			qlSub := ql[step*64 : step*64+64]
			qhSub := qh[step*32 : step*32+32]
			scSub := sc[step*8 : step*8+8]
			xSub := xBlock[n : n+128]

			_ = qlSub[63]
			_ = qhSub[31]
			_ = scSub[7]
			_ = xSub[127]

			var s0, s1, s2, s3 float32

			// Precompute scales for l < 16
			d_sc0 := d * float32(int8(scSub[0]))
			d_sc2 := d * float32(int8(scSub[2]))
			d_sc4 := d * float32(int8(scSub[4]))
			d_sc6 := d * float32(int8(scSub[6]))

			for l := 0; l < 16; l += 4 {
				ql0 := qlSub[l]
				ql32_0 := qlSub[l+32]
				qh0 := qhSub[l]
				q1_0 := float32(int((ql0&0x0f)|((qh0&0x03)<<4)) - 32)
				q2_0 := float32(int((ql32_0&0x0f)|(((qh0>>2)&0x03)<<4)) - 32)
				q3_0 := float32(int((ql0>>4)|(((qh0>>4)&0x03)<<4)) - 32)
				q4_0 := float32(int((ql32_0>>4)|(((qh0>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_0 * xSub[l]
				s1 += d_sc2 * q2_0 * xSub[32+l]
				s2 += d_sc4 * q3_0 * xSub[64+l]
				s3 += d_sc6 * q4_0 * xSub[96+l]

				ql1 := qlSub[l+1]
				ql32_1 := qlSub[l+33]
				qh1 := qhSub[l+1]
				q1_1 := float32(int((ql1&0x0f)|((qh1&0x03)<<4)) - 32)
				q2_1 := float32(int((ql32_1&0x0f)|(((qh1>>2)&0x03)<<4)) - 32)
				q3_1 := float32(int((ql1>>4)|(((qh1>>4)&0x03)<<4)) - 32)
				q4_1 := float32(int((ql32_1>>4)|(((qh1>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_1 * xSub[l+1]
				s1 += d_sc2 * q2_1 * xSub[33+l]
				s2 += d_sc4 * q3_1 * xSub[65+l]
				s3 += d_sc6 * q4_1 * xSub[97+l]

				ql2 := qlSub[l+2]
				ql32_2 := qlSub[l+34]
				qh2 := qhSub[l+2]
				q1_2 := float32(int((ql2&0x0f)|((qh2&0x03)<<4)) - 32)
				q2_2 := float32(int((ql32_2&0x0f)|(((qh2>>2)&0x03)<<4)) - 32)
				q3_2 := float32(int((ql2>>4)|(((qh2>>4)&0x03)<<4)) - 32)
				q4_2 := float32(int((ql32_2>>4)|(((qh2>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_2 * xSub[l+2]
				s1 += d_sc2 * q2_2 * xSub[34+l]
				s2 += d_sc4 * q3_2 * xSub[66+l]
				s3 += d_sc6 * q4_2 * xSub[98+l]

				ql3 := qlSub[l+3]
				ql32_3 := qlSub[l+35]
				qh3 := qhSub[l+3]
				q1_3 := float32(int((ql3&0x0f)|((qh3&0x03)<<4)) - 32)
				q2_3 := float32(int((ql32_3&0x0f)|(((qh3>>2)&0x03)<<4)) - 32)
				q3_3 := float32(int((ql3>>4)|(((qh3>>4)&0x03)<<4)) - 32)
				q4_3 := float32(int((ql32_3>>4)|(((qh3>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_3 * xSub[l+3]
				s1 += d_sc2 * q2_3 * xSub[35+l]
				s2 += d_sc4 * q3_3 * xSub[67+l]
				s3 += d_sc6 * q4_3 * xSub[99+l]
			}

			// Precompute scales for l >= 16
			d_sc1 := d * float32(int8(scSub[1]))
			d_sc3 := d * float32(int8(scSub[3]))
			d_sc5 := d * float32(int8(scSub[5]))
			d_sc7 := d * float32(int8(scSub[7]))

			for l := 16; l < 32; l += 4 {
				ql0 := qlSub[l]
				ql32_0 := qlSub[l+32]
				qh0 := qhSub[l]
				q1_0 := float32(int((ql0&0x0f)|((qh0&0x03)<<4)) - 32)
				q2_0 := float32(int((ql32_0&0x0f)|(((qh0>>2)&0x03)<<4)) - 32)
				q3_0 := float32(int((ql0>>4)|(((qh0>>4)&0x03)<<4)) - 32)
				q4_0 := float32(int((ql32_0>>4)|(((qh0>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_0 * xSub[l]
				s1 += d_sc3 * q2_0 * xSub[32+l]
				s2 += d_sc5 * q3_0 * xSub[64+l]
				s3 += d_sc7 * q4_0 * xSub[96+l]

				ql1 := qlSub[l+1]
				ql32_1 := qlSub[l+33]
				qh1 := qhSub[l+1]
				q1_1 := float32(int((ql1&0x0f)|((qh1&0x03)<<4)) - 32)
				q2_1 := float32(int((ql32_1&0x0f)|(((qh1>>2)&0x03)<<4)) - 32)
				q3_1 := float32(int((ql1>>4)|(((qh1>>4)&0x03)<<4)) - 32)
				q4_1 := float32(int((ql32_1>>4)|(((qh1>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_1 * xSub[l+1]
				s1 += d_sc3 * q2_1 * xSub[33+l]
				s2 += d_sc5 * q3_1 * xSub[65+l]
				s3 += d_sc7 * q4_1 * xSub[97+l]

				ql2 := qlSub[l+2]
				ql32_2 := qlSub[l+34]
				qh2 := qhSub[l+2]
				q1_2 := float32(int((ql2&0x0f)|((qh2&0x03)<<4)) - 32)
				q2_2 := float32(int((ql32_2&0x0f)|(((qh2>>2)&0x03)<<4)) - 32)
				q3_2 := float32(int((ql2>>4)|(((qh2>>4)&0x03)<<4)) - 32)
				q4_2 := float32(int((ql32_2>>4)|(((qh2>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_2 * xSub[l+2]
				s1 += d_sc3 * q2_2 * xSub[34+l]
				s2 += d_sc5 * q3_2 * xSub[66+l]
				s3 += d_sc7 * q4_2 * xSub[98+l]

				ql3 := qlSub[l+3]
				ql32_3 := qlSub[l+35]
				qh3 := qhSub[l+3]
				q1_3 := float32(int((ql3&0x0f)|((qh3&0x03)<<4)) - 32)
				q2_3 := float32(int((ql32_3&0x0f)|(((qh3>>2)&0x03)<<4)) - 32)
				q3_3 := float32(int((ql3>>4)|(((qh3>>4)&0x03)<<4)) - 32)
				q4_3 := float32(int((ql32_3>>4)|(((qh3>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_3 * xSub[l+3]
				s1 += d_sc3 * q2_3 * xSub[35+l]
				s2 += d_sc5 * q3_3 * xSub[67+l]
				s3 += d_sc7 * q4_3 * xSub[99+l]
			}

			sum += (s0 + s1) + (s2 + s3)
		}
	}
	return sum
}

func DequantRowQ6K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ6KInto(row, cols, out)
	return out
}

func DequantRowQ6KInto(row []byte, cols int, out []float32) {
	for b := range cols / 256 {
		base := b * 210
		if base+210 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base+208:]))
		ql := row[base:]
		qh := row[base+128 : base+192]
		sc := row[base+192 : base+208]
		yoff := b * 256
		for n := 0; n < 256; n += 128 {
			for l := range 32 {
				is := l / 16
				q1 := float32(int((ql[l]&0x0f)|((qh[l]&0x03)<<4)) - 32)
				q2 := float32(int((ql[l+32]&0x0f)|(((qh[l]>>2)&0x03)<<4)) - 32)
				q3 := float32(int((ql[l]>>4)|(((qh[l]>>4)&0x03)<<4)) - 32)
				q4 := float32(int((ql[l+32]>>4)|(((qh[l]>>6)&0x03)<<4)) - 32)
				out[yoff+n+l] = d * float32(int8(sc[is])) * q1
				out[yoff+n+32+l] = d * float32(int8(sc[is+2])) * q2
				out[yoff+n+64+l] = d * float32(int8(sc[is+4])) * q3
				out[yoff+n+96+l] = d * float32(int8(sc[is+6])) * q4
			}
			ql = ql[64:]
			qh = qh[32:]
			sc = sc[8:]
		}
	}
}
