package gopherllm

func MatvecQ5KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 176
	ensureLenNoClear(out, rows)
	// Q5_K shares Q4_K's per-sub-block scale/min structure, so the same
	// per-32-element activation sums feed its dmin term.
	if useQ8Activations.Load() && cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, cols, scratch)
		q8, xsc, lease := acquireQ8(x, cols)
		parallelRows(rows, func(start, end int) {
			dotQ5KRowsQ8(data, q8, xsc, xs, cols, rowBytes, start, end, *out)
		})
		releaseQ8(q8, xsc, lease)
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ5KF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func DotQ5KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 176
		if base+176 > len(row) {
			break
		}
		block := row[base : base+176]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		qh := block[16:48]
		q := block[48:176]
		xBlock := x[b*256 : b*256+256]

		_ = qh[31]
		_ = q[127]
		_ = xBlock[255]
		_ = scales[11]

		for step := 0; step < 4; step++ {
			j := step * 64
			is := step * 2

			var sc1, m1, sc2, m2 byte
			if is < 4 {
				sc1, m1 = scales[is]&63, scales[is+4]&63
				sc2, m2 = scales[is+1]&63, scales[is+5]&63
			} else {
				sc1, m1 = (scales[is+4]&0x0f)|((scales[is-4]>>6)<<4), (scales[is+4]>>4)|((scales[is]>>6)<<4)
				sc2, m2 = (scales[is+5]&0x0f)|((scales[is-3]>>6)<<4), (scales[is+5]>>4)|((scales[is+1]>>6)<<4)
			}

			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)

			qSub := q[step*32 : step*32+32]
			xSub1 := xBlock[j : j+32]
			xSub2 := xBlock[j+32 : j+64]

			_ = qSub[31]
			_ = xSub1[31]
			_ = xSub2[31]

			var qd1_0, qd1_1, qd1_2, qd1_3 float32
			var qd2_0, qd2_1, qd2_2, qd2_3 float32
			var xs1_0, xs1_1, xs1_2, xs1_3 float32
			var xs2_0, xs2_1, xs2_2, xs2_3 float32

			switch step {
			case 0:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := (qh_0 & 1) << 4
					hi2_0 := (qh_0 & 2) << 3
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := (qh_1 & 1) << 4
					hi2_1 := (qh_1 & 2) << 3
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := (qh_2 & 1) << 4
					hi2_2 := (qh_2 & 2) << 3
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := (qh_3 & 1) << 4
					hi2_3 := (qh_3 & 2) << 3
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			case 1:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := (qh_0 & 4) << 2
					hi2_0 := (qh_0 & 8) << 1
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := (qh_1 & 4) << 2
					hi2_1 := (qh_1 & 8) << 1
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := (qh_2 & 4) << 2
					hi2_2 := (qh_2 & 8) << 1
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := (qh_3 & 4) << 2
					hi2_3 := (qh_3 & 8) << 1
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			case 2:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := qh_0 & 16
					hi2_0 := (qh_0 & 32) >> 1
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := qh_1 & 16
					hi2_1 := (qh_1 & 32) >> 1
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := qh_2 & 16
					hi2_2 := (qh_2 & 32) >> 1
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := qh_3 & 16
					hi2_3 := (qh_3 & 32) >> 1
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			case 3:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := (qh_0 & 64) >> 2
					hi2_0 := (qh_0 & 128) >> 3
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := (qh_1 & 64) >> 2
					hi2_1 := (qh_1 & 128) >> 3
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := (qh_2 & 64) >> 2
					hi2_2 := (qh_2 & 128) >> 3
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := (qh_3 & 64) >> 2
					hi2_3 := (qh_3 & 128) >> 3
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			}

			qdot1 := (qd1_0 + qd1_1) + (qd1_2 + qd1_3)
			qdot2 := (qd2_0 + qd2_1) + (qd2_2 + qd2_3)
			xsum1 := (xs1_0 + xs1_1) + (xs1_2 + xs1_3)
			xsum2 := (xs2_0 + xs2_1) + (xs2_2 + xs2_3)

			sum += d1*qdot1 - min1*xsum1
			sum += d2*qdot2 - min2*xsum2
		}
	}
	return sum
}

func DequantRowQ5K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ5KInto(row, cols, out)
	return out
}

// DequantRowQ5KInto decodes a Q5_K row into caller-owned storage. Token
// embedding lookup and batched prefill use it to avoid a throwaway row slice.
func DequantRowQ5KInto(row []byte, cols int, out []float32) {
	for b := range cols / 256 {
		base := b * 176
		if base+176 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		dmin := F16ToF32(binaryLE16(row[base+2:]))
		scales := row[base+4 : base+16]
		qh := row[base+16 : base+48]
		q := row[base+48 : base+176]
		yoff := b * 256
		is := 0
		u1 := byte(1)
		u2 := byte(2)
		for j := 0; j < 256; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)
			for l := range 32 {
				hi1 := byte(0)
				if qh[l]&u1 != 0 {
					hi1 = 16
				}
				hi2 := byte(0)
				if qh[l]&u2 != 0 {
					hi2 = 16
				}
				out[yoff+j+l] = d1*float32((q[l]&0x0f)+hi1) - min1
				out[yoff+j+32+l] = d2*float32((q[l]>>4)+hi2) - min2
			}
			q = q[32:]
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}
