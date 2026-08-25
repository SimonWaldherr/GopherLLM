package gopherllm

import "sync"

type q4KQ8RowsTask struct {
	data               []byte
	q8                 []int8
	xscale, xsums, out []float32
	cols, rowBytes     int
}

func (t *q4KQ8RowsTask) runRows(start, end int) {
	// Keep the architecture-specific kernel ABI behind dotQ4KRowsQ8: amd64
	// uses noescape pointers while arm64 uses slices.
	dotQ4KRowsQ8(t.data, t.q8, t.xscale, t.xsums, t.cols, t.rowBytes, start, end, t.out)
}

var q4KQ8RowsTaskPool = sync.Pool{New: func() any { return new(q4KQ8RowsTask) }}

func MatvecQ4KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 144
	ensureLenNoClear(out, rows)
	if cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, cols, scratch)
		if useQ8Activations.Load() {
			q8, xsc, lease := acquireQ8(x, cols)
			task := q4KQ8RowsTaskPool.Get().(*q4KQ8RowsTask)
			task.data, task.q8, task.xscale, task.xsums, task.out = data, q8, xsc, xs, *out
			task.cols, task.rowBytes = cols, rowBytes
			parallelRowsTask(rows, task)
			*task = q4KQ8RowsTask{}
			q4KQ8RowsTaskPool.Put(task)
			releaseQ8(q8, xsc, lease)
		} else {
			parallelRows(rows, func(start, end int) {
				dotQ4KRowsWithXSums(data, x, xs, cols, rowBytes, start, end, *out)
			})
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ4KF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func dotQ4KRowsWithXSums(data []byte, x, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = dotQ4KF32WithXSums(data[off:off+rowBytes], x, xsums, cols)
	}
}

func DotQ4KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		block := row[base : base+144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q := block[16:144]
		xBlock := x[b*256 : b*256+256]

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

			for l := 0; l < 32; l += 8 {
				qv0 := qSub[l]
				qv1 := qSub[l+1]
				qv2 := qSub[l+2]
				qv3 := qSub[l+3]
				qv4 := qSub[l+4]
				qv5 := qSub[l+5]
				qv6 := qSub[l+6]
				qv7 := qSub[l+7]

				x1_0 := xSub1[l]
				x1_1 := xSub1[l+1]
				x1_2 := xSub1[l+2]
				x1_3 := xSub1[l+3]
				x1_4 := xSub1[l+4]
				x1_5 := xSub1[l+5]
				x1_6 := xSub1[l+6]
				x1_7 := xSub1[l+7]

				x2_0 := xSub2[l]
				x2_1 := xSub2[l+1]
				x2_2 := xSub2[l+2]
				x2_3 := xSub2[l+3]
				x2_4 := xSub2[l+4]
				x2_5 := xSub2[l+5]
				x2_6 := xSub2[l+6]
				x2_7 := xSub2[l+7]

				qd1_0 += float32(qv0&0x0f) * x1_0
				qd1_1 += float32(qv1&0x0f) * x1_1
				qd1_2 += float32(qv2&0x0f) * x1_2
				qd1_3 += float32(qv3&0x0f) * x1_3
				qd1_0 += float32(qv4&0x0f) * x1_4
				qd1_1 += float32(qv5&0x0f) * x1_5
				qd1_2 += float32(qv6&0x0f) * x1_6
				qd1_3 += float32(qv7&0x0f) * x1_7

				qd2_0 += float32(qv0>>4) * x2_0
				qd2_1 += float32(qv1>>4) * x2_1
				qd2_2 += float32(qv2>>4) * x2_2
				qd2_3 += float32(qv3>>4) * x2_3
				qd2_0 += float32(qv4>>4) * x2_4
				qd2_1 += float32(qv5>>4) * x2_5
				qd2_2 += float32(qv6>>4) * x2_6
				qd2_3 += float32(qv7>>4) * x2_7

				xs1_0 += x1_0 + x1_4
				xs1_1 += x1_1 + x1_5
				xs1_2 += x1_2 + x1_6
				xs1_3 += x1_3 + x1_7

				xs2_0 += x2_0 + x2_4
				xs2_1 += x2_1 + x2_5
				xs2_2 += x2_2 + x2_6
				xs2_3 += x2_3 + x2_7
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

func fillQ4KXSums(x []float32, cols int, scratch *[]float32) []float32 {
	groups := cols / 32
	ensureLenNoClear(scratch, groups)
	out := *scratch
	if hasQuantSIMD && groups > 0 && len(x) >= groups*32 {
		sumF32Groups32(&x[0], &out[0], groups)
		return out
	}
	for g := range groups {
		base := g * 32
		if base+32 > len(x) {
			out[g] = 0
			continue
		}
		xBlock := x[base : base+32]
		_ = xBlock[31]
		var s0, s1, s2, s3 float32
		for i := 0; i < 32; i += 8 {
			s0 += xBlock[i] + xBlock[i+4]
			s1 += xBlock[i+1] + xBlock[i+5]
			s2 += xBlock[i+2] + xBlock[i+6]
			s3 += xBlock[i+3] + xBlock[i+7]
		}
		out[g] = (s0 + s1) + (s2 + s3)
	}
	return out
}

func dotQ4KF32WithXSums(row []byte, x, xsums []float32, cols int) float32 {
	if hasQuantSIMD && cols > 0 && cols%256 == 0 && len(x) >= cols && len(xsums) >= cols/32 {
		return dotQ4KF32SIMDWithXSums(row, x, xsums, cols)
	}
	return dotQ4KF32ScalarWithXSums(row, x, xsums, cols)
}

// dotQ4KF32SIMDWithXSums computes a Q4_K row dot product using the SIMD
// block kernel. xsums must hold per-32-element sums of x (fillQ4KXSums).
func dotQ4KF32SIMDWithXSums(row []byte, x, xsums []float32, cols int) float32 {
	var qdots [8]float32
	var sum float32
	blocks := cols / 256
	blocks = min(blocks, len(row)/144)
	blocks = min(blocks, len(x)/256)
	blocks = min(blocks, len(xsums)/8)
	if blocks <= 0 {
		return 0
	}
	_ = row[blocks*144-1]
	_ = x[blocks*256-1]
	_ = xsums[blocks*8-1]
	for b := 0; b < blocks; b++ {
		base := b * 144
		block := row[base : base+144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q4kQDots8(&block[16], &x[b*256], &qdots[0])

		xsumBase := b * 8

		// Step 0: is = 0
		sc1_0, m1_0 := scales[0]&63, scales[4]&63
		sc2_0, m2_0 := scales[1]&63, scales[5]&63
		sum += d*byteToFloat32LUT[sc1_0]*qdots[0] - dmin*byteToFloat32LUT[m1_0]*xsums[xsumBase]
		sum += d*byteToFloat32LUT[sc2_0]*qdots[1] - dmin*byteToFloat32LUT[m2_0]*xsums[xsumBase+1]

		// Step 1: is = 2
		sc1_1, m1_1 := scales[2]&63, scales[6]&63
		sc2_1, m2_1 := scales[3]&63, scales[7]&63
		sum += d*byteToFloat32LUT[sc1_1]*qdots[2] - dmin*byteToFloat32LUT[m1_1]*xsums[xsumBase+2]
		sum += d*byteToFloat32LUT[sc2_1]*qdots[3] - dmin*byteToFloat32LUT[m2_1]*xsums[xsumBase+3]

		// Step 2: is = 4
		sc1_2, m1_2 := (scales[8]&0x0f)|((scales[0]>>6)<<4), (scales[8]>>4)|((scales[4]>>6)<<4)
		sc2_2, m2_2 := (scales[9]&0x0f)|((scales[1]>>6)<<4), (scales[9]>>4)|((scales[5]>>6)<<4)
		sum += d*byteToFloat32LUT[sc1_2]*qdots[4] - dmin*byteToFloat32LUT[m1_2]*xsums[xsumBase+4]
		sum += d*byteToFloat32LUT[sc2_2]*qdots[5] - dmin*byteToFloat32LUT[m2_2]*xsums[xsumBase+5]

		// Step 3: is = 6
		sc1_3, m1_3 := (scales[10]&0x0f)|((scales[2]>>6)<<4), (scales[10]>>4)|((scales[6]>>6)<<4)
		sc2_3, m2_3 := (scales[11]&0x0f)|((scales[3]>>6)<<4), (scales[11]>>4)|((scales[7]>>6)<<4)
		sum += d*byteToFloat32LUT[sc1_3]*qdots[6] - dmin*byteToFloat32LUT[m1_3]*xsums[xsumBase+6]
		sum += d*byteToFloat32LUT[sc2_3]*qdots[7] - dmin*byteToFloat32LUT[m2_3]*xsums[xsumBase+7]
	}
	return sum
}

func dotQ4KF32ScalarWithXSums(row []byte, x, xsums []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		block := row[base : base+144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q := block[16:144]
		xBlock := x[b*256 : b*256+256]

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

			for l := 0; l < 32; l += 8 {
				qv0 := qSub[l]
				qv1 := qSub[l+1]
				qv2 := qSub[l+2]
				qv3 := qSub[l+3]
				qv4 := qSub[l+4]
				qv5 := qSub[l+5]
				qv6 := qSub[l+6]
				qv7 := qSub[l+7]

				x1_0 := xSub1[l]
				x1_1 := xSub1[l+1]
				x1_2 := xSub1[l+2]
				x1_3 := xSub1[l+3]
				x1_4 := xSub1[l+4]
				x1_5 := xSub1[l+5]
				x1_6 := xSub1[l+6]
				x1_7 := xSub1[l+7]

				x2_0 := xSub2[l]
				x2_1 := xSub2[l+1]
				x2_2 := xSub2[l+2]
				x2_3 := xSub2[l+3]
				x2_4 := xSub2[l+4]
				x2_5 := xSub2[l+5]
				x2_6 := xSub2[l+6]
				x2_7 := xSub2[l+7]

				qd1_0 += float32(qv0&0x0f) * x1_0
				qd1_1 += float32(qv1&0x0f) * x1_1
				qd1_2 += float32(qv2&0x0f) * x1_2
				qd1_3 += float32(qv3&0x0f) * x1_3
				qd1_0 += float32(qv4&0x0f) * x1_4
				qd1_1 += float32(qv5&0x0f) * x1_5
				qd1_2 += float32(qv6&0x0f) * x1_6
				qd1_3 += float32(qv7&0x0f) * x1_7

				qd2_0 += float32(qv0>>4) * x2_0
				qd2_1 += float32(qv1>>4) * x2_1
				qd2_2 += float32(qv2>>4) * x2_2
				qd2_3 += float32(qv3>>4) * x2_3
				qd2_0 += float32(qv4>>4) * x2_4
				qd2_1 += float32(qv5>>4) * x2_5
				qd2_2 += float32(qv6>>4) * x2_6
				qd2_3 += float32(qv7>>4) * x2_7
			}

			qdot1 := (qd1_0 + qd1_1) + (qd1_2 + qd1_3)
			qdot2 := (qd2_0 + qd2_1) + (qd2_2 + qd2_3)
			xsumBase := b*8 + step*2
			xsum1 := xsums[xsumBase]
			xsum2 := xsums[xsumBase+1]

			sum += d1*qdot1 - min1*xsum1
			sum += d2*qdot2 - min2*xsum2
		}
	}
	return sum
}

func DequantRowQ4K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ4KInto(row, cols, out)
	return out
}

func DequantRowQ4KInto(row []byte, cols int, out []float32) {
	for b := range cols / 256 {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		dmin := F16ToF32(binaryLE16(row[base+2:]))
		scales := row[base+4 : base+16]
		q := row[base+16 : base+144]
		yoff := b * 256
		is := 0
		for j := 0; j < 256; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)
			for l := range 32 {
				out[yoff+j+l] = d1*float32(q[l]&0x0f) - min1
				out[yoff+j+32+l] = d2*float32(q[l]>>4) - min2
			}
			q = q[32:]
			is += 2
		}
	}
}
