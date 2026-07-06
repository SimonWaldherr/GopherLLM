package gopherllm

type PreparedQuantizedWeight struct {
	Type   GGMLType
	Rows   int
	Cols   int
	Blocks int
	Scales []float32
	Mins   []float32
}

const preparedQ6KMaxCols = 4096

func PrepareQuantizedWeight(data []byte, typ GGMLType, rows, cols int) *PreparedQuantizedWeight {
	if rows <= 0 || cols <= 0 || cols%256 != 0 {
		return nil
	}
	blocks := cols / 256
	switch typ {
	case GGMLTypeQ4_K:
		rowBytes := blocks * 144
		if len(data) < rows*rowBytes {
			return nil
		}
		p := &PreparedQuantizedWeight{
			Type:   typ,
			Rows:   rows,
			Cols:   cols,
			Blocks: blocks,
			Scales: make([]float32, rows*blocks*8),
			Mins:   make([]float32, rows*blocks*8),
		}
		for r := range rows {
			rowOff := r * rowBytes
			for b := range blocks {
				block := data[rowOff+b*144 : rowOff+(b+1)*144]
				d := F16ToF32(binaryLE16(block[0:]))
				dmin := F16ToF32(binaryLE16(block[2:]))
				scales := block[4:16]
				base := (r*blocks + b) * 8
				for i := range 8 {
					sc, m := getScaleMinK4(i, scales)
					p.Scales[base+i] = d * float32(sc)
					p.Mins[base+i] = dmin * float32(m)
				}
			}
		}
		return p
	case GGMLTypeQ6_K:
		if cols > preparedQ6KMaxCols {
			return nil
		}
		rowBytes := blocks * 210
		if len(data) < rows*rowBytes {
			return nil
		}
		p := &PreparedQuantizedWeight{
			Type:   typ,
			Rows:   rows,
			Cols:   cols,
			Blocks: blocks,
			Scales: make([]float32, rows*blocks*16),
		}
		for r := range rows {
			rowOff := r * rowBytes
			for b := range blocks {
				block := data[rowOff+b*210 : rowOff+(b+1)*210]
				d := F16ToF32(binaryLE16(block[208:]))
				scaleBytes := block[192:208]
				base := (r*blocks + b) * 16
				_ = scaleBytes[15]
				for i := range 16 {
					p.Scales[base+i] = d * int8ToFloat32LUT[scaleBytes[i]]
				}
			}
		}
		return p
	default:
		return nil
	}
}

func (p *PreparedQuantizedWeight) valid(typ GGMLType, rows, cols int) bool {
	return p != nil && p.Type == typ && p.Rows == rows && p.Cols == cols && p.Blocks == cols/256
}

func MatvecPreparedQ4KInto(data []byte, p *PreparedQuantizedWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !p.valid(GGMLTypeQ4_K, rows, cols) || len(x) < cols || cols%256 != 0 {
		return false
	}
	rowBytes := p.Blocks * 144
	if len(data) < rows*rowBytes || len(p.Scales) < rows*p.Blocks*8 || len(p.Mins) < rows*p.Blocks*8 {
		return false
	}
	if useQ8Activations || !hasPreparedQ4K {
		MatvecQ4KInto(data, x, rows, cols, out)
		return true
	}
	ensureLenNoClear(out, rows)
	scratch := xsumsScratchPool.Get().(*[]float32)
	xs := fillQ4KXSums(x, cols, scratch)
	parallelRows(rows, func(start, end int) {
		dotPreparedQ4KRowsWithXSums(data, p, x, xs, start, end, *out)
	})
	*scratch = xs
	xsumsScratchPool.Put(scratch)
	return true
}

func MatvecPreparedQ6KInto(data []byte, p *PreparedQuantizedWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !p.valid(GGMLTypeQ6_K, rows, cols) || len(x) < cols || cols%256 != 0 {
		return false
	}
	rowBytes := p.Blocks * 210
	if len(data) < rows*rowBytes || len(p.Scales) < rows*p.Blocks*16 {
		return false
	}
	if useQ8Activations || !hasQuantSIMD {
		MatvecQ6KInto(data, x, rows, cols, out)
		return true
	}
	ensureLenNoClear(out, rows)
	scratch := xsumsScratchPool.Get().(*[]float32)
	xs := fillQ6KXSums16(x, cols, scratch)
	ScaleF32(xs, 32)
	parallelRows(rows, func(start, end int) {
		dotPreparedQ6KRowsWithXSums(data, p, x, xs, start, end, *out)
	})
	*scratch = xs
	xsumsScratchPool.Put(scratch)
	return true
}

func MatvecPreparedQ4K2IntoWithXSums(aData []byte, aPrep *PreparedQuantizedWeight, aRows, aCols int, bData []byte, bPrep *PreparedQuantizedWeight, bRows, bCols int, x []float32, xSums *[]float32, aOut, bOut *[]float32) bool {
	if !aPrep.valid(GGMLTypeQ4_K, aRows, aCols) || !bPrep.valid(GGMLTypeQ4_K, bRows, bCols) || aCols <= 0 || aCols != bCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := aPrep.Blocks * 144
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes {
		return false
	}
	if useQ8Activations || !hasPreparedQ4K {
		return MatvecQ4K2IntoWithXSums(aData, aRows, aCols, bData, bRows, bCols, x, xSums, aOut, bOut)
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	xs := fillQ4KXSums(x, aCols, xSums)
	totalRows := aRows + bRows
	parallelRows(totalRows, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, aRows); as < ae {
			dotPreparedQ4KRowsWithXSums(aData, aPrep, x, xs, as, ae, *aOut)
		}
		if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
			dotPreparedQ4KRowsWithXSums(bData, bPrep, x, xs, bs-aRows, be-aRows, *bOut)
		}
	})
	return true
}

func MatvecPreparedQ4K3IntoWithXSums(aData []byte, aPrep *PreparedQuantizedWeight, aRows, aCols int, bData []byte, bPrep *PreparedQuantizedWeight, bRows, bCols int, cData []byte, cPrep *PreparedQuantizedWeight, cRows, cCols int, x []float32, xSums *[]float32, aOut, bOut, cOut *[]float32) bool {
	if !aPrep.valid(GGMLTypeQ4_K, aRows, aCols) || !bPrep.valid(GGMLTypeQ4_K, bRows, bCols) || !cPrep.valid(GGMLTypeQ4_K, cRows, cCols) || aCols <= 0 || aCols != bCols || aCols != cCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := aPrep.Blocks * 144
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes || len(cData) < cRows*rowBytes {
		return false
	}
	if useQ8Activations || !hasPreparedQ4K {
		return Q4KMatvec3IntoWithXSums(
			Q4KMatrix{Data: aData, Rows: aRows, Cols: aCols},
			Q4KMatrix{Data: bData, Rows: bRows, Cols: bCols},
			Q4KMatrix{Data: cData, Rows: cRows, Cols: cCols},
			x,
			xSums,
			aOut,
			bOut,
			cOut,
		)
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	ensureLenNoClear(cOut, cRows)
	xs := fillQ4KXSums(x, aCols, xSums)
	abRows := aRows + bRows
	totalRows := abRows + cRows
	parallelRows(totalRows, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, aRows); as < ae {
			dotPreparedQ4KRowsWithXSums(aData, aPrep, x, xs, as, ae, *aOut)
		}
		if bs, be := clippedRange(start, end, aRows, abRows); bs < be {
			dotPreparedQ4KRowsWithXSums(bData, bPrep, x, xs, bs-aRows, be-aRows, *bOut)
		}
		if cs, ce := clippedRange(start, end, abRows, totalRows); cs < ce {
			dotPreparedQ4KRowsWithXSums(cData, cPrep, x, xs, cs-abRows, ce-abRows, *cOut)
		}
	})
	return true
}

func dotPreparedQ4KRowsWithXSums(data []byte, p *PreparedQuantizedWeight, x, xsums []float32, start, end int, out []float32) {
	rowBytes := p.Blocks * 144
	for r := start; r < end; r++ {
		rowOff := r * rowBytes
		prepOff := r * p.Blocks * 8
		out[r] = dotPreparedQ4KF32NEONWithXSums(
			data[rowOff:rowOff+rowBytes],
			p.Scales[prepOff:prepOff+p.Blocks*8],
			p.Mins[prepOff:prepOff+p.Blocks*8],
			x,
			xsums,
			p.Blocks,
		)
	}
}

func dotPreparedQ6KRowsWithXSums(data []byte, p *PreparedQuantizedWeight, x, xsums []float32, start, end int, out []float32) {
	rowBytes := p.Blocks * 210
	for r := start; r < end; r++ {
		rowOff := r * rowBytes
		prepOff := r * p.Blocks * 16
		out[r] = dotPreparedQ6KF32SIMDWithXSums(
			data[rowOff:rowOff+rowBytes],
			p.Scales[prepOff:prepOff+p.Blocks*16],
			x,
			xsums,
			p.Blocks,
		)
	}
}

func dotPreparedQ4KF32NEONWithXSums(row []byte, scales, mins, x, xsums []float32, blocks int) float32 {
	if blocks > 0 {
		return q4kDotPrepared(&row[16], &x[0], &scales[0], &mins[0], &xsums[0], blocks)
	}
	return 0
}

func dotPreparedQ6KF32SIMDWithXSums(row []byte, scales, x, xsums []float32, blocks int) float32 {
	var qdots [16]float32
	var sum float32
	blocks = min(blocks, len(row)/210)
	blocks = min(blocks, len(scales)/16)
	blocks = min(blocks, len(x)/256)
	blocks = min(blocks, len(xsums)/16)
	if blocks <= 0 {
		return 0
	}
	_ = row[blocks*210-1]
	_ = scales[blocks*16-1]
	_ = x[blocks*256-1]
	_ = xsums[blocks*16-1]
	for b := 0; b < blocks; b++ {
		base := b * 210
		block := row[base : base+210]
		q6kQDots16(&block[0], &block[128], &x[b*256], &qdots[0])
		sc := scales[b*16 : b*16+16]
		xs := xsums[b*16 : b*16+16]
		_ = sc[15]
		_ = xs[15]
		sum += sc[0]*(qdots[0]-xs[0]) +
			sc[1]*(qdots[1]-xs[1]) +
			sc[2]*(qdots[2]-xs[2]) +
			sc[3]*(qdots[3]-xs[3]) +
			sc[4]*(qdots[4]-xs[4]) +
			sc[5]*(qdots[5]-xs[5]) +
			sc[6]*(qdots[6]-xs[6]) +
			sc[7]*(qdots[7]-xs[7]) +
			sc[8]*(qdots[8]-xs[8]) +
			sc[9]*(qdots[9]-xs[9]) +
			sc[10]*(qdots[10]-xs[10]) +
			sc[11]*(qdots[11]-xs[11]) +
			sc[12]*(qdots[12]-xs[12]) +
			sc[13]*(qdots[13]-xs[13]) +
			sc[14]*(qdots[14]-xs[14]) +
			sc[15]*(qdots[15]-xs[15])
	}
	return sum
}
