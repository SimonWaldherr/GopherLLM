package main

type PreparedQuantizedWeight struct {
	Type   GGMLType
	Rows   int
	Cols   int
	Blocks int
	Scales []float32
	Mins   []float32
}

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
	if !hasQuantNEON {
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

func MatvecPreparedQ4K2IntoWithXSums(aData []byte, aPrep *PreparedQuantizedWeight, aRows, aCols int, bData []byte, bPrep *PreparedQuantizedWeight, bRows, bCols int, x []float32, xSums *[]float32, aOut, bOut *[]float32) bool {
	if !aPrep.valid(GGMLTypeQ4_K, aRows, aCols) || !bPrep.valid(GGMLTypeQ4_K, bRows, bCols) || aCols <= 0 || aCols != bCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := aPrep.Blocks * 144
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes {
		return false
	}
	if !hasQuantNEON {
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
	if !hasQuantNEON {
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

func dotPreparedQ4KF32NEONWithXSums(row []byte, scales, mins, x, xsums []float32, blocks int) float32 {
	if blocks > 0 {
		return q4kDotPrepared(&row[16], &x[0], &scales[0], &mins[0], &xsums[0], blocks)
	}
	return 0
}
