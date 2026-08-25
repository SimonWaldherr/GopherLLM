package gopherllm

import "sync"

type q4K2Q6KRowsTask struct {
	aData, bData, cData      []byte
	x                        []float32
	q8                       []int8
	xscale, q4xs, q6xs       []float32
	aOut, bOut, cOut         []float32
	aRows, abRows, totalRows int
	cols, q4RowBytes         int
	q6RowBytes               int
	useQ8                    bool
}

func (t *q4K2Q6KRowsTask) runRows(start, end int) {
	if as, ae := clippedRange(start, end, 0, t.aRows); as < ae {
		if t.useQ8 {
			dotQ4KRowsQ8(t.aData, t.q8, t.xscale, t.q4xs, t.cols, t.q4RowBytes, as, ae, t.aOut)
		} else {
			dotQ4KRowsWithXSums(t.aData, t.x, t.q4xs, t.cols, t.q4RowBytes, as, ae, t.aOut)
		}
	}
	if bs, be := clippedRange(start, end, t.aRows, t.abRows); bs < be {
		if t.useQ8 {
			dotQ4KRowsQ8(t.bData, t.q8, t.xscale, t.q4xs, t.cols, t.q4RowBytes, bs-t.aRows, be-t.aRows, t.bOut)
		} else {
			dotQ4KRowsWithXSums(t.bData, t.x, t.q4xs, t.cols, t.q4RowBytes, bs-t.aRows, be-t.aRows, t.bOut)
		}
	}
	if cs, ce := clippedRange(start, end, t.abRows, t.totalRows); cs < ce {
		if t.useQ8 {
			dotQ6KRowsQ8(t.cData, t.q8, t.xscale, t.q6xs, t.cols, t.q6RowBytes, cs-t.abRows, ce-t.abRows, t.cOut)
		} else {
			dotQ6KRowsWithXSums(t.cData, t.x, t.q6xs, t.cols, t.q6RowBytes, cs-t.abRows, ce-t.abRows, t.cOut)
		}
	}
}

var q4K2Q6KRowsTaskPool = sync.Pool{New: func() any { return new(q4K2Q6KRowsTask) }}

func MatvecQ4K2Into(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, x []float32, aOut, bOut *[]float32) bool {
	scratch := []float32{}
	return MatvecQ4K2IntoWithXSums(aData, aRows, aCols, bData, bRows, bCols, x, &scratch, aOut, bOut)
}

func MatvecQ4K2IntoWithXSums(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, x []float32, xSums *[]float32, aOut, bOut *[]float32) bool {
	if aCols <= 0 || aCols != bCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := (aCols / 256) * 144
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	xs := fillQ4KXSums(x, aCols, xSums)
	totalRows := aRows + bRows
	if useQ8Activations.Load() {
		q8, xsc, lease := acquireQ8(x, aCols)
		parallelRows(totalRows, func(start, end int) {
			if as, ae := clippedRange(start, end, 0, aRows); as < ae {
				dotQ4KRowsQ8(aData, q8, xsc, xs, aCols, rowBytes, as, ae, *aOut)
			}
			if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
				dotQ4KRowsQ8(bData, q8, xsc, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
			}
		})
		releaseQ8(q8, xsc, lease)
		return true
	}
	parallelRows(totalRows, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, aRows); as < ae {
			dotQ4KRowsWithXSums(aData, x, xs, aCols, rowBytes, as, ae, *aOut)
		}
		if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
			dotQ4KRowsWithXSums(bData, x, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
		}
	})
	return true
}

func MatvecQ4K2Q6KIntoWithXSums(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, cData []byte, cRows, cCols int, x []float32, q4Sums *[]float32, aOut, bOut, cOut *[]float32) bool {
	if aCols <= 0 || aCols != bCols || aCols != cCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	q4RowBytes := (aCols / 256) * 144
	q6RowBytes := (aCols / 256) * 210
	if len(aData) < aRows*q4RowBytes || len(bData) < bRows*q4RowBytes || len(cData) < cRows*q6RowBytes {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	ensureLenNoClear(cOut, cRows)
	q4xs := fillQ4KXSums(x, aCols, q4Sums)
	q6Scratch := xsumsScratchPool.Get().(*[]float32)
	q6xs := fillQ6KXSums16(x, aCols, q6Scratch)
	ScaleF32(q6xs, 32)
	abRows := aRows + bRows
	totalRows := abRows + cRows
	task := q4K2Q6KRowsTaskPool.Get().(*q4K2Q6KRowsTask)
	task.aData, task.bData, task.cData, task.x = aData, bData, cData, x
	task.q4xs, task.q6xs = q4xs, q6xs
	task.aOut, task.bOut, task.cOut = *aOut, *bOut, *cOut
	task.aRows, task.abRows, task.totalRows = aRows, abRows, totalRows
	task.cols, task.q4RowBytes, task.q6RowBytes = aCols, q4RowBytes, q6RowBytes
	if useQ8Activations.Load() {
		q8, xsc, lease := acquireQ8(x, aCols)
		task.q8, task.xscale, task.useQ8 = q8, xsc, true
		parallelRowsTask(totalRows, task)
		releaseQ8(q8, xsc, lease)
	} else {
		parallelRowsTask(totalRows, task)
	}
	*task = q4K2Q6KRowsTask{}
	q4K2Q6KRowsTaskPool.Put(task)
	*q6Scratch = q6xs
	xsumsScratchPool.Put(q6Scratch)
	return true
}

func MatvecQ6K2Into(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, x []float32, aOut, bOut *[]float32) bool {
	if aCols <= 0 || aCols != bCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := (aCols / 256) * 210
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	scratch := xsumsScratchPool.Get().(*[]float32)
	xs := fillQ6KXSums16(x, aCols, scratch)
	ScaleF32(xs, 32)
	totalRows := aRows + bRows
	if useQ8Activations.Load() {
		q8, xsc, lease := acquireQ8(x, aCols)
		parallelRows(totalRows, func(start, end int) {
			if as, ae := clippedRange(start, end, 0, aRows); as < ae {
				dotQ6KRowsQ8(aData, q8, xsc, xs, aCols, rowBytes, as, ae, *aOut)
			}
			if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
				dotQ6KRowsQ8(bData, q8, xsc, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
			}
		})
		releaseQ8(q8, xsc, lease)
	} else {
		parallelRows(totalRows, func(start, end int) {
			if as, ae := clippedRange(start, end, 0, aRows); as < ae {
				dotQ6KRowsWithXSums(aData, x, xs, aCols, rowBytes, as, ae, *aOut)
			}
			if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
				dotQ6KRowsWithXSums(bData, x, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
			}
		})
	}
	*scratch = xs
	xsumsScratchPool.Put(scratch)
	return true
}

func MatvecQ6K3Into(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, cData []byte, cRows, cCols int, x []float32, aOut, bOut, cOut *[]float32) bool {
	if aCols <= 0 || aCols != bCols || aCols != cCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := (aCols / 256) * 210
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes || len(cData) < cRows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	ensureLenNoClear(cOut, cRows)
	scratch := xsumsScratchPool.Get().(*[]float32)
	xs := fillQ6KXSums16(x, aCols, scratch)
	ScaleF32(xs, 32)
	abRows := aRows + bRows
	totalRows := abRows + cRows
	if useQ8Activations.Load() {
		q8, xsc, lease := acquireQ8(x, aCols)
		parallelRows(totalRows, func(start, end int) {
			if as, ae := clippedRange(start, end, 0, aRows); as < ae {
				dotQ6KRowsQ8(aData, q8, xsc, xs, aCols, rowBytes, as, ae, *aOut)
			}
			if bs, be := clippedRange(start, end, aRows, abRows); bs < be {
				dotQ6KRowsQ8(bData, q8, xsc, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
			}
			if cs, ce := clippedRange(start, end, abRows, totalRows); cs < ce {
				dotQ6KRowsQ8(cData, q8, xsc, xs, cCols, rowBytes, cs-abRows, ce-abRows, *cOut)
			}
		})
		releaseQ8(q8, xsc, lease)
	} else {
		parallelRows(totalRows, func(start, end int) {
			if as, ae := clippedRange(start, end, 0, aRows); as < ae {
				dotQ6KRowsWithXSums(aData, x, xs, aCols, rowBytes, as, ae, *aOut)
			}
			if bs, be := clippedRange(start, end, aRows, abRows); bs < be {
				dotQ6KRowsWithXSums(bData, x, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
			}
			if cs, ce := clippedRange(start, end, abRows, totalRows); cs < ce {
				dotQ6KRowsWithXSums(cData, x, xs, cCols, rowBytes, cs-abRows, ce-abRows, *cOut)
			}
		})
	}
	*scratch = xs
	xsumsScratchPool.Put(scratch)
	return true
}
