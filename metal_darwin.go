//go:build darwin && cgo && metal

package main

import metalbackend "gopherllm/internal/metal"

const metalQ6KMinRows = 32768

type MetalWeight struct {
	q6   *metalbackend.Weight
	typ  GGMLType
	rows int
	cols int
}

func MetalAvailable() bool {
	return metalbackend.Available()
}

func prepareMetalWeight(data []byte, typ GGMLType, rows, cols int) *MetalWeight {
	if typ != GGMLTypeQ6_K || rows < metalQ6KMinRows || cols <= 0 || cols%256 != 0 {
		return nil
	}
	q6 := metalbackend.PrepareQ6K(data, rows, cols)
	if q6 == nil {
		return nil
	}
	return &MetalWeight{q6: q6, typ: typ, rows: rows, cols: cols}
}

func matvecMetalQ6KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if w == nil || w.q6 == nil || w.typ != GGMLTypeQ6_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ6K(w.q6, x, *out)
}

func releaseMetalWeight(w *MetalWeight) {
	if w == nil || w.q6 == nil {
		return
	}
	metalbackend.Release(w.q6)
	w.q6 = nil
}
