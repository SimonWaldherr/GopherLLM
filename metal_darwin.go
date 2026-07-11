//go:build darwin && cgo && metal

package gopherllm

import metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"

const (
	metalQ4KMinRows = 8192
	metalQ6KMinRows = 3072
)

type MetalWeight struct {
	q4   *metalbackend.Weight
	q6   *metalbackend.Weight
	typ  GGMLType
	rows int
	cols int
}

func MetalAvailable() bool {
	return metalbackend.Available()
}

func MetalError() string {
	return metalbackend.LastError()
}

func prepareMetalWeight(data []byte, typ GGMLType, rows, cols int) *MetalWeight {
	if cols <= 0 || cols%256 != 0 {
		return nil
	}
	// Ministral-3 3B keeps FFN gate/up in Q4_K and down/output in Q6_K. These
	// measured row floors select those large shapes while leaving small Q/K/V
	// projections on faster prepared CPU kernels. Setting --metal off is the
	// complete rollback if a device regresses.
	w := &MetalWeight{typ: typ, rows: rows, cols: cols}
	switch typ {
	case GGMLTypeQ4_K:
		if rows < metalQ4KMinRows {
			return nil
		}
		w.q4 = metalbackend.PrepareQ4K(data, rows, cols)
	case GGMLTypeQ6_K:
		if rows < metalQ6KMinRows {
			return nil
		}
		w.q6 = metalbackend.PrepareQ6K(data, rows, cols)
	default:
		return nil
	}
	if w.q4 == nil && w.q6 == nil {
		return nil
	}
	return w
}

func matvecMetalQ4KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if w == nil || w.q4 == nil || w.typ != GGMLTypeQ4_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ4K(w.q4, x, *out)
}

func matvecMetalQ6KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if w == nil || w.q6 == nil || w.typ != GGMLTypeQ6_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ6K(w.q6, x, *out)
}

func matvecMetalQ4K2Into(a, b *MetalWeight, x []float32, aRows, bRows, cols int, aOut, bOut *[]float32) bool {
	if a == nil || b == nil || a.q4 == nil || b.q4 == nil ||
		a.typ != GGMLTypeQ4_K || b.typ != GGMLTypeQ4_K ||
		a.rows != aRows || b.rows != bRows || a.cols != cols || b.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	return metalbackend.MatvecQ4K2(a.q4, b.q4, x, *aOut, *bOut)
}

func releaseMetalWeight(w *MetalWeight) {
	if w == nil {
		return
	}
	if w.q4 != nil {
		metalbackend.Release(w.q4)
		w.q4 = nil
	}
	if w.q6 != nil {
		metalbackend.Release(w.q6)
		w.q6 = nil
	}
}
