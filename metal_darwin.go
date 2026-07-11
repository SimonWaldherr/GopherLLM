//go:build darwin && cgo && metal

package gopherllm

import (
	"os"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

const (
	metalQ4KPrepareMinRows = 1024
	metalQ6KPrepareMinRows = 1024
	metalQ4KDirectMinRows  = 8192
	metalQ6KDirectMinRows  = 3072
)

var metalFusedFFNEnabled = os.Getenv("GOPHERLLM_METAL_FUSED_FFN") != "0"

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

func prepareMetalWeight(data []byte, typ GGMLType, rows, cols int, borrow bool) *MetalWeight {
	if cols <= 0 || cols%256 != 0 {
		return nil
	}
	// Small Q/K/V handles are retained for the fused attention command buffer;
	// individual dispatch still uses the higher measured direct thresholds.
	w := &MetalWeight{typ: typ, rows: rows, cols: cols}
	switch typ {
	case GGMLTypeQ4_K:
		if rows < metalQ4KPrepareMinRows {
			return nil
		}
		w.q4 = metalbackend.PrepareQ4K(data, rows, cols, borrow)
	case GGMLTypeQ6_K:
		if rows < metalQ6KPrepareMinRows {
			return nil
		}
		w.q6 = metalbackend.PrepareQ6K(data, rows, cols, borrow)
	default:
		return nil
	}
	if w.q4 == nil && w.q6 == nil {
		return nil
	}
	return w
}

func metalWeightUsesDirect(w *MetalWeight) bool {
	if w == nil {
		return false
	}
	switch w.typ {
	case GGMLTypeQ4_K:
		return w.rows >= metalQ4KDirectMinRows || (w.rows >= 3072 && w.cols >= 4096)
	case GGMLTypeQ6_K:
		return w.rows >= metalQ6KDirectMinRows
	default:
		return false
	}
}

func matvecMetalQ4KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !metalWeightUsesDirect(w) || w.q4 == nil || w.typ != GGMLTypeQ4_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ4K(w.q4, x, *out)
}

func matvecMetalQ6KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !metalWeightUsesDirect(w) || w.q6 == nil || w.typ != GGMLTypeQ6_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ6K(w.q6, x, *out)
}

func matvecMetalQ4K2Into(a, b *MetalWeight, x []float32, aRows, bRows, cols int, aOut, bOut *[]float32) bool {
	if !metalWeightUsesDirect(a) || !metalWeightUsesDirect(b) || a.q4 == nil || b.q4 == nil ||
		a.typ != GGMLTypeQ4_K || b.typ != GGMLTypeQ4_K ||
		a.rows != aRows || b.rows != bRows || a.cols != cols || b.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	return metalbackend.MatvecQ4K2(a.q4, b.q4, x, *aOut, *bOut)
}

func matvecMetalQ4K2Q6KInto(qWeight, kWeight, vWeight *MetalWeight, x []float32, qRows, kRows, vRows, cols int, q, k, v *[]float32) bool {
	if qWeight == nil || kWeight == nil || vWeight == nil ||
		qWeight.q4 == nil || kWeight.q4 == nil || vWeight.q6 == nil ||
		qWeight.typ != GGMLTypeQ4_K || kWeight.typ != GGMLTypeQ4_K || vWeight.typ != GGMLTypeQ6_K ||
		qWeight.rows != qRows || kWeight.rows != kRows || vWeight.rows != vRows ||
		qWeight.cols != cols || kWeight.cols != cols || vWeight.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(q, qRows)
	ensureLenNoClear(k, kRows)
	ensureLenNoClear(v, vRows)
	return metalbackend.MatvecQ4K2Q6K(qWeight.q4, kWeight.q4, vWeight.q6, x, *q, *k, *v)
}

func matvecMetalSwiGLUInto(gate, up, down *MetalWeight, x []float32, out *[]float32) bool {
	if !metalFusedFFNEnabled || !metalWeightUsesDirect(gate) || !metalWeightUsesDirect(up) || !metalWeightUsesDirect(down) ||
		gate.q4 == nil || up.q4 == nil || down.q6 == nil ||
		gate.typ != GGMLTypeQ4_K || up.typ != GGMLTypeQ4_K || down.typ != GGMLTypeQ6_K ||
		gate.cols != up.cols || gate.rows != up.rows || down.cols != gate.rows || len(x) < gate.cols {
		return false
	}
	ensureLenNoClear(out, down.rows)
	return metalbackend.MatvecQ4K2SwiGLUQ6K(gate.q4, up.q4, down.q6, x, *out)
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
