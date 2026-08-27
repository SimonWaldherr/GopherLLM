//go:build darwin && cgo && metal

package gopherllm

import (
	"os"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

const (
	metalQ4KDirectMinRows = 8192
	metalQ5KDirectMinRows = 3072
	metalQ6KDirectMinRows = 3072
	// Bound the process-wide Metal prefill workspace. Larger externally supplied
	// ForwardBatchInto calls retain the exact CPU batch implementation.
	metalBatchFFNMaxTokens = 256
)

var metalFusedFFNEnabled = os.Getenv("GOPHERLLM_METAL_FUSED_FFN") != "0"

type MetalWeight struct {
	q4   *metalbackend.Weight
	q5   *metalbackend.Weight
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

// metalWeightMayUseDirect is deliberately shared by preparation and dispatch.
// A Metal handle owns several shared buffers, so retaining a matrix below the
// direct-dispatch crossover wastes load time and memory: every current Metal
// call rejects it on the same row threshold. This matters for Ministral GQA,
// whose narrow Q/K/V and attention-output projections stay on the CPU while
// its large FFN and vocabulary projections still use Metal.
func metalWeightMayUseDirect(typ GGMLType, rows int) bool {
	switch typ {
	case GGMLTypeQ4_K:
		// Q4_K attention-output projections in Mistral-family GQA models are
		// typically 3K-5K rows. On an M2 Max their CPU Q8_K/NEON path is
		// 35-50% faster than a standalone Metal dispatch; reserve direct GPU
		// work for the large FFN/vocabulary-shaped matrices that amortize the
		// command-buffer boundary.
		return rows >= metalQ4KDirectMinRows
	case GGMLTypeQ5_K:
		// Q5_K carries Q4_K's scale structure with an extra bitplane, so its
		// CPU kernel is slightly slower per row than Q4_K's while the GPU cost
		// is nearly identical — the crossover therefore sits at the Q6_K
		// threshold rather than Q4_K's higher one.
		return rows >= metalQ5KDirectMinRows
	case GGMLTypeQ6_K:
		return rows >= metalQ6KDirectMinRows
	default:
		return false
	}
}

func prepareMetalWeight(data []byte, typ GGMLType, rows, cols int, borrow bool) *MetalWeight {
	if cols <= 0 || cols%256 != 0 || !metalWeightMayUseDirect(typ, rows) {
		return nil
	}
	w := &MetalWeight{typ: typ, rows: rows, cols: cols}
	switch typ {
	case GGMLTypeQ4_K:
		w.q4 = metalbackend.PrepareQ4K(data, rows, cols, borrow)
	case GGMLTypeQ5_K:
		w.q5 = metalbackend.PrepareQ5K(data, rows, cols, borrow)
	case GGMLTypeQ6_K:
		w.q6 = metalbackend.PrepareQ6K(data, rows, cols, borrow)
	default:
		return nil
	}
	if w.q4 == nil && w.q5 == nil && w.q6 == nil {
		return nil
	}
	return w
}

func metalWeightUsesDirect(w *MetalWeight) bool {
	return w != nil && metalWeightMayUseDirect(w.typ, w.rows)
}

func matvecMetalQ4KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !metalWeightUsesDirect(w) || w.q4 == nil || w.typ != GGMLTypeQ4_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ4K(w.q4, x, *out)
}

func matvecMetalQ5KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !metalWeightUsesDirect(w) || w.q5 == nil || w.typ != GGMLTypeQ5_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ5K(w.q5, x, *out)
}

func matvecMetalQ6KInto(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !metalWeightUsesDirect(w) || w.q6 == nil || w.typ != GGMLTypeQ6_K || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ6K(w.q6, x, *out)
}

func argmaxMetalQ6K(w *MetalWeight, x []float32) (uint32, bool) {
	return argmaxMetalQ6KPenalized(w, x, nil, 1)
}

func argmaxMetalQ6KPenalized(w *MetalWeight, x []float32, recent []uint32, repeatPenalty float32) (uint32, bool) {
	if !metalWeightUsesDirect(w) || w.q6 == nil || w.typ != GGMLTypeQ6_K || len(x) < w.cols {
		return 0, false
	}
	return metalbackend.ArgmaxQ6KPenalized(w.q6, x, recent, repeatPenalty)
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
	// GQA models such as Ministral/Devstral have only 1K K/V rows. Although
	// fusing Q/K/V removes two command buffers, those narrow projections still
	// do not amortize one GPU round-trip: real Ministral 3B/14B measurements
	// put this path 45-65% behind the fused CPU kernel. MHA-sized projections
	// remain eligible because each individual weight clears the direct gate.
	if !metalWeightUsesDirect(qWeight) || !metalWeightUsesDirect(kWeight) || !metalWeightUsesDirect(vWeight) ||
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

// matvecMetalSwiGLUBatchInto keeps the complete SwiGLU FFN on Metal across a
// contiguous prefill chunk. The generic batched graph owns X and Proj as
// [batch][width] slabs, so this can avoid materializing the much larger
// [batch][hidden] Gate, Up, and Hidden arrays on the CPU.
func matvecMetalSwiGLUBatchInto(gate, up, down *MetalWeight, x []float32, batch int, out *[]float32) bool {
	if !metalFusedFFNEnabled || batch < 2 || batch > metalBatchFFNMaxTokens || !metalWeightUsesDirect(gate) || !metalWeightUsesDirect(up) || !metalWeightUsesDirect(down) ||
		gate.q4 == nil || up.q4 == nil || down.q6 == nil ||
		gate.typ != GGMLTypeQ4_K || up.typ != GGMLTypeQ4_K || down.typ != GGMLTypeQ6_K ||
		gate.cols <= 0 || gate.rows <= 0 || down.rows <= 0 ||
		gate.cols != up.cols || gate.rows != up.rows || down.cols != gate.rows ||
		batch > len(x)/gate.cols || batch > int(^uint(0)>>1)/down.rows {
		return false
	}
	ensureLenNoClear(out, batch*down.rows)
	return metalbackend.MatvecQ4K2SwiGLUQ6KBatch(gate.q4, up.q4, down.q6, x, *out, batch)
}

func releaseMetalWeight(w *MetalWeight) {
	if w == nil {
		return
	}
	if w.q4 != nil {
		metalbackend.Release(w.q4)
		w.q4 = nil
	}
	if w.q5 != nil {
		metalbackend.Release(w.q5)
		w.q5 = nil
	}
	if w.q6 != nil {
		metalbackend.Release(w.q6)
		w.q6 = nil
	}
}
