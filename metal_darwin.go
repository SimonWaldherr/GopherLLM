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
	q8   *metalbackend.Weight
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

// metalWeightMayUseDirect gates standalone matvec offload. Contracting Q4_K
// FFN weights have a separate preparation exception for fused FFNs; this
// must not lower the standalone threshold for narrow attention projections.
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
	case GGMLTypeQ8_0:
		return rows >= 2048
	default:
		return false
	}
}

func prepareMetalWeight(data []byte, typ GGMLType, rows, cols int, borrow bool) *MetalWeight {
	// Contracting Q4_K FFN weights are prepared for fusion only; standalone
	// attention-sized Q4_K matvecs retain their CPU crossover.
	fusedQ4Down := typ == GGMLTypeQ4_K && rows >= 2048 && cols/2 >= rows
	if cols <= 0 || cols%256 != 0 || (!metalWeightMayUseDirect(typ, rows) && !fusedQ4Down) {
		return nil
	}
	// Keep narrow Qwen attention projections on the CPU. Retain large FFN
	// gate/up, contracting FFN down, and vocabulary projections for offload.
	if typ == GGMLTypeQ8_0 && rows < 8192 && cols/2 < rows {
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
	case GGMLTypeQ8_0:
		w.q8 = metalbackend.PrepareQ8_0(data, rows, cols, borrow)
	default:
		return nil
	}
	if w.q4 == nil && w.q5 == nil && w.q6 == nil && w.q8 == nil {
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

// argmaxMetalQ6KBatch keeps a short Q6_K vocabulary verification window on
// the GPU. xs is contiguous [batch][hidden] data and tokens receives one
// greedy winner per position. Unlike the single-token path it deliberately
// has no repeat penalty: each verifier position has different history, so
// sharing one penalty set would be incorrect. The backend preserves the
// sampler's finite-value filtering and lowest-token tie rule.
func argmaxMetalQ6KBatch(w *MetalWeight, xs []float32, tokens []uint32, batch int) bool {
	if !metalWeightUsesDirect(w) || w.q6 == nil || w.typ != GGMLTypeQ6_K || batch <= 0 || batch > metalBatchArgmaxMaxTokens ||
		w.cols <= 0 || batch > len(xs)/w.cols || len(tokens) < batch {
		return false
	}
	return metalbackend.ArgmaxQ6KBatch(w.q6, xs, tokens, batch)
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
	if metalFusedFFNEnabled && metalQ4DownSwiGLUWeightsReady(gate, up, down) {
		if len(x) < gate.cols {
			return false
		}
		ensureLenNoClear(out, down.rows)
		return metalbackend.MatvecQ4K2SwiGLUQ4KBatch(gate.q4, up.q4, down.q4, x, *out, 1)
	}
	if metalFusedFFNEnabled && metalQ8SwiGLUWeightsReady(gate, up, down) {
		if len(x) < gate.cols {
			return false
		}
		ensureLenNoClear(out, down.rows)
		return metalbackend.MatvecQ8_0SwiGLUBatch(gate.q8, up.q8, down.q8, x, *out, 1)
	}
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
	if metalFusedFFNEnabled && metalQ4DownSwiGLUWeightsReady(gate, up, down) {
		if batch < 2 || batch > metalBatchFFNMaxTokens || batch > len(x)/gate.cols || batch > int(^uint(0)>>1)/down.rows {
			return false
		}
		ensureLenNoClear(out, batch*down.rows)
		return metalbackend.MatvecQ4K2SwiGLUQ4KBatch(gate.q4, up.q4, down.q4, x, *out, batch)
	}
	if metalFusedFFNEnabled && metalQ8SwiGLUWeightsReady(gate, up, down) {
		if batch < 2 || batch > metalBatchFFNMaxTokens || batch > len(x)/gate.cols || batch > int(^uint(0)>>1)/down.rows {
			return false
		}
		ensureLenNoClear(out, batch*down.rows)
		return metalbackend.MatvecQ8_0SwiGLUBatch(gate.q8, up.q8, down.q8, x, *out, batch)
	}
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

// metalBatchFFNPrefillChunk returns the larger runner-local default only when
// every layer of a standard dense decoder can take the GPU-resident SwiGLU
// batch path. Checking the actual prepared handles matters: LoadOptions may
// request Metal while a particular tensor is unsupported or preparation
// failed, and such a mixed graph must retain the CPU-oriented chunk size.
func (r *Runner) metalBatchFFNPrefillChunk() int {
	if r == nil || r.kind != loadedStandard || r.outOfCore || r.config.UsesMLA ||
		r.config.usesPlainMLP() || r.config.UseGELU || !metalFusedFFNEnabled || len(r.standard.Layers) == 0 {
		return 0
	}
	chunk := metalBatchFFNMaxTokens
	for i := range r.standard.Layers {
		layer := &r.standard.Layers[i]
		if layer.W1.Type == GGMLTypeQ8_0 {
			// Keep Qwen attention on its CPU-oriented slab size.
			chunk = 128
		}
		if layer.MoE != nil || layer.HasGateUp || len(layer.FFNUpBias) != 0 || len(layer.FFNDownBias) != 0 ||
			!metalSwiGLUBatchWeightsReady(layer.W1.Metal, layer.W3.Metal, layer.W2.Metal) {
			return 0
		}
	}
	return chunk
}

// metalSwiGLUBatchWeightsReady mirrors the shape and handle portion of
// matvecMetalSwiGLUBatchInto. It intentionally does not inspect activations:
// this is a load-time/default decision, while the dispatcher keeps validating
// every call before submitting work to Metal.
func metalSwiGLUBatchWeightsReady(gate, up, down *MetalWeight) bool {
	if metalQ8SwiGLUWeightsReady(gate, up, down) || metalQ4DownSwiGLUWeightsReady(gate, up, down) {
		return true
	}
	return metalWeightUsesDirect(gate) && metalWeightUsesDirect(up) && metalWeightUsesDirect(down) &&
		gate.q4 != nil && up.q4 != nil && down.q6 != nil &&
		gate.typ == GGMLTypeQ4_K && up.typ == GGMLTypeQ4_K && down.typ == GGMLTypeQ6_K &&
		gate.cols > 0 && gate.rows > 0 && down.rows > 0 &&
		gate.cols == up.cols && gate.rows == up.rows && down.cols == gate.rows
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
	if w.q8 != nil {
		metalbackend.Release(w.q8)
		w.q8 = nil
	}
}

func metalQ8SwiGLUWeightsReady(gate, up, down *MetalWeight) bool {
	return metalWeightUsesDirect(gate) && metalWeightUsesDirect(up) && metalWeightUsesDirect(down) &&
		gate.typ == GGMLTypeQ8_0 && up.typ == GGMLTypeQ8_0 && down.typ == GGMLTypeQ8_0 &&
		gate.q8 != nil && up.q8 != nil && down.q8 != nil &&
		gate.cols > 0 && gate.rows > 0 && down.rows > 0 &&
		gate.cols == up.cols && gate.rows == up.rows && down.cols == gate.rows
}

func matvecMetalQ8_0Into(w *MetalWeight, x []float32, rows, cols int, out *[]float32) bool {
	if !metalWeightUsesDirect(w) || w.typ != GGMLTypeQ8_0 || w.q8 == nil || w.rows != rows || w.cols != cols || len(x) < cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return metalbackend.MatvecQ8_0(w.q8, x, *out)
}

func argmaxMetalQ8_0Penalized(w *MetalWeight, x []float32, recent []uint32, penalty float32) (uint32, bool) {
	if !metalWeightUsesDirect(w) || w.typ != GGMLTypeQ8_0 || w.q8 == nil || len(x) < w.cols {
		return 0, false
	}
	return metalbackend.ArgmaxQ8_0Penalized(w.q8, x, recent, penalty)
}

// Q4_K contractions are eligible only as part of a complete FFN.
func metalQ4DownSwiGLUWeightsReady(gate, up, down *MetalWeight) bool {
	return metalWeightUsesDirect(gate) && metalWeightUsesDirect(up) && down != nil &&
		gate.typ == GGMLTypeQ4_K && up.typ == GGMLTypeQ4_K && down.typ == GGMLTypeQ4_K &&
		gate.q4 != nil && up.q4 != nil && down.q4 != nil &&
		gate.cols > 0 && gate.rows > 0 && down.rows >= 2048 && down.cols/2 >= down.rows &&
		gate.cols == up.cols && gate.rows == up.rows && down.cols == gate.rows
}
