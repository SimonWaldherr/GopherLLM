//go:build darwin && cgo && metal

package engine

import metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"

type MetalLittleBitWeight struct {
	handle *metalbackend.LittleBitWeight
	rows   int
	cols   int
}

func newMetalLittleBitWeight(w *LittleBitWeight, rows, cols int) *MetalLittleBitWeight {
	if w == nil {
		return nil
	}
	main := metalLittleBitFactors(w.Main)
	var residual *metalbackend.LittleBitFactors
	if w.Residual != nil {
		f := metalLittleBitFactors(*w.Residual)
		residual = &f
	}
	handle := metalbackend.PrepareLittleBit(rows, cols, main, residual, metalLittleBitScale(w.Bias))
	if handle == nil {
		return nil
	}
	return &MetalLittleBitWeight{handle: handle, rows: rows, cols: cols}
}

func metalLittleBitFactors(f LittleBitFactors) metalbackend.LittleBitFactors {
	return metalbackend.LittleBitFactors{
		Rank: f.Rank, U: f.U, V: f.V,
		U1: metalLittleBitScale(f.U1), U2: metalLittleBitScale(f.U2),
		V1: metalLittleBitScale(f.V1), V2: metalLittleBitScale(f.V2),
	}
}

func metalLittleBitScale(s LittleBitScale) metalbackend.LittleBitScale {
	var kind metalbackend.LittleBitScaleKind
	switch s.Type {
	case GGMLTypeF32:
		kind = metalbackend.LittleBitF32
	case GGMLTypeF16:
		kind = metalbackend.LittleBitF16
	case GGMLTypeBF16:
		kind = metalbackend.LittleBitBF16
	}
	return metalbackend.LittleBitScale{Data: s.Raw, Kind: kind}
}

func matvecMetalLittleBitInto(w *MetalLittleBitWeight, x []float32, rows, cols int, out *[]float32) bool {
	if w == nil || w.rows != rows || w.cols != cols || len(x) != cols {
		return false
	}
	ensureLenNoClear(out, rows)
	return matvecMetalLittleBitBatch(w, x, *out, 1)
}

func matvecMetalLittleBitBatch(w *MetalLittleBitWeight, x, out []float32, batch int) bool {
	return w != nil && metalbackend.MatvecLittleBit(w.handle, x, out, batch)
}

func metalLittleBitProjection(w Weight) *metalbackend.LittleBitWeight {
	if w.LittleBit == nil || w.LittleBit.Metal == nil {
		return nil
	}
	h := w.LittleBit.Metal
	if h.rows != w.Rows || h.cols != w.Cols {
		return nil
	}
	return h.handle
}

func matvecMetalLittleBitQKV(q, k, v Weight, x, qOut, kOut, vOut []float32, batch int) bool {
	return metalbackend.MatvecLittleBitQKV(metalLittleBitProjection(q), metalLittleBitProjection(k),
		metalLittleBitProjection(v), x, qOut, kOut, vOut, batch)
}

func matvecMetalLittleBitQKVInto(q, k, v Weight, x []float32, qOut, kOut, vOut *[]float32) bool {
	qm, km, vm := metalLittleBitProjection(q), metalLittleBitProjection(k), metalLittleBitProjection(v)
	if qm == nil || km == nil || vm == nil || q.Cols != k.Cols || q.Cols != v.Cols || len(x) != q.Cols {
		return false
	}
	ensureLenNoClear(qOut, q.Rows)
	ensureLenNoClear(kOut, k.Rows)
	ensureLenNoClear(vOut, v.Rows)
	return metalbackend.MatvecLittleBitQKV(qm, km, vm, x, *qOut, *kOut, *vOut, 1)
}

func matvecMetalLittleBitSwiGLUInto(gate, up, down Weight, x []float32, batch int, out *[]float32) bool {
	gm, um, dm := metalLittleBitProjection(gate), metalLittleBitProjection(up), metalLittleBitProjection(down)
	if gm == nil || um == nil || dm == nil || batch <= 0 || batch > 256 ||
		gate.Cols != up.Cols || gate.Rows != up.Rows || gate.Rows != down.Cols ||
		batch > len(x)/gate.Cols || batch > int(^uint(0)>>1)/down.Rows {
		return false
	}
	ensureLenNoClear(out, batch*down.Rows)
	return metalbackend.MatvecLittleBitSwiGLU(gm, um, dm, x, *out, batch)
}

func releaseMetalLittleBitWeight(w *MetalLittleBitWeight) {
	if w != nil {
		metalbackend.ReleaseLittleBit(w.handle)
	}
}

// LittleBitMetalMemory reports the loaded Metal factor buffers and the shared
// process-wide LittleBit activation workspace. Bytes exclude KV, ordinary
// tensors, model file mappings, and Metal driver allocations.
func (r *Runner) LittleBitMetalMemory() (factors, scratch uint64) {
	if r == nil {
		return 0, 0
	}
	r.genLock.Lock()
	defer r.genLock.Unlock()
	seen := make(map[*MetalLittleBitWeight]bool)
	add := func(w Weight) {
		if w.LittleBit != nil {
			h := w.LittleBit.Metal
			if h != nil && !seen[h] {
				factors += uint64(metalbackend.LittleBitWeightBytes(h.handle))
				seen[h] = true
			}
		}
	}
	add(r.standard.Output)
	for _, l := range r.standard.Layers {
		for _, w := range [...]Weight{l.WQ, l.WK, l.WV, l.WO, l.W1, l.W2, l.W3} {
			add(w)
		}
	}
	return factors, metalbackend.LittleBitScratchBytes()
}
