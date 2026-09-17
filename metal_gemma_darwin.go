//go:build darwin && cgo && metal

package gopherllm

import (
	"os"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func metalGemmaEnabled() bool { return os.Getenv("GOPHERLLM_METAL_GEMMA") != "0" }

// Gemma E2B has contracting 1536-row down projections and 6144-row gates.
// Prepare those FFN tensors for fusion without changing standalone offload
// thresholds for attention projections or CPU-only models.
func prepareMetalGeluWeight(w *Weight, borrow bool) {
	if !metalGemmaEnabled() || w == nil || w.Metal != nil || w.F32 != nil || w.Rows < 1024 || w.Cols < 1024 || w.Cols%256 != 0 {
		return
	}
	m := &MetalWeight{typ: w.Type, rows: w.Rows, cols: w.Cols}
	switch w.Type {
	case GGMLTypeQ4_K:
		m.q4 = metalbackend.PrepareQ4K(w.Raw, w.Rows, w.Cols, borrow)
	case GGMLTypeQ6_K:
		m.q6 = metalbackend.PrepareQ6K(w.Raw, w.Rows, w.Cols, borrow)
	case GGMLTypeQ8_0:
		m.q8 = metalbackend.PrepareQ8_0(w.Raw, w.Rows, w.Cols, borrow)
	default:
		return
	}
	if m.q4 != nil || m.q6 != nil || m.q8 != nil {
		w.Metal = m
	}
}

func metalGeluHandle(w Weight) (*metalbackend.Weight, uint32) {
	if w.F32 != nil || w.Metal == nil || w.Rows != w.Metal.rows || w.Cols != w.Metal.cols || w.Type != w.Metal.typ {
		return nil, 0
	}
	switch w.Type {
	case GGMLTypeQ4_K:
		return w.Metal.q4, 4
	case GGMLTypeQ6_K:
		return w.Metal.q6, 6
	case GGMLTypeQ8_0:
		return w.Metal.q8, 8
	}
	return nil, 0
}

func matvecMetalGeGLUInto(g, u, d Weight, x []float32, batch int, out *[]float32) bool {
	if !metalGemmaEnabled() || batch < 1 || batch > metalBatchFFNMaxTokens || g.Cols <= 0 || d.Rows <= 0 || g.Rows != u.Rows || g.Cols != u.Cols || d.Cols != g.Rows || len(x)/batch != g.Cols || len(x)%batch != 0 || d.Rows > int(^uint(0)>>1)/batch {
		return false
	}
	gp, gq := metalGeluHandle(g)
	up, uq := metalGeluHandle(u)
	dp, dq := metalGeluHandle(d)
	if gp == nil || up == nil || dp == nil {
		return false
	}
	ensureLenNoClear(out, batch*d.Rows)
	return metalbackend.GeluFFN(gp, up, dp, [3]uint32{gq, uq, dq}, x, *out, batch)
}

func matvecMetalGemmaOutputInto(w Weight, x []float32, out *[]float32) bool {
	if !metalGemmaEnabled() {
		return false
	}
	p, q := metalGeluHandle(w)
	if p == nil {
		return false
	}
	ensureLenNoClear(out, w.Rows)
	return metalbackend.GemmaOutput(p, q, x, *out, nil, 1, nil)
}
func argmaxMetalGemmaOutput(w Weight, x []float32, recent []uint32, penalty float32) (uint32, bool) {
	if !metalGemmaEnabled() {
		return 0, false
	}
	p, q := metalGeluHandle(w)
	if p == nil {
		return 0, false
	}
	var next uint32
	ok := metalbackend.GemmaOutput(p, q, x, nil, recent, penalty, &next)
	return next, ok
}
