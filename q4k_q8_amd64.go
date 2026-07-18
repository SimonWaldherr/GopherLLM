//go:build amd64

package gopherllm

import (
	"os"
	"sync"
)

// useQ8Activations enables the int8-activation fast path for Q4_K and Q6_K
// matvecs: the activation vector is quantized once per matvec to int8 with
// one scale per 256-element block (llama.cpp's Q8_K convention) and each
// weight row is dotted against it by a full-row VPMADDUBSW kernel
// (q4kDotQ8KRow / q6kDotQ8KRow). This is the same activation quantization
// llama.cpp applies unconditionally to every K-quant matvec, so it is on by
// default; set GOPHERLLM_Q8_ACTIVATIONS=0 to force the exact float kernels
// (A/B testing, accuracy debugging).
var useQ8Activations = hasAVX2 && hasF16C && os.Getenv("GOPHERLLM_Q8_ACTIVATIONS") != "0"

var (
	q8ScratchPool     = sync.Pool{New: func() any { s := make([]int8, 0, 16384); return &s }}
	xscaleScratchPool = sync.Pool{New: func() any { s := make([]float32, 0, 64); return &s }}
)

// acquireQ8 quantizes x once (shared across all rows/matrices of a matvec)
// and returns the int8 activations, per-256-element-block scales, and a
// release func that returns the scratch buffers to their pools. cols must be
// a multiple of 256, which every Q4_K/Q6_K matvec entry point guarantees
// before taking this path.
func acquireQ8(x []float32, cols int) (q8 []int8, xscale []float32, release func()) {
	blocks := cols / 256
	q8s := q8ScratchPool.Get().(*[]int8)
	scs := xscaleScratchPool.Get().(*[]float32)
	ensureLenNoClear(q8s, cols)
	ensureLenNoClear(scs, blocks)
	q8 = *q8s
	xscale = *scs
	q8kQuantize(&x[0], &q8[0], &xscale[0], blocks)
	return q8, xscale, func() {
		*q8s = q8
		q8ScratchPool.Put(q8s)
		*scs = xscale
		xscaleScratchPool.Put(scs)
	}
}

// dotQ4KRowsQ8 fills out[start:end] with Q4_K row dots against Q8K-quantized
// activations. xsums must hold the per-32-element float sums of the ORIGINAL
// activations (fillQ4KXSums); the dmin term stays exact float.
func dotQ4KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q4kDotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

// dotQ6KRowsQ8 is the Q6_K analogue. xsums must hold the per-16-element sums
// of the original activations pre-scaled by 32 (fillQ6KXSums16 +
// ScaleF32(xs, 32)), matching the float path's offset folding.
func dotQ6KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q6kDotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

type batchQ8Scratch struct {
	q8    []int8
	xsc   []float32
	xsums []float32
}

var batchQ8Pool = sync.Pool{New: func() any { return &batchQ8Scratch{} }}

// matvecBatchQ8 is the batched-prefill analogue of the int8-activation
// matvec: every prompt token's activation vector is quantized once, then each
// weight row is streamed from memory exactly once and dotted against all
// tokens in row tiles small enough to keep both the tile and one token's
// int8 activations cache-resident. Compared to the dequantize-to-f32 batch
// path this cuts activation traffic 4x and weight-decode instructions ~3-6x.
// Returns false (nothing written) for shapes or types it does not handle.
func matvecBatchQ8(w Weight, xs, outs [][]float32) bool {
	p := len(xs)
	if p == 0 {
		return false
	}
	cols := len(xs[0])
	if cols <= 0 || cols%256 != 0 || w.Rows <= 0 || w.Cols != cols {
		return false
	}
	blocks := cols / 256
	var rowBytes, sumsPerTok int
	var kernel func(*byte, *int8, *float32, *float32, int) float32
	switch w.Type {
	case GGMLTypeQ4_K:
		rowBytes = blocks * 144
		sumsPerTok = blocks * 8
		kernel = q4kDotQ8KRow
	case GGMLTypeQ5_K:
		rowBytes = blocks * 176
		sumsPerTok = blocks * 8
		kernel = q5kDotQ8KRow
	case GGMLTypeQ6_K:
		rowBytes = blocks * 210
		sumsPerTok = blocks * 16
		kernel = q6kDotQ8KRow
	case GGMLTypeQ8_0:
		rowBytes = blocks * 272
		// Q8_0 is symmetric (no dmin/offset term), so it needs no xsums;
		// keep one slot per token so the shared indexing stays valid.
		sumsPerTok = 1
		kernel = func(row *byte, q8 *int8, xsc *float32, _ *float32, blocks int) float32 {
			return q8_0DotQ8KRow(row, q8, xsc, blocks)
		}
	default:
		return false
	}
	if len(w.Raw) < w.Rows*rowBytes {
		return false
	}
	for t := range p {
		if len(xs[t]) < cols || len(outs[t]) < w.Rows {
			return false
		}
	}

	scratch := batchQ8Pool.Get().(*batchQ8Scratch)
	ensureLenNoClear(&scratch.q8, p*cols)
	ensureLenNoClear(&scratch.xsc, p*blocks)
	ensureLenNoClear(&scratch.xsums, p*sumsPerTok)
	q8All := scratch.q8
	xscAll := scratch.xsc
	xsumsAll := scratch.xsums
	for t := range p {
		q8kQuantize(&xs[t][0], &q8All[t*cols], &xscAll[t*blocks], blocks)
		sub := xsumsAll[t*sumsPerTok : (t+1)*sumsPerTok : (t+1)*sumsPerTok]
		switch w.Type {
		case GGMLTypeQ4_K, GGMLTypeQ5_K:
			fillQ4KXSums(xs[t], cols, &sub)
		case GGMLTypeQ6_K:
			fillQ6KXSums16(xs[t], cols, &sub)
			ScaleF32(sub, 32)
		}
	}

	// Row tile sized so a tile of raw rows (~80-120 KB) stays L2-resident
	// while the per-token int8 activations (cols bytes) run through L1.
	const rowTile = 16
	parallelRows(w.Rows, func(start, end int) {
		for tileStart := start; tileStart < end; tileStart += rowTile {
			tileEnd := min(tileStart+rowTile, end)
			for t := range p {
				q8 := q8All[t*cols:]
				xsc := xscAll[t*blocks:]
				xsum := xsumsAll[t*sumsPerTok:]
				out := outs[t]
				for r := tileStart; r < tileEnd; r++ {
					out[r] = kernel(&w.Raw[r*rowBytes], &q8[0], &xsc[0], &xsum[0], blocks)
				}
			}
		}
	})
	batchQ8Pool.Put(scratch)
	return true
}
