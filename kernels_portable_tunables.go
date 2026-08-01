//go:build !amd64

package gopherllm

import (
	"os"
	"runtime"
	"sync"
)

// The int8-activation ("Q8K") matvec path on every non-amd64 target.
//
// This used to be a wall of panics: the path was amd64-only because it was
// written against VPMADDUBSW, so `useQ8Activations` was a compile-time false
// and all of this was dead code. That left Apple Silicon — where the whole
// point is that decode has compute headroom under a much wider memory bus —
// dequantizing every weight to f32 and doing float FMAs.
//
// The arithmetic now lives in quant_q8k_portable.go, shared by all
// architectures and differentially tested against the amd64 assembly. Where a
// target also has hand-written int8 SIMD (hasQ8KDotAsm), the row dots go
// through it; otherwise they go through the portable scalar kernels.
//
// Default-on is deliberately gated on hasQ8KDotAsm rather than on mere
// availability. On a target whose FLOAT path is already vectorised — arm64 has
// NEON q4kDotPrepared/q4kQDots8 — scalar integer dots can lose to vectorised
// float ones, and turning this on by default would be a silent regression. So
// where there is no int8 assembly the path is merely *offered*: the autotuner
// measures it against the float path on the real machine and model
// (tuneQ8Activations, gated on q8ActivationsAvailable) and GOPHERLLM_Q8_ACTIVATIONS
// forces the decision either way.

// useQ8Activations mirrors the amd64 variable of the same name. It is a var
// rather than a const now, so the branches in the matvec entry points are live
// and the autotuner can flip them.
var useQ8Activations = defaultQ8Activations()

func defaultQ8Activations() bool {
	switch os.Getenv("GOPHERLLM_Q8_ACTIVATIONS") {
	case "0":
		return false
	case "1":
		return true
	}
	return hasQ8KDotAsm
}

var (
	q8ScratchPool     = sync.Pool{New: func() any { s := make([]int8, 0, 16384); return &s }}
	xscaleScratchPool = sync.Pool{New: func() any { s := make([]float32, 0, 64); return &s }}
)

// acquireQ8 quantizes x once for the whole matvec and hands back the int8
// activations, the per-256-block scales, and a release func. cols must be a
// multiple of 256, which every caller checks before taking this path.
func acquireQ8(x []float32, cols int) (q8 []int8, xscale []float32, release func()) {
	blocks := cols / 256
	q8s := q8ScratchPool.Get().(*[]int8)
	scs := xscaleScratchPool.Get().(*[]float32)
	ensureLenNoClear(q8s, cols)
	ensureLenNoClear(scs, blocks)
	q8 = *q8s
	xscale = *scs
	q8kQuantize(x, q8, xscale, blocks)
	return q8, xscale, func() {
		*q8s = q8
		q8ScratchPool.Put(q8s)
		*scs = xscale
		xscaleScratchPool.Put(scs)
	}
}

func dotQ4KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q4kDotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ5KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q5kDotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ6KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q6kDotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ8_0RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q8_0DotQ8KRow(data[r*rowBytes:], q8, xscale, blocks)
	}
}

func dotQ4_0RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q4_0DotQ8KRowPortable(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ4_1RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q4_1DotQ8KRowPortable(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotMXFP4RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = mxfp4DotQ8KRowPortable(data[r*rowBytes:], q8, xscale, blocks)
	}
}

// q8kRowLayout describes how one weight type lays out a 256-element
// superchunk and how many activation-sum slots it needs per token, so the
// batched path and the single matvec agree on both.
type q8kRowLayout struct {
	rowBytes   int
	sumsPerTok int
	dot        func(row []byte, q8 []int8, xsc, xsums []float32, blocks int) float32
}

func q8kLayoutFor(t GGMLType, blocks int) (q8kRowLayout, bool) {
	switch t {
	case GGMLTypeQ4_K:
		return q8kRowLayout{blocks * 144, blocks * 8, q4kDotQ8KRow}, true
	case GGMLTypeQ5_K:
		return q8kRowLayout{blocks * 176, blocks * 8, q5kDotQ8KRow}, true
	case GGMLTypeQ6_K:
		return q8kRowLayout{blocks * 210, blocks * 16, q6kDotQ8KRow}, true
	case GGMLTypeQ8_0:
		// Symmetric: no offset term, so one dummy sums slot keeps the shared
		// per-token indexing valid.
		return q8kRowLayout{blocks * 272, 1, func(row []byte, q8 []int8, xsc, _ []float32, blocks int) float32 {
			return q8_0DotQ8KRow(row, q8, xsc, blocks)
		}}, true
	case GGMLTypeQ4_0:
		return q8kRowLayout{blocks * 144, blocks * 8, q4_0DotQ8KRowPortable}, true
	case GGMLTypeQ4_1:
		return q8kRowLayout{blocks * 160, blocks * 8, q4_1DotQ8KRowPortable}, true
	case GGMLTypeMXFP4:
		return q8kRowLayout{blocks * 136, 1, func(row []byte, q8 []int8, xsc, _ []float32, blocks int) float32 {
			return mxfp4DotQ8KRowPortable(row, q8, xsc, blocks)
		}}, true
	}
	return q8kRowLayout{}, false
}

// fillQ8KXSums writes the activation-sum term this weight type needs.
func fillQ8KXSums(t GGMLType, x []float32, cols int, sub *[]float32) {
	switch t {
	case GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ4_0, GGMLTypeQ4_1:
		fillQ4KXSums(x, cols, sub)
	case GGMLTypeQ6_K:
		fillQ6KXSums16(x, cols, sub)
		ScaleF32(*sub, 32)
	}
}

type batchQ8Scratch struct {
	q8    []int8
	xsc   []float32
	xsums []float32
}

var batchQ8Pool = sync.Pool{New: func() any { return &batchQ8Scratch{} }}

// matvecBatchQ8 is the batched-prefill analogue of the int8-activation matvec:
// each prompt token's activations are quantized once, then every weight row is
// streamed from memory exactly once and dotted against all tokens in row tiles
// small enough that both the tile and one token's int8 activations stay
// cache-resident.
//
// This used to be a hardcoded `return false` on non-amd64, which meant prefill
// on arm64 had no batched path at all: it fell back to re-reading and
// re-decoding every weight row once per prompt token. Returns false (writing
// nothing) for shapes and types it does not handle.
func matvecBatchQ8(w Weight, xs, outs [][]float32) bool {
	if !useQ8Activations {
		return false
	}
	p := len(xs)
	if p == 0 {
		return false
	}
	cols := len(xs[0])
	if cols <= 0 || cols%256 != 0 || w.Rows <= 0 || w.Cols != cols {
		return false
	}
	blocks := cols / 256
	layout, ok := q8kLayoutFor(w.Type, blocks)
	if !ok {
		return false
	}
	if len(w.Raw) < w.Rows*layout.rowBytes {
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
	ensureLenNoClear(&scratch.xsums, p*layout.sumsPerTok)
	q8All, xscAll, xsumsAll := scratch.q8, scratch.xsc, scratch.xsums
	for t := range p {
		q8kQuantize(xs[t], q8All[t*cols:], xscAll[t*blocks:], blocks)
		sub := xsumsAll[t*layout.sumsPerTok : (t+1)*layout.sumsPerTok : (t+1)*layout.sumsPerTok]
		fillQ8KXSums(w.Type, xs[t], cols, &sub)
	}

	// Row tile sized so a tile of raw rows stays L2-resident while each token's
	// int8 activations run through L1.
	const rowTile = 16
	parallelRows(w.Rows, func(start, end int) {
		for tileStart := start; tileStart < end; tileStart += rowTile {
			tileEnd := min(tileStart+rowTile, end)
			for t := range p {
				q8 := q8All[t*cols:]
				xsc := xscAll[t*blocks:]
				xsum := xsumsAll[t*layout.sumsPerTok:]
				out := outs[t]
				for r := tileStart; r < tileEnd; r++ {
					out[r] = layout.dot(w.Raw[r*layout.rowBytes:], q8, xsc, xsum, blocks)
				}
			}
		}
	})
	batchQ8Pool.Put(scratch)
	return true
}

// argmaxQ6KRowsQ8 finds argmax(W·x) over a Q6_K matrix with the same int8
// kernel as the materializing matvec, skipping the logits writeback. Returns
// false when the int8 path is off so callers keep the exact float kernel.
func argmaxQ6KRowsQ8(data []byte, x, xsums []float32, rows, cols, rowBytes int) (uint32, bool) {
	if !useQ8Activations || rows <= 0 || cols <= 0 || cols%256 != 0 {
		return 0, false
	}
	blocks := cols / 256
	q8, xsc, release := acquireQ8(x, cols)
	defer release()
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for r := start; r < end; r++ {
			// Rows are scanned ascending and the comparison is strict, so ties
			// keep the lowest token id — matching argmaxMatvecRows.
			v := q6kDotQ8KRow(data[r*rowBytes:], q8, xsc, xsums, blocks)
			if finiteLogit(v) && (!localFound || v > localValue) {
				localToken, localValue, localFound = r, v, true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken, bestValue, found = localToken, localValue, true
		}
		mu.Unlock()
	})
	return uint32(bestToken), true
}

// The int8-activation path is now implemented everywhere, so the autotuner is
// allowed to consider it. It reports available even without int8 assembly:
// "available" means correct and selectable, and letting --auto measure the
// scalar kernels against the vectorised float path on the real machine is
// strictly better than guessing which wins.

func q8ActivationsAvailable() bool { return true }

func q8ActivationsEnabled() bool { return useQ8Activations }

func setQ8Activations(on bool) { useQ8Activations = on }

func kvF16Available() bool { return true }

func kvF16Enabled() bool { return useF16KVCache }

func setKVF16(on bool) { useF16KVCache = on }

func cpuFeatureString() string {
	if hasQ8KDotAsm {
		return runtime.GOARCH + "+dotprod"
	}
	return runtime.GOARCH
}
