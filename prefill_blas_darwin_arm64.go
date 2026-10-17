//go:build darwin && arm64 && cgo

package gopherllm

import (
	"sync"

	"github.com/SimonWaldherr/GopherLLM/internal/voxtralblas"
)

// Bound temporary weight expansion independently of model size. This path is
// also built on iOS: a full F32 copy of a quantized matrix would be too large
// for a phone, while a reusable row tile fits in at most four MiB.
const prefillBLASTileElements = 1 << 20
const prefillBLASInputElements = 2 << 20
const prefillBLASMatrixMinTokens = 32
const hasApplePrefillBLAS = true

type prefillBLASScratch struct {
	input, weights, output []float32
	dequant                prefillBLASDequantTask
}

var prefillBLASPool = sync.Pool{New: func() any { return new(prefillBLASScratch) }}

func prefillBLASRowTiles(rows, cols, tokens int, expandWeights bool) (compute, scatter int) {
	scatter = min(128, prefillBLASTileElements/cols, prefillBLASInputElements/tokens, rows)
	compute = scatter
	if !expandWeights && tokens >= prefillBLASMatrixMinTokens {
		// F32 weights already exist: only temporary output memory is bounded.
		compute = min(rows, prefillBLASInputElements/tokens)
		if compute < rows {
			compute -= compute % scatter
		}
	}
	return
}

type prefillBLASDequantTask struct {
	raw                     []byte
	dequant                 func([]byte, int, []float32)
	weights                 []float32
	cols, rowBytes, rowBase int
}

func (t *prefillBLASDequantTask) runRows(start, end int) {
	for row := start; row < end; row++ {
		offset := (t.rowBase + row) * t.rowBytes
		t.dequant(t.raw[offset:offset+t.rowBytes], t.cols, t.weights[row*t.cols:(row+1)*t.cols])
	}
}

// matvecBatchBLAS uses Accelerate's matrix kernel for F32 projections and the
// explicitly selected dequantize-once float path. Q8 activation arithmetic
// keeps its existing kernels and exact fused-projection contract. Small
// batches retain NEON to avoid packing and cgo overhead.
func matvecBatchBLAS(w Weight, xs, outs [][]float32, useQ8 bool) bool {
	p := len(xs)
	if p < 16 || len(outs) != p {
		return false
	}
	cols, rows := len(xs[0]), w.Rows
	if cols < 64 || cols > prefillBLASTileElements || p > prefillBLASInputElements/cols {
		return false
	}
	var dequant func([]byte, int, []float32)
	rowBytes := 0
	if w.F32 != nil {
		if len(w.F32)%cols != 0 {
			return false
		}
		rows = len(w.F32) / cols
	} else {
		// Dequantization plus tile dispatch loses to NEON on short batches.
		// Only long float batches amortize that work (e.g. Nomic embeddings).
		if useQ8 || p < 128 || rows <= 0 || w.Cols != cols {
			return false
		}
		dequant = dequantRowInto(w, cols)
		if dequant == nil {
			return false
		}
		var ok bool
		rowBytes, ok = w.Type.DataSize(cols)
		if !ok || rowBytes <= 0 || rows > len(w.Raw)/rowBytes {
			return false
		}
	}
	if rows < 32 || rows > 1<<31-1 || rows < (1<<18)/(p*cols) {
		return false
	}
	for i := range xs {
		if len(xs[i]) != cols || len(outs[i]) < rows || batchSlicesOverlap(outs[i], w.F32) {
			return false
		}
	}

	b := prefillBLASPool.Get().(*prefillBLASScratch)
	defer prefillBLASPool.Put(b)
	// Let Accelerate choose processor-specific matrix blocking on M4/SME as
	// well as earlier Apple silicon. Contiguous F32 batches need just one GEMM
	// and, when disjoint, no activation copies or additional output scratch.
	if dequant == nil && p >= prefillBLASMatrixMinTokens {
		input, directInput := contiguousBatchRows(xs, cols)
		output, directOutput := contiguousBatchRows(outs, rows)
		if directInput && directOutput && !batchSlicesOverlap(input, output) {
			voxtralblas.Mul(p, rows, cols, w.F32, input, output)
			return true
		}
	}
	// Snapshot inputs once, including the in-place case, before writing any
	// tile's outputs. Per-token views need not share a backing allocation.
	ensureLenNoClear(&b.input, p*cols)
	for i, x := range xs {
		copy(b.input[i*cols:(i+1)*cols], x)
	}
	// Larger F32 GEMMs avoid repeated packing/dispatch inside Accelerate while
	// retaining the eight-MiB bound on temporary output memory.
	tileRows, copyRows := prefillBLASRowTiles(rows, cols, p, dequant != nil)
	ensureLenNoClear(&b.output, p*tileRows)
	if dequant != nil {
		ensureLenNoClear(&b.weights, tileRows*cols)
		b.dequant = prefillBLASDequantTask{raw: w.Raw, dequant: dequant, weights: b.weights, cols: cols, rowBytes: rowBytes}
		defer func() { b.dequant = prefillBLASDequantTask{} }()
	}
	for start := 0; start < rows; start += tileRows {
		n := min(tileRows, rows-start)
		weights := w.F32
		if dequant != nil {
			b.dequant.rowBase = start
			parallelRowsBatchedTask(n, &b.dequant)
			weights = b.weights[:n*cols]
		} else {
			weights = weights[start*cols : (start+n)*cols]
		}
		voxtralblas.Mul(p, n, cols, weights, b.input, b.output[:p*n])
		// Retain the old tile-major scatter order, including when caller-owned
		// output views overlap one another. Inputs were already snapshotted.
		for offset := 0; offset < n; offset += copyRows {
			end := min(offset+copyRows, n)
			for token := range outs {
				copy(outs[token][start+offset:start+end], b.output[token*n+offset:token*n+end])
			}
		}
	}
	return true
}
