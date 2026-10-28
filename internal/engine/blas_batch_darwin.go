//go:build darwin && cgo

package engine

import (
	"context"
	"sync"

	"github.com/SimonWaldherr/GopherLLM/internal/voxtralblas"
)

type blasScratchPool struct{ input, output, scores []float32 }

var blasScratchBuf = sync.Pool{New: func() any { return &blasScratchPool{} }}

// Channel-major pointwise convolutions already have the layout W * X.
// Accelerate can consume it directly without an im2col or transpose buffer.
func blasGemmNN(m, n, k int, a, b, out []float32) bool {
	if m <= 0 || n <= 0 || k <= 0 || m > len(a)/k || k > len(b)/n || m > len(out)/n || batchSlicesOverlap(out, a) || batchSlicesOverlap(out, b) {
		return false
	}
	voxtralblas.GemmNN(m, n, k, a, b, out)
	return true
}

// Small audio batches benefit from Accelerate's matrix-matrix kernel. Expand
// the encoder once, rather than dequantizing every matrix for every chunk.
func prepareVoxtralStreamWeights(ctx context.Context, w *VoxtralRealtimeWeights) error {
	for i := range w.Encoder.Layers {
		l := &w.Encoder.Layers[i]
		for _, m := range []*Weight{&l.Q, &l.K, &l.V, &l.Out, &l.FFNGate, &l.FFNUp, &l.FFNDown} {
			if err := ctx.Err(); err != nil {
				return err
			}
			if m.F32 != nil || m.Rows <= 0 || m.Cols <= 0 {
				continue
			}
			data := make([]float32, m.Rows*m.Cols)
			var row []float32
			for r := range m.Rows {
				m.RowInto(r, m.Cols, &row)
				copy(data[r*m.Cols:], row)
			}
			m.F32 = data
		}
	}
	return nil
}

func blasMatvecBatch(w Weight, xs, outs [][]float32) {
	if len(xs) == 0 {
		return
	}
	cols := len(xs[0])
	rows := 0
	if cols > 0 {
		rows = len(w.F32) / cols
	}
	if rows == 0 || len(xs) < 2 {
		matvecBatch(w, xs, outs)
		return
	}
	b := blasScratchBuf.Get().(*blasScratchPool)
	defer blasScratchBuf.Put(b)
	input, directInput := contiguousBatchRows(xs, cols)
	output, directOutput := contiguousBatchRows(outs, rows)
	directOutput = directOutput && len(outs) == len(xs) && !batchSlicesOverlap(output, w.F32)
	if !directInput || directOutput && batchSlicesOverlap(input, output) {
		ensureLenNoClear(&b.input, len(xs)*cols)
		input = b.input
		for i, x := range xs {
			copy(input[i*cols:], x)
		}
	}
	if !directOutput {
		ensureLenNoClear(&b.output, len(xs)*rows)
		output = b.output
	}
	voxtralblas.Mul(len(xs), rows, cols, w.F32, input, output)
	if !directOutput {
		for i := range outs {
			copy(outs[i], output[i*rows:(i+1)*rows])
		}
	}
}

// stride is the per-head element stride within k/v; the caller may keep k/v
// allocated to a larger, stable per-head capacity across calls than the
// current valid position count (past+len(q)) so a persistent streaming
// cache can just write its new rows in, instead of repacking on every call.
func blasAttentionBatch(q [][]float32, k, v []float32, out [][]float32, heads, dim, past, window, stride int, scale float32) bool {
	n, total, width := len(q), past+len(q), heads*dim
	if n == 0 {
		return true
	}
	b := blasScratchBuf.Get().(*blasScratchPool)
	defer blasScratchBuf.Put(b)
	workers := voxtralblas.AttentionWorkers(n, total, heads, numThreads())
	ensureLenNoClear(&b.scores, workers*n*total)
	queries, directQueries := contiguousBatchRows(q, width)
	output, directOutput := contiguousBatchRows(out, width)
	directOutput = directOutput && len(out) == n && !batchSlicesOverlap(output, k) && !batchSlicesOverlap(output, v)
	if !directQueries || directOutput && batchSlicesOverlap(queries, output) {
		ensureLenNoClear(&b.input, n*width)
		queries = b.input
		for t := range q {
			copy(queries[t*width:], q[t])
		}
	}
	if !directOutput {
		ensureLenNoClear(&b.output, n*width)
		output = b.output
	}
	voxtralblas.AttentionWithWorkers(n, total, heads, dim, past, window, stride, scale, queries, k, v, b.scores, output, workers)
	if !directOutput {
		for t := range out {
			copy(out[t], output[t*width:(t+1)*width])
		}
	}
	return true
}
