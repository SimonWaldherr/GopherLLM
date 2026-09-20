//go:build darwin && cgo

package gopherllm

import (
	"context"
	"sync"

	"github.com/SimonWaldherr/GopherLLM/internal/voxtralblas"
)

type voxtralBLASScratch struct{ input, output, scores []float32 }

var voxtralBLASPool = sync.Pool{New: func() any { return &voxtralBLASScratch{} }}

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

func voxtralMatvecBatch(w Weight, xs, outs [][]float32) {
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
	b := voxtralBLASPool.Get().(*voxtralBLASScratch)
	defer voxtralBLASPool.Put(b)
	ensureLenNoClear(&b.input, len(xs)*cols)
	ensureLenNoClear(&b.output, len(xs)*rows)
	input, output := b.input, b.output
	for i, x := range xs {
		copy(input[i*cols:], x)
	}
	voxtralblas.Mul(len(xs), rows, cols, w.F32, input, output)
	for i := range outs {
		copy(outs[i], output[i*rows:(i+1)*rows])
	}
}

// stride is the per-head element stride within k/v; the caller may keep k/v
// allocated to a larger, stable per-head capacity across calls than the
// current valid position count (past+len(q)) so a persistent streaming
// cache can just write its new rows in, instead of repacking on every call.
func voxtralAttentionBatch(q [][]float32, k, v []float32, out [][]float32, heads, dim, past, window, stride int, scale float32) bool {
	n, total, width := len(q), past+len(q), heads*dim
	if n == 0 {
		return true
	}
	b := voxtralBLASPool.Get().(*voxtralBLASScratch)
	defer voxtralBLASPool.Put(b)
	ensureLenNoClear(&b.input, n*width)
	ensureLenNoClear(&b.output, n*width)
	ensureLenNoClear(&b.scores, n*total)
	queries, output := b.input, b.output
	for t := range q {
		copy(queries[t*width:], q[t])
	}
	voxtralblas.Attention(n, total, heads, dim, past, window, stride, scale, queries, k, v, b.scores, output)
	for t := range out {
		copy(out[t], output[t*width:(t+1)*width])
	}
	return true
}
