package engine

import (
	"math"
	"sync"
)

const littleBitCPUBatchTile = 32

// littleBitBatchTask keeps factor rows stationary across a tile of prompt
// tokens. A worker reads each packed row and its scales before computing all
// token results for that row. Each factor needs only one worker dispatch.
type littleBitBatchTask struct {
	scaledInput, latent       []float32
	inputs, latentRows        [littleBitCPUBatchTile][]float32
	xs, outs                  [][]float32
	packed                    []byte
	scales, latentScale, bias LittleBitScale
	rowBytes                  int
	add                       bool
}

var littleBitBatchPool = sync.Pool{New: func() any { return new(littleBitBatchTask) }}

func (t *littleBitBatchTask) runRows(start, end int) {
	for row := start; row < end; row++ {
		packed := t.packed[row*t.rowBytes : (row+1)*t.rowBytes]
		scale := t.scales.at(row)
		if len(t.latentScale.Raw) != 0 {
			scale *= t.latentScale.at(row)
		}
		hasBias := len(t.bias.Raw) != 0
		var bias float32
		if hasBias {
			bias = t.bias.at(row)
		}
		i := 0
		if hasFastLittleBitDot4 {
			for ; i+4 <= len(t.xs); i += 4 {
				a, b, c, d := littleBitDot4(packed, t.xs[i], t.xs[i+1], t.xs[i+2], t.xs[i+3])
				storeLittleBitBatchValue(t.outs[i], row, a*scale, t.add, hasBias, bias)
				storeLittleBitBatchValue(t.outs[i+1], row, b*scale, t.add, hasBias, bias)
				storeLittleBitBatchValue(t.outs[i+2], row, c*scale, t.add, hasBias, bias)
				storeLittleBitBatchValue(t.outs[i+3], row, d*scale, t.add, hasBias, bias)
			}
		}
		for ; i < len(t.xs); i++ {
			storeLittleBitBatchValue(t.outs[i], row, littleBitDot(packed, t.xs[i])*scale, t.add, hasBias, bias)
		}
	}
}

func storeLittleBitBatchValue(out []float32, row int, value float32, add, hasBias bool, bias float32) {
	if add {
		value = out[row] + value
	}
	if hasBias {
		value += bias
	}
	out[row] = value
}

func (t *littleBitBatchTask) project(xs, outs [][]float32, rows int, packed []byte, scales, latentScale, bias LittleBitScale, add bool) {
	t.xs, t.outs, t.packed = xs, outs, packed
	t.scales, t.latentScale, t.bias, t.add = scales, latentScale, bias, add
	t.rowBytes = littleBitPackedRowBytes(len(xs[0]))
	if rows >= 32 && len(xs[0]) >= 65536/len(xs)/rows {
		parallelRowsBatchedTask(rows, t)
	} else {
		t.runRows(0, rows)
	}
}

func (t *littleBitBatchTask) factors(f *LittleBitFactors, xs, outs [][]float32, rows, cols int, bias LittleBitScale, add bool) {
	ensureLenNoClear(&t.scaledInput, len(xs)*cols)
	ensureLenNoClear(&t.latent, len(xs)*f.Rank)
	input, latent := t.inputs[:len(xs)], t.latentRows[:len(xs)]
	for i := range xs {
		input[i] = t.scaledInput[i*cols : (i+1)*cols]
		latent[i] = t.latent[i*f.Rank : (i+1)*f.Rank]
	}
	for col := range cols {
		scale := f.V2.at(col)
		for i := range xs {
			input[i][col] = xs[i][col] * scale
		}
	}
	t.project(input, latent, f.Rank, f.V, f.V1, f.U2, LittleBitScale{}, false)
	t.project(latent, outs, rows, f.U, f.U1, LittleBitScale{}, bias, add)
}

func (w *LittleBitWeight) matvecBatchCPU(xs, outs [][]float32, rows, cols int) bool {
	if len(xs) < 2 || rows <= 0 || cols <= 0 || len(outs) < len(xs) {
		return false
	}
	maxWidth := math.MaxInt / (4 * min(len(xs), littleBitCPUBatchTile)) // float32 scratch bytes
	if cols > maxWidth || w.Main.Rank > maxWidth || (w.Residual != nil && w.Residual.Rank > maxWidth) {
		return false
	}
	for i := range xs {
		if len(xs[i]) != cols {
			return false
		}
	}
	for i := range xs {
		ensureLenNoClear(&outs[i], rows)
	}
	t := littleBitBatchPool.Get().(*littleBitBatchTask)
	for start := 0; start < len(xs); start += littleBitCPUBatchTile {
		end := min(start+littleBitCPUBatchTile, len(xs))
		if w.Residual == nil {
			t.factors(&w.Main, xs[start:end], outs[start:end], rows, cols, w.Bias, false)
		} else {
			t.factors(&w.Main, xs[start:end], outs[start:end], rows, cols, LittleBitScale{}, false)
			t.factors(w.Residual, xs[start:end], outs[start:end], rows, cols, w.Bias, true)
		}
	}
	// Clear all references into the borrowed factors and caller-owned rows.
	*t = littleBitBatchTask{scaledInput: t.scaledInput[:0], latent: t.latent[:0]}
	littleBitBatchPool.Put(t)
	return true
}
