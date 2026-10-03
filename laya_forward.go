package gopherllm

import (
	"context"
	"math"
)

// Layer workspaces are reused across the sequential questions in one request,
// then released. Long-context buffers are never retained across requests.
type layaBuffer struct {
	data []float32
	rows [][]float32
}

func (b *layaBuffer) resize(n, d int, zero bool) [][]float32 {
	if cap(b.data) < n*d {
		b.data = make([]float32, n*d)
	} else {
		b.data = b.data[:n*d]
		if zero {
			clear(b.data)
		}
	}
	if cap(b.rows) < n {
		b.rows = make([][]float32, n)
	} else {
		b.rows = b.rows[:n]
	}
	for i := range b.rows {
		b.rows[i] = b.data[i*d : (i+1)*d]
	}
	return b.rows
}

type layaWorkspace struct {
	input, norm, qkv, attention, projection, hidden layaBuffer
	scores                                          []float32
	rotary                                          [2]layaRotary
}

type layaRotary struct {
	theta float64
	dim   int
	data  []float32
}

// ModernBERT has global and local RoPE parameters. Cache both sets of rotations
// in the request workspace; positions are shared across heads and layers. Each
// row stores cosines followed by sines, retaining the original float32 rounding.
func (w *layaWorkspace) rotations(n, dim int, theta float64) []float32 {
	table := &w.rotary[0]
	for i := range w.rotary {
		t := &w.rotary[i]
		if t.dim == dim && t.theta == theta {
			table = t
			break
		}
		if t.dim == 0 {
			table = t
		}
	}
	if table.dim != dim || table.theta != theta {
		*table = layaRotary{theta: theta, dim: dim}
	}
	previous := len(table.data) / dim
	if previous < n {
		table.data = append(table.data, make([]float32, (n-previous)*dim)...)
		for j := 0; j < dim/2; j++ {
			frequency := math.Pow(theta, -float64(2*j)/float64(dim))
			for p := previous; p < n; p++ {
				angle := float64(p) * frequency
				table.data[p*dim+j] = float32(math.Cos(angle))
				table.data[p*dim+j+dim/2] = float32(math.Sin(angle))
			}
		}
	}
	return table.data[:n*dim]
}

func layaRotate(qkv [][]float32, heads int, theta float64, work *layaWorkspace) {
	d := len(qkv[0]) / 3
	hd := d / heads
	half := hd / 2
	rotations := work.rotations(len(qkv), hd, theta)
	for p, row := range qkv {
		cosines := rotations[p*hd : p*hd+half]
		sines := rotations[p*hd+half : (p+1)*hd]
		for h := 0; h < heads; h++ {
			q, k := row[h*hd:(h+1)*hd], row[d+h*hd:d+(h+1)*hd]
			for j, c := range cosines {
				s := sines[j]
				a, b := q[j], q[j+half]
				q[j] = a*c - b*s
				q[j+half] = b*c + a*s
				a, b = k[j], k[j+half]
				k[j] = a*c - b*s
				k[j+half] = b*c + a*s
			}
		}
	}
}

func (n layaNorm) into(x [][]float32, b *layaBuffer) [][]float32 {
	if len(n.weight) == 0 {
		return x
	}
	out := b.resize(len(x), len(x[0]), false)
	for i := range x {
		layerNormInto(x[i], n.weight, n.bias, n.epsilon, &out[i])
	}
	return out
}
func (l layaLinear) into(x [][]float32, b *layaBuffer, stride int) [][]float32 {
	out := b.resize(len(x), l.weight.Rows, false)
	if !layaProjectAccelerated(l.weight, x, b, stride) {
		blasMatvecBatch(l.weight, x, out)
	}
	if l.bias != nil {
		for _, r := range out {
			for j, v := range l.bias {
				r[j] += v
			}
		}
	}
	return out
}
func (n layaNorm) inPlace(x [][]float32) {
	if len(n.weight) == 0 {
		return
	}
	// LayerNorm reads the complete row for mean/variance before writing values.
	for i := range x {
		layerNormInto(x[i], n.weight, n.bias, n.epsilon, &x[i])
	}
}
func layaResidual(x, y [][]float32) {
	for i := range x {
		for j := range x[i] {
			x[i][j] += y[i][j]
		}
	}
}
func layaSoftmax(x []float32) {
	hi := x[0]
	for _, v := range x[1:] {
		hi = max(hi, v)
	}
	sum := float64(0)
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - hi)))
		sum += float64(x[i])
	}
	for i := range x {
		x[i] /= float32(sum)
	}
}
func (l layaLayer) forward(ctx context.Context, x [][]float32, work *layaWorkspace) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	n, d := len(x), len(x[0])
	hd := d / l.heads
	qkv := l.qkv.into(l.norm1.into(x, &work.norm), &work.qkv, d)
	if l.theta > 0 {
		layaRotate(qkv, l.heads, l.theta, work)
	}
	attended := work.attention.resize(n, d, true)
	scale := float32(1 / math.Sqrt(float64(hd)))
	if accelerated, err := layaAttentionAccelerated(ctx, n, l.heads, hd, l.window, scale, work); accelerated {
		if err != nil {
			return err
		}
	} else {
		// Only one score row per worker: memory is O(sequence*hidden), not O(n²*heads).
		parallelChunks(n*l.heads, func(start, end int) {
			scores := make([]float32, n)
			for job := start; job < end; job++ {
				if ctx.Err() != nil {
					return
				}
				p, h := job/l.heads, job%l.heads
				lo, hi := 0, n
				if l.window >= 0 {
					lo = max(0, p-l.window)
					hi = min(n, p+l.window+1)
				}
				q := qkv[p][h*hd : (h+1)*hd]
				for j := lo; j < hi; j++ {
					scores[j-lo] = DotF32(q, qkv[j][d+h*hd:d+(h+1)*hd]) * scale
				}
				row := scores[:hi-lo]
				layaSoftmax(row)
				dest := attended[p][h*hd : (h+1)*hd]
				for j := lo; j < hi; j++ {
					v := qkv[j][2*d+h*hd : 2*d+(h+1)*hd]
					AxpyF32(dest, row[j-lo], v)
				}
			}
		})
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	layaResidual(x, l.output.into(attended, &work.projection, d))
	hidden := l.up.into(l.norm2.into(x, &work.norm), &work.hidden, d)
	activate := func(start, end int) {
		for i := start; i < end; i++ {
			row := hidden[i]
			if l.gated {
				half := len(row) / 2
				for j := 0; j < half; j++ {
					row[j] = geluExact(row[j]) * row[j+half]
				}
				hidden[i] = row[:half]
			} else {
				// PyTorch TransformerEncoderLayer defaults to ReLU.
				for j := range row {
					row[j] = max(0, row[j])
				}
			}
		}
	}
	if n >= 8 && len(hidden[0]) >= 256 {
		parallelChunks(n, activate)
	} else {
		activate(0, n)
	}
	layaResidual(x, l.down.into(hidden, &work.projection, l.up.weight.Rows))
	return ctx.Err()
}
func (m *LayaModel) forward(ctx context.Context, ids []uint32, markers []int, kind int, work *layaWorkspace) ([]float32, float32, error) {
	d := m.enc.Dim
	x := work.input.resize(len(ids), d, false)
	for i, id := range ids {
		m.embedding.RowInto(int(id), d, &x[i])
	}
	m.embNorm.inPlace(x)
	for _, l := range m.layers {
		if err := l.forward(ctx, x, work); err != nil {
			return nil, 0, err
		}
	}
	m.finalNorm.inPlace(x)
	for _, r := range x {
		for j := range r {
			r[j] += m.typeEmb[kind*d+j]
		}
	}
	for _, l := range m.head {
		if err := l.forward(ctx, x, work); err != nil {
			return nil, 0, err
		}
	}
	selected := make([][]float32, len(markers))
	for i, p := range markers {
		selected[i] = x[p]
	}
	hidden := m.score1.into(m.scoreNorm.into(selected, &work.norm), &work.hidden, d)
	for _, r := range hidden {
		for i := range r {
			r[i] = geluExact(r[i])
		}
	}
	scores := m.score2.into(hidden, &work.projection, m.score1.weight.Rows)
	logits := make([]float32, len(scores))
	for i := range scores {
		logits[i] = scores[i][0]
	}
	p := append([]float32(nil), logits...)
	layaSoftmax(p)
	top, second, entropy := float32(0), float32(0), float32(0)
	for _, v := range p {
		if v > top {
			second = top
			top = v
		} else if v > second {
			second = v
		}
		entropy -= v * float32(math.Log(float64(max(v, 1e-9))))
	}
	features := append(append([]float32(nil), x[0]...), top, top-second, entropy/float32(math.Log(float64(max(2, len(p))))), float32(max(2, len(p)))/255)
	act := m.act1.into([][]float32{features}, &work.hidden, len(features))
	for i := range act[0] {
		act[0][i] = geluExact(act[0][i])
	}
	act = m.act2.into(act, &work.projection, m.act1.weight.Rows)
	layaSoftmax(act[0])
	return logits, act[0][0], ctx.Err()
}
