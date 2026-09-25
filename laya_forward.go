package gopherllm

import (
	"context"
	"math"
)

func layaRows(n, d int) [][]float32 {
	flat := make([]float32, n*d)
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = flat[i*d : (i+1)*d]
	}
	return rows
}

// Layer workspaces are reused within one forward pass, then released. They are
// never shared across requests or retained after a long-context request.
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

type layaWorkspace struct{ norm, qkv, attention, projection, hidden layaBuffer }

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
func (l layaLinear) into(x [][]float32, b *layaBuffer) [][]float32 {
	out := b.resize(len(x), l.weight.Rows, false)
	blasMatvecBatch(l.weight, x, out)
	if l.bias != nil {
		for _, r := range out {
			for j, v := range l.bias {
				r[j] += v
			}
		}
	}
	return out
}
func (n layaNorm) apply(x [][]float32) [][]float32 {
	if len(n.weight) == 0 {
		return x
	}
	out := layaRows(len(x), len(x[0]))
	for i := range x {
		layerNormInto(x[i], n.weight, n.bias, n.epsilon, &out[i])
	}
	return out
}
func (l layaLinear) apply(x [][]float32) [][]float32 {
	out := layaRows(len(x), l.weight.Rows)
	blasMatvecBatch(l.weight, x, out)
	if l.bias != nil {
		for _, r := range out {
			for j, b := range l.bias {
				r[j] += b
			}
		}
	}
	return out
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
	qkv := l.qkv.into(l.norm1.into(x, &work.norm), &work.qkv)
	if l.theta > 0 {
		// Frequencies depend on the dimension; rotations on position, not head.
		for j := 0; j < hd/2; j++ {
			frequency := math.Pow(l.theta, -float64(2*j)/float64(hd))
			for p, row := range qkv {
				angle := float64(p) * frequency
				s, c := float32(math.Sin(angle)), float32(math.Cos(angle))
				for h := 0; h < l.heads; h++ {
					for _, base := range []int{h * hd, d + h*hd} {
						a, b := row[base+j], row[base+j+hd/2]
						row[base+j] = a*c - b*s
						row[base+j+hd/2] = b*c + a*s
					}
				}
			}
		}
	}
	attended := work.attention.resize(n, d, true)
	scale := float32(1 / math.Sqrt(float64(hd)))
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
	if err := ctx.Err(); err != nil {
		return err
	}
	layaResidual(x, l.output.into(attended, &work.projection))
	hidden := l.up.into(l.norm2.into(x, &work.norm), &work.hidden)
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
	layaResidual(x, l.down.into(hidden, &work.projection))
	return ctx.Err()
}
func (m *LayaModel) forward(ctx context.Context, ids []uint32, markers []int, kind int) ([]float32, float32, error) {
	d := m.enc.Dim
	x := layaRows(len(ids), d)
	for i, id := range ids {
		m.embedding.RowInto(int(id), d, &x[i])
	}
	var work layaWorkspace
	x = m.embNorm.apply(x)
	for _, l := range m.layers {
		if err := l.forward(ctx, x, &work); err != nil {
			return nil, 0, err
		}
	}
	x = m.finalNorm.apply(x)
	for _, r := range x {
		for j := range r {
			r[j] += m.typeEmb[kind*d+j]
		}
	}
	for _, l := range m.head {
		if err := l.forward(ctx, x, &work); err != nil {
			return nil, 0, err
		}
	}
	selected := make([][]float32, len(markers))
	for i, p := range markers {
		selected[i] = x[p]
	}
	hidden := m.score1.apply(m.scoreNorm.apply(selected))
	for _, r := range hidden {
		for i := range r {
			r[i] = geluExact(r[i])
		}
	}
	scores := m.score2.apply(hidden)
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
	act := m.act1.apply([][]float32{features})
	for i := range act[0] {
		act[0][i] = geluExact(act[0][i])
	}
	act = m.act2.apply(act)
	layaSoftmax(act[0])
	return logits, act[0][0], ctx.Err()
}
