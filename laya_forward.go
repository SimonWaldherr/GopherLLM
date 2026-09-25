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
func (l layaLayer) forward(ctx context.Context, x [][]float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	n, d := len(x), len(x[0])
	hd := d / l.heads
	qkv := l.qkv.apply(l.norm1.apply(x))
	if l.theta > 0 {
		for p, row := range qkv {
			for h := 0; h < l.heads; h++ {
				for j := 0; j < hd/2; j++ {
					angle := float64(p) * math.Pow(l.theta, -float64(2*j)/float64(hd))
					s, c := float32(math.Sin(angle)), float32(math.Cos(angle))
					for _, base := range []int{h * hd, d + h*hd} {
						a, b := row[base+j], row[base+j+hd/2]
						row[base+j] = a*c - b*s
						row[base+j+hd/2] = b*c + a*s
					}
				}
			}
		}
	}
	attended := layaRows(n, d)
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
				for k := range dest {
					dest[k] += row[j-lo] * v[k]
				}
			}
		}
	})
	if err := ctx.Err(); err != nil {
		return err
	}
	layaResidual(x, l.output.apply(attended))
	hidden := l.up.apply(l.norm2.apply(x))
	if l.gated {
		for i, row := range hidden {
			half := len(row) / 2
			for j := 0; j < half; j++ {
				row[j] = geluExact(row[j]) * row[j+half]
			}
			hidden[i] = row[:half]
		}
	} else {
		for _, row := range hidden {
			for j := range row {
				row[j] = max(0, row[j])
			}
		}
	} // PyTorch TransformerEncoderLayer defaults to ReLU.
	layaResidual(x, l.down.apply(hidden))
	return ctx.Err()
}
func (m *LayaModel) forward(ctx context.Context, ids []uint32, markers []int, kind int) ([]float32, float32, error) {
	d := m.enc.Dim
	x := layaRows(len(ids), d)
	for i, id := range ids {
		m.embedding.RowInto(int(id), d, &x[i])
	}
	x = m.embNorm.apply(x)
	for _, l := range m.layers {
		if err := l.forward(ctx, x); err != nil {
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
		if err := l.forward(ctx, x); err != nil {
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
