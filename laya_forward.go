package gopherllm

import (
	"context"
	"math"
	"sync"
	"unsafe"
)

// Tensor scratch and RoPE tables are reused across questions and requests.
// A model retains at most 32 MiB; unusually long contexts are released after
// their request. No token-dependent activations or K/V rows are reused as input.
type layaBuffer struct {
	data     []float32
	rows     [][]float32
	normTask layaNormTask
}

func (b *layaBuffer) resize(n, d int, zero bool) [][]float32 {
	return resizeMatrixRows(&b.data, &b.rows, n, d, zero)
}

type layaWorkspace struct {
	input, norm, qkv, attention, projection, hidden layaBuffer
	scores                                          []float32
	rotary                                          [2]layaRotary
	attentionTask                                   *layaAttentionTask
}

const layaWorkspaceRetentionBytes int64 = 32 << 20

// Predict holds the model mutex until this deferred cleanup completes. Every
// forward overwrites its active tensor rows; only allocation capacity and
// position-only RoPE tables survive. Worker score pools stay request-local.
func (m *LayaModel) retainWorkspace(work *layaWorkspace) {
	work.attentionTask = nil
	if work.retainedBytes() <= layaWorkspaceRetentionBytes {
		m.workspace = work
	} else {
		m.workspace = nil
	}
}

func (w *layaWorkspace) retainedBytes() int64 {
	bytes := int64(unsafe.Sizeof(*w))
	for _, b := range []*layaBuffer{&w.input, &w.norm, &w.qkv, &w.attention, &w.projection, &w.hidden} {
		bytes += int64(cap(b.data))*4 + int64(cap(b.rows))*int64(unsafe.Sizeof([]float32(nil)))
	}
	bytes += int64(cap(w.scores)) * 4
	for i := range w.rotary {
		bytes += int64(cap(w.rotary[i].data)) * 4
	}
	return bytes
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
	n.apply(x, out, &b.normTask)
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
	n.apply(x, x, nil)
}

type layaNormTask struct {
	norm    layaNorm
	in, out [][]float32
}

func (t *layaNormTask) runRows(start, end int) {
	for i := start; i < end; i++ {
		layerNormInto(t.in[i], t.norm.weight, t.norm.bias, t.norm.epsilon, &t.out[i])
	}
}

func (n layaNorm) apply(in, out [][]float32, task *layaNormTask) {
	// Keep small heads and short requests inline; larger token batches have
	// independent rows and preserve each row's original reduction order.
	if len(in) >= 32 && len(in[0]) >= 256 && numThreads() > 1 {
		if task == nil {
			task = &layaNormTask{}
		}
		*task = layaNormTask{norm: n, in: in, out: out}
		parallelChunksTask(len(in), task)
		*task = layaNormTask{} // Do not retain input views after the barrier.
		return
	}
	for i := range in {
		layerNormInto(in[i], n.weight, n.bias, n.epsilon, &out[i])
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
	softmaxF64InPlace(x, hi)
}

// A request-local pool reuses one score row per active worker, retaining the
// worker pool's fine-grained scheduling on CPUs with unequal core speeds.
type layaAttentionTask struct {
	ctx                               context.Context
	n, heads, dim, window, scoreWidth int
	scale                             float32
	qkv, out                          [][]float32
	scorePool                         sync.Pool
}

func layaAttentionPortable(ctx context.Context, n, heads, dim, window int, scale float32, work *layaWorkspace) error {
	if work.attentionTask == nil {
		work.attentionTask = &layaAttentionTask{}
	}
	task := work.attentionTask
	task.ctx, task.n, task.heads, task.dim, task.window, task.scale = ctx, n, heads, dim, window, scale
	task.qkv, task.out = work.qkv.rows, work.attention.rows
	task.scoreWidth = n
	if window >= 0 {
		task.scoreWidth = min(n, 2*min(window, n)+1)
	}
	parallelChunksTask(n*heads, task)
	task.ctx, task.qkv, task.out = nil, nil, nil
	return ctx.Err()
}

func (t *layaAttentionTask) runRows(start, end int) {
	slot := t.scorePool.Get()
	var scratch *[]float32
	if slot == nil {
		scratch = new([]float32)
	} else {
		scratch = slot.(*[]float32)
	}
	if cap(*scratch) < t.scoreWidth {
		*scratch = make([]float32, t.scoreWidth)
	} else {
		*scratch = (*scratch)[:t.scoreWidth]
	}
	defer t.scorePool.Put(scratch)
	scores := *scratch
	width := t.heads * t.dim
	for job := start; job < end; job++ {
		if t.ctx.Err() != nil {
			return
		}
		p, h := job/t.heads, job%t.heads
		lo, hi := 0, t.n
		if t.window >= 0 {
			lo = max(0, p-t.window)
			hi = min(t.n, p+t.window+1)
		}
		off := h * t.dim
		q := t.qkv[p][off : off+t.dim]
		for j := lo; j < hi; j++ {
			scores[j-lo] = DotF32(q, t.qkv[j][width+off:width+off+t.dim]) * t.scale
		}
		row := scores[:hi-lo]
		layaSoftmax(row)
		dest := t.out[p][off : off+t.dim]
		clear(dest)
		for j := lo; j < hi; j++ {
			AxpyF32(dest, row[j-lo], t.qkv[j][2*width+off:2*width+off+t.dim])
		}
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
	attended := work.attention.resize(n, d, false)
	scale := float32(1 / math.Sqrt(float64(hd)))
	if accelerated, err := layaAttentionAccelerated(ctx, n, l.heads, hd, l.window, scale, work); accelerated {
		if err != nil {
			return err
		}
	} else {
		if err := layaAttentionPortable(ctx, n, l.heads, hd, l.window, scale, work); err != nil {
			return err
		}
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
