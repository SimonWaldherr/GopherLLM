package gopherllm

import "math"

// Owned by a decode workspace and cleared after the worker barrier. Avoid an
// escaping closure per recurrent layer without retaining model/request views.
type qwen35DeltaTask struct {
	headDim, keyDim, groups int
	state, q, k, v          []float32
	alpha, beta, a, dt      []float32
	out, recall             []float32
}

func (t *qwen35DeltaTask) runRows(start, end int) {
	n := t.headDim
	for h := start; h < end; h++ {
		alpha := nemotronSoftplus(t.alpha[h] + t.dt[h])
		decay := float32(math.Exp(float64(alpha * t.a[h])))
		beta := fastSigmoidF32(t.beta[h])
		group := h % t.groups
		q, k := t.q[group*t.keyDim:(group+1)*t.keyDim], t.k[group*t.keyDim:(group+1)*t.keyDim]
		qwen35DeltaHead(t.state[h*n*n:(h+1)*n*n], q, k, t.v[h*n:(h+1)*n],
			t.out[h*n:(h+1)*n], t.recall[h*n:(h+1)*n], decay, beta)
	}
}

// Keep the original per-row kernels as the fallback and numerical reference.
// Heads have independent matrices; updating a row cannot affect another row's
// recall. q, k and v must not alias the state matrix or output.
func qwen35DeltaHeadFallback(state, q, k, v, out, recall []float32, decay, beta float32) {
	n := len(v)
	for d := range n {
		row := state[d*n : (d+1)*n]
		ScaleF32(row, decay)
		recall[d] = DotF32(row, k)
	}
	for d := range n {
		row := state[d*n : (d+1)*n]
		delta := beta * (v[d] - recall[d])
		AxpyF32(row, delta, k)
		out[d] = DotF32(row, q)
	}
}
