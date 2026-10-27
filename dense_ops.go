package gopherllm

import "context"

// denseGemmScratch allocates worker state only for the portable path. Keeping
// the task behind a pointer lets short-lived Accelerate callers stay on stack.
type denseGemmScratch struct{ task *denseGemmTask }

type denseGemmTask struct {
	ctx       context.Context
	n, k      int
	a, b, out []float32
}

// multiply consumes contiguous row-major operands and disjoint output. Callers
// validate shapes and own the scratch until the worker barrier has returned.
func (s *denseGemmScratch) multiply(ctx context.Context, m, n, k int, a, b, out []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if blasGemmNN(m, n, k, a, b, out) {
		return ctx.Err()
	}
	if s.task == nil {
		s.task = new(denseGemmTask)
	}
	t := s.task
	*t = denseGemmTask{ctx: ctx, n: n, k: k, a: a, b: b, out: out}
	parallelRowsTask(m, t)
	*t = denseGemmTask{}
	return ctx.Err()
}

func (t *denseGemmTask) runRows(start, end int) {
	for row := start; row < end; row++ {
		if t.ctx.Err() != nil {
			return
		}
		// Keep a destination tile hot while accumulating in the original k
		// order. TTS tiles already fit; spatial attention can have wider rows.
		for j := 0; j < t.n; j += 256 {
			width := min(256, t.n-j)
			dst := t.out[row*t.n+j : row*t.n+j+width]
			clear(dst)
			for p := range t.k {
				AxpyF32(dst, t.a[row*t.k+p], t.b[p*t.n+j:p*t.n+j+width])
			}
		}
	}
}

// copyPaddedWindow copies src[start:start+len(dst)], filling out-of-range
// positions with zero. Input and output must not overlap. Convolution patch
// builders use this for contiguous rows instead of checking every element.
func copyPaddedWindow(dst, src []float32, start int) {
	if start >= len(src) || start <= -len(dst) {
		clear(dst)
		return
	}
	if start < 0 {
		clear(dst[:-start])
		dst = dst[-start:]
		start = 0
	}
	n := copy(dst, src[start:])
	clear(dst[n:])
}

// Copy a window from [src[0], src[0], src[1], src[1], ...], with zero padding.
// Only the requested patch is materialized. Inputs must not overlap.
func copyPaddedRepeat2(dst, src []float32, start int) {
	if start >= 2*len(src) || start <= -len(dst) {
		clear(dst)
		return
	}
	if start < 0 {
		clear(dst[:-start])
		dst = dst[-start:]
		start = 0
	}
	n := min(len(dst), 2*len(src)-start)
	clear(dst[n:])
	dst = dst[:n]
	if start&1 != 0 && len(dst) > 0 {
		dst[0] = src[start/2]
		dst = dst[1:]
		start++
	}
	src = src[start/2:]
	for i := 0; i < len(dst)/2; i++ {
		dst[2*i], dst[2*i+1] = src[i], src[i]
	}
	if len(dst)&1 != 0 {
		dst[len(dst)-1] = src[len(dst)/2]
	}
}
