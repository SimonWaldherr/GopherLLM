package engine

import (
	"context"
	"errors"
	"math"
)

// The supported graph follows UNet2DModel and ResnetBlock2D in Diffusers.
// The original checkpoint uses the pre-0.17 query/key/value/proj_attn names.
// All feature maps are one CHW sample; batches and text conditioning are absent.
type diffusionTensor struct {
	data    []float32
	c, h, w int
}

type diffusionConv struct {
	in, out, kernel int
	weight, bias    []float32
}

type diffusionNorm struct{ weight, bias []float32 }

type diffusionResnet struct {
	norm1, norm2 diffusionNorm
	conv1, conv2 diffusionConv
	time         diffusionConv
	shortcut     *diffusionConv
}

type diffusionAttention struct {
	norm                   diffusionNorm
	query, key, value, out diffusionConv
}

type diffusionBlock struct {
	resnets    []diffusionResnet
	attentions []*diffusionAttention
	resample   *diffusionConv
}

type diffusionUNet struct {
	input, output diffusionConv
	time1, time2  diffusionConv
	outputNorm    diffusionNorm
	down, up      [4]diffusionBlock
	mid1, mid2    diffusionResnet
	midAttention  *diffusionAttention
}

// A generation owns its buffers. The network uses the same allocation sequence
// on every step, so slots can be reused without a global pool retaining memory
// after generation. im2col is one shared scratch buffer, not one per convolution.
type diffusionWorkspace struct {
	ctx     context.Context
	buffers [][]float32
	next    int
	columns []float32
	err     error
	norm    diffusionNormTask
	matmul  denseGemmScratch
}

func (s *diffusionWorkspace) alloc(n int) []float32 {
	if s.next == len(s.buffers) {
		s.buffers = append(s.buffers, make([]float32, n))
	} else if cap(s.buffers[s.next]) < n {
		s.buffers[s.next] = make([]float32, n)
	}
	out := s.buffers[s.next][:n]
	s.next++
	return out
}

func (s *diffusionWorkspace) canceled() bool {
	if s.err == nil {
		s.err = s.ctx.Err()
	}
	return s.err != nil
}

// GEMM consumes row-major operands and has no dependency on the text-model
// runtime. The existing Accelerate path is used on macOS; other builds retain
// the native Go path. Cancellation boundaries cover the portable tile loops.
func (s *diffusionWorkspace) gemm(m, n, k int, a, b, out []float32) {
	if s.canceled() {
		return
	}
	s.err = s.matmul.multiply(s.ctx, m, n, k, a, b, out)
}

func (c diffusionConv) forward(s *diffusionWorkspace, x diffusionTensor, stride, padding int) diffusionTensor {
	if s.canceled() {
		return x
	}
	h, w := x.h, x.w
	if stride == 2 {
		// This checkpoint pads only right and bottom before its stride-2
		// convolutions (downsample_padding=0), rather than symmetric padding.
		h, w = x.h/2, x.w/2
	}
	n := h * w
	y := diffusionTensor{s.alloc(c.out * n), c.out, h, w}
	input := x.data
	if c.kernel != 1 {
		needed := c.in * c.kernel * c.kernel * n
		if cap(s.columns) < needed {
			s.columns = make([]float32, needed)
		}
		input = s.columns[:needed]
		for ch := range c.in {
			for ky := range c.kernel {
				for kx := range c.kernel {
					row := input[((ch*c.kernel+ky)*c.kernel+kx)*n:][:n]
					for oy := range h {
						iy := oy*stride + ky - padding
						dst := row[oy*w : (oy+1)*w]
						if iy < 0 || iy >= x.h {
							clear(dst)
							continue
						}
						src := x.data[(ch*x.h+iy)*x.w : (ch*x.h+iy+1)*x.w]
						if stride == 1 {
							copyPaddedWindow(dst, src, kx-padding)
							continue
						}
						for ox := range w {
							ix := ox*stride + kx - padding
							if ix >= 0 && ix < x.w {
								dst[ox] = src[ix]
							} else {
								dst[ox] = 0
							}
						}
					}
				}
			}
		}
	}
	s.gemm(c.out, n, c.in*c.kernel*c.kernel, c.weight, input, y.data)
	for ch, bias := range c.bias {
		for i := range n {
			y.data[ch*n+i] += bias
		}
	}
	return y
}

type diffusionNormTask struct {
	norm     diffusionNorm
	dst, src []float32
	spatial  int
	activate bool
}

func (n diffusionNorm) apply(s *diffusionWorkspace, dst, src []float32, spatial int, activate bool) {
	// Groups are independent, including when normalizing in place. Keep each
	// group's reduction order unchanged and reuse the task across denoising steps.
	s.norm = diffusionNormTask{n, dst, src, spatial, activate}
	if len(src) >= 32768 {
		parallelChunksTask(32, &s.norm)
	} else {
		s.norm.runRows(0, 32)
	}
	s.norm = diffusionNormTask{}
}

func (t *diffusionNormTask) runRows(start, end int) {
	groupSize := len(t.norm.weight) / 32 * t.spatial
	for group := start; group < end; group++ {
		values := t.src[group*groupSize : (group+1)*groupSize]
		mean := float64(0)
		for _, value := range values {
			mean += float64(value)
		}
		mean /= float64(groupSize)
		variance := float64(0)
		for _, value := range values {
			diff := float64(value) - mean
			variance += diff * diff
		}
		invStd := float32(1 / math.Sqrt(variance/float64(groupSize)+1e-6))
		for i, value := range values {
			index := group*groupSize + i
			channel := index / t.spatial
			y := (value-float32(mean))*invStd*t.norm.weight[channel] + t.norm.bias[channel]
			if t.activate {
				y = siluExact(y)
			}
			t.dst[index] = y
		}
	}
}

func (r diffusionResnet) forward(s *diffusionWorkspace, x diffusionTensor, time diffusionTensor) diffusionTensor {
	if s.canceled() {
		return x
	}
	normalized := diffusionTensor{s.alloc(len(x.data)), x.c, x.h, x.w}
	r.norm1.apply(s, normalized.data, x.data, x.h*x.w, true)
	y := r.conv1.forward(s, normalized, 1, 1)
	if s.canceled() {
		return x
	}
	projected := r.time.forward(s, time, 1, 0)
	n := y.h * y.w
	for ch := range y.c {
		for i := range n {
			y.data[ch*n+i] += projected.data[ch]
		}
	}
	r.norm2.apply(s, y.data, y.data, n, true)
	y = r.conv2.forward(s, y, 1, 1)
	if s.canceled() {
		return x
	}
	if r.shortcut != nil {
		x = r.shortcut.forward(s, x, 1, 0)
	}
	if s.canceled() {
		return x
	}
	for i, value := range x.data {
		y.data[i] += value
	}
	return y
}

func (a *diffusionAttention) forward(s *diffusionWorkspace, x diffusionTensor) diffusionTensor {
	if s.canceled() {
		return x
	}
	n := x.h * x.w
	normalized := diffusionTensor{s.alloc(len(x.data)), x.c, x.h, x.w}
	a.norm.apply(s, normalized.data, x.data, n, false)
	q := a.query.forward(s, normalized, 1, 0)
	k := a.key.forward(s, normalized, 1, 0)
	v := a.value.forward(s, normalized, 1, 0)
	if s.canceled() {
		return x
	}
	queries := s.alloc(n * x.c)
	for ch := range x.c {
		for p := range n {
			queries[p*x.c+ch] = q.data[ch*n+p]
		}
	}
	scores := s.alloc(n * n)
	s.gemm(n, n, x.c, queries, k.data, scores)
	if s.canceled() {
		return x
	}
	scale := float32(1 / math.Sqrt(float64(x.c)))
	for p := range n {
		row := scores[p*n : (p+1)*n]
		maximum := float32(-math.MaxFloat32)
		for i := range row {
			row[i] *= scale
			maximum = max(maximum, row[i])
		}
		softmaxF64InPlace(row, maximum)
	}
	transposed := s.alloc(n * n)
	for i := range n {
		for j := range n {
			transposed[j*n+i] = scores[i*n+j]
		}
	}
	mixed := diffusionTensor{s.alloc(len(x.data)), x.c, x.h, x.w}
	s.gemm(x.c, n, n, v.data, transposed, mixed.data)
	y := a.out.forward(s, mixed, 1, 0)
	if s.canceled() {
		return x
	}
	for i, value := range x.data {
		y.data[i] += value
	}
	return y
}

func diffusionUpsample(s *diffusionWorkspace, x diffusionTensor) diffusionTensor {
	y := diffusionTensor{s.alloc(len(x.data) * 4), x.c, x.h * 2, x.w * 2}
	for ch := range x.c {
		for iy := range y.h {
			for ix := range y.w {
				y.data[(ch*y.h+iy)*y.w+ix] = x.data[(ch*x.h+iy/2)*x.w+ix/2]
			}
		}
	}
	return y
}

func (n *diffusionUNet) forward(s *diffusionWorkspace, sample []float32, timestep int) ([]float32, error) {
	if len(sample) != 3*32*32 || timestep < 0 || timestep >= 1000 {
		return nil, errors.New("diffusion: invalid UNet input")
	}
	s.next = 0
	if s.canceled() {
		return nil, s.err
	}
	time := diffusionTensor{s.alloc(128), 128, 1, 1}
	for i := range 64 {
		// Match the training convention: sine first, frequency shift 1.
		exponent := float32(-math.Log(10000)) * float32(i) / 63
		angle := float32(timestep) * float32(math.Exp(float64(exponent)))
		time.data[i] = float32(math.Sin(float64(angle)))
		time.data[i+64] = float32(math.Cos(float64(angle)))
	}
	time = n.time1.forward(s, time, 1, 0)
	for i := range time.data {
		time.data[i] = siluExact(time.data[i])
	}
	time = n.time2.forward(s, time, 1, 0)
	// Every residual block applies SiLU before its time projection. All use
	// the same embedding, so evaluate that activation once for this step.
	for i := range time.data {
		time.data[i] = siluExact(time.data[i])
	}
	x := n.input.forward(s, diffusionTensor{sample, 3, 32, 32}, 1, 1)
	if s.canceled() {
		return nil, s.err
	}
	var skips [12]diffusionTensor
	skips[0] = x
	skipCount := 1
	for _, block := range n.down {
		for i, resnet := range block.resnets {
			x = resnet.forward(s, x, time)
			if len(block.attentions) != 0 {
				x = block.attentions[i].forward(s, x)
			}
			if s.canceled() {
				return nil, s.err
			}
			skips[skipCount] = x
			skipCount++
		}
		if block.resample != nil {
			x = block.resample.forward(s, x, 2, 0)
			if s.canceled() {
				return nil, s.err
			}
			skips[skipCount] = x
			skipCount++
		}
	}
	x = n.mid1.forward(s, x, time)
	x = n.midAttention.forward(s, x)
	x = n.mid2.forward(s, x, time)
	for _, block := range n.up {
		for i, resnet := range block.resnets {
			if s.canceled() {
				return nil, s.err
			}
			skipCount--
			skip := skips[skipCount]
			combined := diffusionTensor{s.alloc(len(x.data) + len(skip.data)), x.c + skip.c, x.h, x.w}
			copy(combined.data, x.data)
			copy(combined.data[len(x.data):], skip.data)
			x = resnet.forward(s, combined, time)
			if len(block.attentions) != 0 {
				x = block.attentions[i].forward(s, x)
			}
		}
		if block.resample != nil {
			if s.canceled() {
				return nil, s.err
			}
			x = block.resample.forward(s, diffusionUpsample(s, x), 1, 1)
		}
	}
	if s.canceled() {
		return nil, s.err
	}
	n.outputNorm.apply(s, x.data, x.data, x.h*x.w, true)
	x = n.output.forward(s, x, 1, 1)
	if s.canceled() {
		return nil, s.err
	}
	return x.data, nil
}

// DDIM eta=0 uses the pretrained DDPM linear beta schedule. Leading timestep
// spacing, epsilon prediction, clipping and final alpha=1 match Diffusers'
// DDIMScheduler.from_config(original_scheduler.config) defaults.
func diffusionAlphas() [1000]float32 {
	var alphas [1000]float32
	product := float64(1)
	for i := range alphas {
		beta := float32(0.0001 + (0.02-0.0001)*float64(i)/999)
		product *= float64(1 - beta)
		alphas[i] = float32(product)
	}
	return alphas
}

func diffusionDDIMStep(sample, noise []float32, alpha, previousAlpha float32) {
	sqrtAlpha := float32(math.Sqrt(float64(alpha)))
	sqrtBeta := float32(math.Sqrt(float64(1 - alpha)))
	sqrtPrevious := float32(math.Sqrt(float64(previousAlpha)))
	direction := float32(math.Sqrt(float64(1 - previousAlpha)))
	for i := range sample {
		original := (sample[i] - sqrtBeta*noise[i]) / sqrtAlpha
		original = max(float32(-1), min(float32(1), original))
		sample[i] = sqrtPrevious*original + direction*noise[i]
	}
}

func (n *diffusionUNet) sample(ctx context.Context, sample []float32, steps int) error {
	if len(sample) != 3*32*32 || steps < 1 || steps > 1000 {
		return errors.New("diffusion: invalid sampling input")
	}
	workspace := diffusionWorkspace{ctx: ctx}
	alphas := diffusionAlphas()
	ratio := 1000 / steps
	for step := steps - 1; step >= 0; step-- {
		t := step * ratio
		noise, err := n.forward(&workspace, sample, t)
		if err != nil {
			return err
		}
		previous := float32(1)
		if t >= ratio {
			previous = alphas[t-ratio]
		}
		diffusionDDIMStep(sample, noise, alphas[t], previous)
	}
	return ctx.Err()
}
