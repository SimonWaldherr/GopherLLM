package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
)

// FLUX.2/3's decoder differs from Qwen's RGBA decoder: group normalization,
// three RGB channels and a 2x2 packed 32-channel latent with batch statistics.
// Convolutions share the bounded im2col implementation used by Qwen's VAE.
type flux3VAEResnet struct {
	norm1, norm2 diffusionNorm
	conv1, conv2 qi21VAEConv
	shortcut     *qi21VAEConv
}

type flux3VAE struct {
	mean, variance            []float32
	post, input, output       qi21VAEConv
	mid                       [2]flux3VAEResnet
	attentionNorm, outputNorm diffusionNorm
	q, k, v, projection       qi21VAEConv
	up                        [4][]flux3VAEResnet
	upsample                  [3]qi21VAEConv
}

func newFlux3VAE(ctx context.Context, src *qi21Weights) (*flux3VAE, error) {
	shape := src.tensors["decoder.conv_out.weight"].shape
	if len(shape) != 4 || shape[0] != 3 || shape[1] < 32 || shape[1] > 256 || shape[1]%32 != 0 {
		return nil, errors.New("flux3: expected a native FLUX RGB autoencoder")
	}
	base := shape[1]
	var err error
	floats := func(name string, shape ...int) []float32 {
		if err != nil {
			return nil
		}
		err = ctx.Err()
		if err != nil {
			return nil
		}
		var v []float32
		v, err = src.floats(name, shape...)
		return v
	}
	conv := func(name string, in, out, k int) qi21VAEConv {
		return qi21VAEConv{floats(name+".weight", out, in, k, k), floats(name+".bias", out), in, out, k}
	}
	norm := func(name string, c int) diffusionNorm {
		return diffusionNorm{floats(name+".weight", c), floats(name+".bias", c)}
	}
	res := func(name string, in, out int) flux3VAEResnet {
		r := flux3VAEResnet{norm1: norm(name+".norm1", in), norm2: norm(name+".norm2", out), conv1: conv(name+".conv1", in, out, 3), conv2: conv(name+".conv2", out, out, 3)}
		if in != out {
			c := conv(name+".nin_shortcut", in, out, 1)
			r.shortcut = &c
		}
		return r
	}
	d := 4 * base
	v := &flux3VAE{mean: floats("bn.running_mean", 128), variance: floats("bn.running_var", 128), post: conv("decoder.post_quant_conv", 32, 32, 1), input: conv("decoder.conv_in", 32, d, 3)}
	v.mid = [2]flux3VAEResnet{res("decoder.mid.block_1", d, d), res("decoder.mid.block_2", d, d)}
	p := "decoder.mid.attn_1."
	v.attentionNorm = norm(p+"norm", d)
	v.q, v.k, v.v, v.projection = conv(p+"q", d, d, 1), conv(p+"k", d, d, 1), conv(p+"v", d, d, 1), conv(p+"proj_out", d, d, 1)
	for level := 3; level >= 0; level-- {
		out := base * []int{1, 2, 4, 4}[level]
		for j := range 3 {
			v.up[level] = append(v.up[level], res(fmt.Sprintf("decoder.up.%d.block.%d", level, j), d, out))
			d = out
		}
		if level > 0 {
			v.upsample[level-1] = conv(fmt.Sprintf("decoder.up.%d.upsample.conv", level), d, d, 3)
		}
	}
	v.outputNorm = norm("decoder.norm_out", d)
	v.output = conv("decoder.conv_out", d, 3, 3)
	if err != nil {
		return nil, fmt.Errorf("flux3 VAE: %w", err)
	}
	for _, f := range v.variance {
		if f < 0 {
			return nil, errors.New("flux3: negative VAE latent variance")
		}
	}
	return v, nil
}

type flux3VAEWorkspace struct {
	conv   qi21VAEWorkspace
	norm   diffusionWorkspace
	scores []float32
}

func (r *flux3VAEResnet) forward(s *flux3VAEWorkspace, x diffusionTensor) diffusionTensor {
	if s.conv.err != nil {
		return x
	}
	normalized := diffusionTensor{make([]float32, len(x.data)), x.c, x.h, x.w}
	r.norm1.apply(&s.norm, normalized.data, x.data, x.h*x.w, true)
	y := r.conv1.forward(&s.conv, normalized)
	if s.conv.err != nil {
		return x
	}
	r.norm2.apply(&s.norm, y.data, y.data, y.h*y.w, true)
	y = r.conv2.forwardInto(&s.conv, y, normalized.data)
	if r.shortcut != nil {
		x = r.shortcut.forward(&s.conv, x)
	}
	if s.conv.err == nil {
		addInPlace(y.data, x.data)
	}
	return y
}

func (v *flux3VAE) attention(s *flux3VAEWorkspace, x diffusionTensor) diffusionTensor {
	if s.conv.err != nil {
		return x
	}
	n := x.h * x.w
	y := diffusionTensor{make([]float32, len(x.data)), x.c, x.h, x.w}
	v.attentionNorm.apply(&s.norm, y.data, x.data, n, false)
	flat := func(c qi21VAEConv) []float32 {
		z := c.forward(&s.conv, y)
		out := make([]float32, len(z.data))
		for p := range n {
			for ch := range x.c {
				out[p*x.c+ch] = z.data[ch*n+p]
			}
		}
		return out
	}
	q, k, values := flat(v.q), flat(v.k), flat(v.v)
	if s.conv.err != nil {
		return x
	}
	mixed := make([]float32, len(x.data))
	imageAttention(s.conv.ctx, q, k, values, mixed, n, n, 1, x.c, 0, 0, &s.scores)
	for p := range n {
		for ch := range x.c {
			y.data[ch*n+p] = mixed[p*x.c+ch]
		}
	}
	y = v.projection.forward(&s.conv, y)
	if s.conv.err == nil {
		addInPlace(y.data, x.data)
	}
	return y
}

func (v *flux3VAE) validateSize(h, w int) error {
	if h < 1 || w < 1 || h > 128 || w > 128 {
		return errors.New("flux3: invalid VAE latent grid")
	}
	// The largest RGB feature maps have baseWidth channels. Four live maps
	// cover a residual/upsample plus a 32 MiB convolution workspace.
	if uint64(h)*uint64(w)*256*uint64(v.output.in)*16+32<<20 > 2<<30 {
		return errors.New("flux3: VAE workspace exceeds 2 GiB; reduce image dimensions")
	}
	return nil
}

func (v *flux3VAE) decode(ctx context.Context, tokens []float32, h, w int) (*image.NRGBA, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := v.validateSize(h, w); err != nil {
		return nil, err
	}
	if len(tokens) != h*w*128 {
		return nil, errors.New("flux3: invalid VAE latent shape")
	}
	x := diffusionTensor{make([]float32, 32*4*h*w), 32, 2 * h, 2 * w}
	for row := range h {
		for col := range w {
			for ch := range 32 {
				for dy := range 2 {
					for dx := range 2 {
						packed := ch*4 + dy*2 + dx
						f := tokens[(row*w+col)*128+packed]
						if !finite32(f) {
							return nil, errors.New("flux3: non-finite latent")
						}
						x.data[(ch*x.h+row*2+dy)*x.w+col*2+dx] = f*float32(math.Sqrt(float64(v.variance[packed]+1e-4))) + v.mean[packed]
					}
				}
			}
		}
	}
	s := &flux3VAEWorkspace{conv: qi21VAEWorkspace{ctx: ctx}}
	x = v.input.forward(&s.conv, v.post.forward(&s.conv, x))
	x = v.mid[0].forward(s, x)
	x = v.attention(s, x)
	x = v.mid[1].forward(s, x)
	for level := 3; level >= 0; level-- {
		for i := range v.up[level] {
			x = v.up[level][i].forward(s, x)
		}
		if s.conv.err != nil {
			return nil, s.conv.err
		}
		if level > 0 {
			x = v.upsample[level-1].upsample(&s.conv, x)
		}
	}
	v.outputNorm.apply(&s.norm, x.data, x.data, x.h*x.w, true)
	x = v.output.forward(&s.conv, x)
	if s.conv.err != nil {
		return nil, s.conv.err
	}
	img := image.NewNRGBA(image.Rect(0, 0, x.w, x.h))
	for p := range x.h * x.w {
		for ch := range 3 {
			f := x.data[ch*x.h*x.w+p]
			if !finite32(f) {
				return nil, errors.New("flux3: non-finite decoded image")
			}
			img.Pix[4*p+ch] = uint8((max(-1, min(1, f)) + 1) * 127.5)
		}
		img.Pix[4*p+3] = 255
	}
	return img, ctx.Err()
}
