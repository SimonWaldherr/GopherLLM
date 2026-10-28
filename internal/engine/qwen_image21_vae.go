package engine

// Native image specialization of Diffusers' AutoencoderKLQwenImage21:
// https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py
// The temporal convolutions are unused when decoding the first and only frame.

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"
)

type qwenImage21VAEConfig struct {
	ClassName          string    `json:"_class_name"`
	BaseDim            int       `json:"base_dim"`
	DecoderBaseDim     int       `json:"decoder_base_dim"`
	ZDim               int       `json:"z_dim"`
	DimMult            []int     `json:"dim_mult"`
	NumResBlocks       int       `json:"num_res_blocks"`
	InChannels         int       `json:"in_channels"`
	OutChannels        int       `json:"out_channels"`
	IsResidual         bool      `json:"is_residual"`
	PatchSize          *int      `json:"patch_size"`
	ScaleFactorSpatial int       `json:"scale_factor_spatial"`
	TemporalDownsample []bool    `json:"temperal_downsample"`
	LatentsMean        []float32 `json:"latents_mean"`
	LatentsStd         []float32 `json:"latents_std"`
	AttentionScales    []float32 `json:"attn_scales"`
	Dropout            float32   `json:"dropout"`
}

type qi21VAEConv struct {
	weight, bias    []float32
	in, out, kernel int
}

type qi21VAEResidual struct {
	norm1, norm2 []float32
	conv1, conv2 qi21VAEConv
	shortcut     *qi21VAEConv
}

type qi21VAEUp struct {
	resnets  []qi21VAEResidual
	conv     *qi21VAEConv
	temporal int
}

type qwenImage21VAE struct {
	config                    qwenImage21VAEConfig
	postQuant, input, output  qi21VAEConv
	mid1, mid2                qi21VAEResidual
	attentionNorm, outputNorm []float32
	qkv, projection           qi21VAEConv
	up                        []qi21VAEUp
}

func loadQwenImage21VAE(ctx context.Context, src qi21WeightSource, cfg qwenImage21VAEConfig) (*qwenImage21VAE, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.ClassName != "AutoencoderKLQwenImage21" || cfg.DecoderBaseDim < 1 || cfg.DecoderBaseDim > 256 ||
		cfg.ZDim < 1 || cfg.ZDim > 128 || cfg.NumResBlocks != 2 || cfg.InChannels != 4 || cfg.OutChannels != 4 ||
		!cfg.IsResidual || cfg.PatchSize != nil || cfg.ScaleFactorSpatial != 16 || cfg.Dropout != 0 || len(cfg.AttentionScales) != 0 ||
		!slices.Equal(cfg.DimMult, []int{1, 2, 4, 8, 8}) || !slices.Equal(cfg.TemporalDownsample, []bool{false, true, true, true}) ||
		len(cfg.LatentsMean) != cfg.ZDim || len(cfg.LatentsStd) != cfg.ZDim {
		return nil, errors.New("qwen image 2.1: unsupported RGBA VAE configuration")
	}
	for i := range cfg.ZDim {
		if !finite32(cfg.LatentsMean[i]) || !finite32(cfg.LatentsStd[i]) || cfg.LatentsStd[i] <= 0 {
			return nil, errors.New("qwen image 2.1: invalid VAE latent normalization")
		}
	}
	var loadErr error
	floats := func(name string, shape ...int) []float32 {
		if loadErr != nil {
			return nil
		}
		if loadErr = ctx.Err(); loadErr != nil {
			return nil
		}
		var values []float32
		values, loadErr = src.floats(name, shape...)
		return values
	}
	conv := func(name string, in, out, kernel int) qi21VAEConv {
		return qi21VAEConv{floats(name+".weight", out, in, kernel, kernel), floats(name+".bias", out), in, out, kernel}
	}
	residual := func(name string, in, out int) qi21VAEResidual {
		r := qi21VAEResidual{norm1: floats(name+".norm1.gamma", in, 1, 1, 1), norm2: floats(name+".norm2.gamma", out, 1, 1, 1),
			conv1: conv(name+".conv1", in, out, 3), conv2: conv(name+".conv2", out, out, 3)}
		if in != out {
			c := conv(name+".conv_shortcut", in, out, 1)
			r.shortcut = &c
		}
		return r
	}
	dim := cfg.DecoderBaseDim * 8
	v := &qwenImage21VAE{config: cfg,
		postQuant: conv("post_quant_conv", cfg.ZDim, cfg.ZDim, 1), input: conv("decoder.conv_in", cfg.ZDim, dim, 3),
		mid1: residual("decoder.mid_block.resnets.0", dim, dim), mid2: residual("decoder.mid_block.resnets.1", dim, dim),
		attentionNorm: floats("decoder.mid_block.attentions.0.norm.gamma", dim, 1, 1),
		qkv:           conv("decoder.mid_block.attentions.0.to_qkv", dim, 3*dim, 1), projection: conv("decoder.mid_block.attentions.0.proj", dim, dim, 1)}
	for i, mult := range []int{8, 8, 4, 2, 1} {
		out := cfg.DecoderBaseDim * mult
		u := qi21VAEUp{temporal: 1}
		if i < 3 {
			u.temporal = 2
		}
		for j := range cfg.NumResBlocks + 1 {
			u.resnets = append(u.resnets, residual(fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", i, j), dim, out))
			dim = out
		}
		if i < 4 {
			c := conv(fmt.Sprintf("decoder.up_blocks.%d.upsampler.resample.1", i), out, out, 3)
			u.conv = &c
		}
		v.up = append(v.up, u)
	}
	v.outputNorm = floats("decoder.norm_out.gamma", dim, 1, 1, 1)
	v.output = conv("decoder.conv_out", dim, 4, 3)
	if loadErr == nil {
		loadErr = ctx.Err()
	}
	if loadErr != nil {
		return nil, fmt.Errorf("qwen image 2.1 VAE: %w", loadErr)
	}
	return v, nil
}

// Convolution patches cover at most 256 spatial positions, avoiding a full
// image-sized im2col buffer. Scratch is shared by all operators in one decode.
type qi21VAEWorkspace struct {
	ctx            context.Context
	patch, product []float32
	upsampleRow    []float32
	matmul         denseGemmScratch
	err            error
}

func (s *qi21VAEWorkspace) gemm(m, n, k int, a, b, out []float32) {
	if s.err == nil {
		s.err = s.matmul.multiply(s.ctx, m, n, k, a, b, out)
	}
}

func (c qi21VAEConv) forward(s *qi21VAEWorkspace, x diffusionTensor) diffusionTensor {
	return c.forwardSpatial(s, x, false, nil)
}

// The caller owns dst and keeps it disjoint from x. Residual blocks can reuse
// their consumed normalization buffer instead of allocating another map.
func (c qi21VAEConv) forwardInto(s *qi21VAEWorkspace, x diffusionTensor, dst []float32) diffusionTensor {
	return c.forwardSpatial(s, x, false, dst)
}

// Fold nearest-neighbor upsampling into patch extraction. The expanded
// feature map can occupy hundreds of MiB but has no other consumer.
func (c qi21VAEConv) upsample(s *qi21VAEWorkspace, x diffusionTensor) diffusionTensor {
	return c.forwardSpatial(s, x, true, nil)
}

func (c qi21VAEConv) forwardSpatial(s *qi21VAEWorkspace, x diffusionTensor, upsample bool, dst []float32) diffusionTensor {
	if s.err != nil {
		return x
	}
	if err := s.ctx.Err(); err != nil {
		s.err = err
		return x
	}
	h, w := x.h, x.w
	if upsample {
		h *= 2
		w *= 2
	}
	n, k := h*w, c.in*c.kernel*c.kernel
	ensureLenNoClear(&dst, c.out*n)
	y := diffusionTensor{dst, c.out, h, w}
	if c.kernel == 1 && !upsample {
		s.gemm(c.out, n, c.in, c.weight, x.data, y.data)
	} else {
		const block = 256
		if cap(s.patch) < k*min(n, block) {
			s.patch = make([]float32, k*min(n, block))
		}
		if cap(s.product) < c.out*min(n, block) {
			s.product = make([]float32, c.out*min(n, block))
		}
		if upsample {
			ensureLenNoClear(&s.upsampleRow, min(n, block)+c.kernel-1)
		}
		for start := 0; start < n; start += block {
			width := min(block, n-start)
			patch, product := s.patch[:k*width], s.product[:c.out*width]
			for ch := range c.in {
				for ky := range c.kernel {
					for p := 0; p < width; {
						yy, xx := (start+p)/w+ky-c.kernel/2, (start+p)%w
						run := min(width-p, w-xx)
						var src []float32
						if yy >= 0 && yy < h {
							if upsample {
								yy /= 2
							}
							src = x.data[(ch*x.h+yy)*x.w : (ch*x.h+yy+1)*x.w]
							if upsample {
								// Expand one small row window for all horizontal
								// kernel offsets, including their zero padding.
								row := s.upsampleRow[:run+c.kernel-1]
								copyPaddedRepeat2(row, src, xx-c.kernel/2)
								src = row
							}
						}
						for kx := range c.kernel {
							row := patch[((ch*c.kernel+ky)*c.kernel+kx)*width:][:width]
							dst := row[p : p+run]
							if src == nil {
								clear(dst)
							} else if upsample {
								copy(dst, src[kx:kx+run])
							} else {
								copyPaddedWindow(dst, src, xx+kx-c.kernel/2)
							}
						}
						p += run
					}
				}
			}
			s.gemm(c.out, width, k, c.weight, patch, product)
			if s.err != nil {
				return x
			}
			for ch := range c.out {
				copy(y.data[ch*n+start:ch*n+start+width], product[ch*width:(ch+1)*width])
			}
		}
	}
	for ch, bias := range c.bias {
		for p := range n {
			y.data[ch*n+p] += bias
		}
	}
	return y
}

// The reference uses F.normalize(x, dim=channels) * sqrt(channels), with a
// clamp on the L2 norm, not an epsilon added to the mean square.
func qi21VAENorm(x diffusionTensor, gamma []float32, activate bool) {
	n := x.h * x.w
	scale := float32(math.Sqrt(float64(x.c)))
	for p := range n {
		sum := float32(0)
		for ch := range x.c {
			value := x.data[ch*n+p]
			sum += value * value
		}
		denom := max(float32(math.Sqrt(float64(sum))), float32(1e-12))
		for ch := range x.c {
			value := x.data[ch*n+p] / denom * scale * gamma[ch]
			if activate {
				value = siluExact(value)
			}
			x.data[ch*n+p] = value
		}
	}
}

func (r qi21VAEResidual) forward(s *qi21VAEWorkspace, x diffusionTensor) diffusionTensor {
	if s.err != nil {
		return x
	}
	shortcut := x
	if r.shortcut != nil {
		shortcut = r.shortcut.forward(s, x)
	}
	if s.err != nil {
		return x
	}
	normalized := diffusionTensor{slices.Clone(x.data), x.c, x.h, x.w}
	qi21VAENorm(normalized, r.norm1, true)
	y := r.conv1.forward(s, normalized)
	if s.err != nil {
		return x
	}
	qi21VAENorm(y, r.norm2, true)
	y = r.conv2.forwardInto(s, y, normalized.data)
	if s.err != nil {
		return x
	}
	for i, value := range shortcut.data {
		y.data[i] += value
	}
	return y
}

func (v *qwenImage21VAE) attention(s *qi21VAEWorkspace, x diffusionTensor) diffusionTensor {
	if s.err != nil {
		return x
	}
	n := x.h * x.w
	normalized := diffusionTensor{slices.Clone(x.data), x.c, x.h, x.w}
	qi21VAENorm(normalized, v.attentionNorm, false)
	qkv := v.qkv.forward(s, normalized)
	if s.err != nil {
		return x
	}
	q, k, values := qkv.data[:x.c*n], qkv.data[x.c*n:2*x.c*n], qkv.data[2*x.c*n:]
	queries := make([]float32, n*x.c)
	for ch := range x.c {
		for p := range n {
			queries[p*x.c+ch] = q[ch*n+p]
		}
	}
	scores := make([]float32, n*n)
	s.gemm(n, n, x.c, queries, k, scores)
	if s.err != nil {
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
	transposed := make([]float32, n*n)
	for i := range n {
		for j := range n {
			transposed[j*n+i] = scores[i*n+j]
		}
	}
	mixed := diffusionTensor{make([]float32, len(x.data)), x.c, x.h, x.w}
	s.gemm(x.c, n, n, values, transposed, mixed.data)
	y := v.projection.forward(s, mixed)
	if s.err != nil {
		return x
	}
	for i, value := range x.data {
		y.data[i] += value
	}
	return y
}

// DupUp3D repeats channels before rearranging them into temporal/spatial
// positions. For the first frame only the last temporal subpixel survives.
func qi21VAEAddUpShortcut(dst, src diffusionTensor, temporal int) {
	repeats := dst.c * temporal * 4 / src.c
	n := dst.h * dst.w
	for ch := range dst.c {
		for yy := range dst.h {
			for xx := range dst.w {
				inputChannel := (ch*temporal*4 + (temporal-1)*4 + (yy%2)*2 + xx%2) / repeats
				dst.data[ch*n+yy*dst.w+xx] += src.data[(inputChannel*src.h+yy/2)*src.w+xx/2]
			}
		}
	}
}

func (v *qwenImage21VAE) decodeTile(s *qi21VAEWorkspace, x diffusionTensor) diffusionTensor {
	x = v.postQuant.forward(s, x)
	if s.err != nil {
		return x
	}
	x = v.input.forward(s, x)
	if s.err != nil {
		return x
	}
	x = v.mid1.forward(s, x)
	if s.err != nil {
		return x
	}
	x = v.attention(s, x)
	if s.err != nil {
		return x
	}
	x = v.mid2.forward(s, x)
	if s.err != nil {
		return x
	}
	for _, up := range v.up {
		shortcut := x
		for _, residual := range up.resnets {
			x = residual.forward(s, x)
			if s.err != nil {
				return x
			}
		}
		if up.conv != nil {
			x = up.conv.upsample(s, x)
			if s.err != nil {
				return x
			}
			qi21VAEAddUpShortcut(x, shortcut, up.temporal)
		}
	}
	qi21VAENorm(x, v.outputNorm, true)
	return v.output.forward(s, x)
}

// decode accepts the transformer's normalized [H*W, Z] token-major latents.
// Large images use the official VAE's 16/12-latent tile/stride and ordered
// vertical-then-horizontal blending. This matches enable_tiling, not an
// untiled decoder's global mid-block attention.
func (v *qwenImage21VAE) decode(ctx context.Context, latents []float32, h, w int) (*image.NRGBA, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if v == nil || h < 1 || w < 1 || h > 256 || w > 256 || len(latents) != h*w*v.config.ZDim {
		return nil, errors.New("qwen image 2.1: invalid VAE latent shape")
	}
	for _, value := range latents {
		if !finite32(value) {
			return nil, errors.New("qwen image 2.1: non-finite VAE latents")
		}
	}
	out := image.NewNRGBA(image.Rect(0, 0, w*16, h*16))
	s := &qi21VAEWorkspace{ctx: ctx}
	tileSize, stride := 16, 12
	if h <= tileSize && w <= tileSize {
		stride = tileSize
	}
	previous := make([]diffusionTensor, (w+stride-1)/stride)
	for top := 0; top < h; top += stride {
		var left diffusionTensor
		for start := 0; start < w; start += stride {
			th, tw := min(tileSize, h-top), min(tileSize, w-start)
			input := diffusionTensor{make([]float32, v.config.ZDim*th*tw), v.config.ZDim, th, tw}
			for ch := range input.c {
				for y := range th {
					for x := range tw {
						input.data[(ch*th+y)*tw+x] = latents[((top+y)*w+start+x)*input.c+ch]*v.config.LatentsStd[ch] + v.config.LatentsMean[ch]
					}
				}
			}
			tile := v.decodeTile(s, input)
			if s.err != nil {
				return nil, s.err
			}
			column := start / stride
			if top > 0 {
				qi21VAEBlend(previous[column], tile, 64, true)
			}
			if start > 0 {
				qi21VAEBlend(left, tile, 64, false)
			}
			previous[column], left = tile, tile
			for y := range min(stride*16, tile.h) {
				for x := range min(stride*16, tile.w) {
					for ch := range 4 {
						value := tile.data[(ch*tile.h+y)*tile.w+x]
						if !finite32(value) {
							return nil, errors.New("qwen image 2.1: non-finite VAE output")
						}
						out.Pix[(top*16+y)*out.Stride+(start*16+x)*4+ch] = uint8(math.RoundToEven(float64(max(float32(0), min(float32(1), value/2+.5)) * 255)))
					}
				}
			}
		}
	}
	return out, nil
}

func qi21VAEBlend(a, b diffusionTensor, extent int, vertical bool) {
	if vertical {
		extent = min(extent, a.h, b.h)
		for y := range extent {
			blend := float32(float64(y) / float64(extent))
			keep := float32(1 - float64(y)/float64(extent))
			for ch := range b.c {
				for x := range b.w {
					i := (ch*b.h+y)*b.w + x
					b.data[i] = a.data[(ch*a.h+a.h-extent+y)*a.w+x]*keep + b.data[i]*blend
				}
			}
		}
	} else {
		extent = min(extent, a.w, b.w)
		for x := range extent {
			blend := float32(float64(x) / float64(extent))
			keep := float32(1 - float64(x)/float64(extent))
			for ch := range b.c {
				for y := range b.h {
					i := (ch*b.h+y)*b.w + x
					b.data[i] = a.data[(ch*a.h+y)*a.w+a.w-extent+x]*keep + b.data[i]*blend
				}
			}
		}
	}
}
