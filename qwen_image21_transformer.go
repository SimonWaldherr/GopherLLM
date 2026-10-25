package gopherllm

// Qwen-Image-2.1's single-stream transformer follows the Apache-2.0 Diffusers
// reference (Qwen-Image Team / Hugging Face), revision
// 6256aa7666cedd47443adc8f82da9a10e110b09c, transformer_qwenimage21.py.
// This implementation supports text conditioning; image editing needs the
// separate vision/latent token layout and is intentionally not inferred here.

import (
	"context"
	"errors"
	"fmt"
	"math"
)

type qi21TransformerConfig struct {
	ClassName       string  `json:"_class_name"`
	PatchSize       int     `json:"patch_size"`
	InChannels      int     `json:"in_channels"`
	OutChannels     int     `json:"out_channels"`
	NumLayers       int     `json:"num_layers"`
	HeadDim         int     `json:"attention_head_dim"`
	Heads           int     `json:"num_attention_heads"`
	ContextDim      int     `json:"context_in_dim"`
	MLPRatio        int     `json:"mlp_ratio"`
	AxesDims        []int   `json:"axes_dims_rope"`
	Epsilon         float32 `json:"eps"`
	CausalCondition bool    `json:"causal_condition"`
}

type qi21TransformerLayer struct {
	q, k, v, out   Weight
	gate, up, down Weight
	qNorm, kNorm   []float32
}

type qi21Transformer struct {
	cfg                                                                  qi21TransformerConfig
	dim                                                                  int
	image, textIn, textOut, timeIn, timeOut, modulation, normOut, output Weight
	textNorm                                                             []float32
	layers                                                               []qi21TransformerLayer
}

func qi21NewTransformer(ctx context.Context, source qi21WeightSource, cfg qi21TransformerConfig) (*qi21Transformer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.ClassName != "QwenImage21Transformer2DModel" || cfg.PatchSize != 1 || !cfg.CausalCondition || cfg.InChannels < 1 || cfg.InChannels > 256 || cfg.OutChannels != cfg.InChannels || cfg.NumLayers < 1 || cfg.NumLayers > 128 || cfg.Heads < 1 || cfg.Heads > 128 || cfg.HeadDim < 2 || cfg.HeadDim > 512 || cfg.ContextDim < 1 || cfg.ContextDim > 16384 || cfg.MLPRatio < 1 || cfg.MLPRatio > 8 || !finite32(cfg.Epsilon) || cfg.Epsilon <= 0 {
		return nil, errors.New("qwen image: unsupported transformer configuration")
	}
	dim := cfg.Heads * cfg.HeadDim
	if dim > 16384 || len(cfg.AxesDims) != 3 {
		return nil, errors.New("qwen image: invalid transformer dimensions")
	}
	sum := 0
	for _, n := range cfg.AxesDims {
		if n <= 0 || n > cfg.HeadDim || n%2 != 0 {
			return nil, errors.New("qwen image: rotary axes must have positive even dimensions within the attention head")
		}
		sum += n
	}
	if sum != cfg.HeadDim {
		return nil, errors.New("qwen image: rotary axes do not cover the attention head")
	}
	m := &qi21Transformer{cfg: cfg, dim: dim, layers: make([]qi21TransformerLayer, cfg.NumLayers)}
	var err error
	matrix := func(dst *Weight, name string, rows, cols int) {
		if err == nil {
			if err = ctx.Err(); err == nil {
				*dst, err = source.matrix(name, rows, cols)
			}
		}
	}
	floats := func(dst *[]float32, name string, n int) {
		if err == nil {
			*dst, err = source.floats(name, n)
		}
	}
	matrix(&m.image, "img_in.weight", dim, cfg.InChannels)
	matrix(&m.textIn, "txt_in.in_layer.weight", dim, cfg.ContextDim)
	matrix(&m.textOut, "txt_in.out_layer.weight", dim, dim)
	floats(&m.textNorm, "txt_in.text_norm.weight", cfg.ContextDim)
	matrix(&m.timeIn, "time_text_embed.timestep_embedder.linear_1.weight", dim, 256)
	matrix(&m.timeOut, "time_text_embed.timestep_embedder.linear_2.weight", dim, dim)
	matrix(&m.modulation, "modulation.1.weight", 4*dim, dim)
	matrix(&m.normOut, "norm_out.linear.weight", dim, dim)
	matrix(&m.output, "proj_out.weight", cfg.OutChannels, dim)
	for i := range m.layers {
		l := &m.layers[i]
		prefix := fmt.Sprintf("transformer_blocks.%d.", i)
		matrix(&l.q, prefix+"attn.to_q.weight", dim, dim)
		matrix(&l.k, prefix+"attn.to_k.weight", dim, dim)
		matrix(&l.v, prefix+"attn.to_v.weight", dim, dim)
		matrix(&l.out, prefix+"attn.to_out.0.weight", dim, dim)
		floats(&l.qNorm, prefix+"attn.norm_q.weight", cfg.HeadDim)
		floats(&l.kNorm, prefix+"attn.norm_k.weight", cfg.HeadDim)
		matrix(&l.gate, prefix+"img_mlp.gate_layer.weight", dim*cfg.MLPRatio, dim)
		matrix(&l.up, prefix+"img_mlp.proj.weight", dim*cfg.MLPRatio, dim)
		matrix(&l.down, prefix+"img_mlp.out.weight", dim, dim*cfg.MLPRatio)
	}
	if err != nil {
		return nil, fmt.Errorf("qwen image transformer: %w", err)
	}
	return m, nil
}

const qi21TokenTile = 256

// Balance the remaining rows when every block can stay on the packed BLAS
// path. For example, 288 rows become 144+144 instead of 256+32. If the input
// budget cannot accommodate that, preserve full blocks and a small tail.
func qi21BatchRows(rows, cols int) int {
	limit := min(qi21TokenTile, max(1, prefillBLASInputElements/cols))
	if rows <= limit {
		return rows
	}
	blocks := (rows-1)/limit + 1
	if rows/blocks >= prefillBLASPackedMinTokens {
		return (rows-1)/blocks + 1
	}
	return limit
}

// Projection chunks keep BLAS input packing within eight MiB. Weight expansion
// remains bounded by the backend's row tile; matrices stay packed in memory.
func qi21Project(ctx context.Context, w Weight, input, output []float32) error {
	if w.Cols <= 0 || w.Rows <= 0 || len(input)%w.Cols != 0 || len(output) != len(input)/w.Cols*w.Rows {
		return errors.New("qwen image: invalid projection shape")
	}
	var inRows, outRows [qi21TokenTile][]float32
	rows := len(input) / w.Cols
	for first := 0; first < rows; {
		if err := ctx.Err(); err != nil {
			return err
		}
		n := qi21BatchRows(rows-first, w.Cols)
		in := input[first*w.Cols : (first+n)*w.Cols]
		out := output[first*w.Rows : (first+n)*w.Rows]
		for i := 0; i < n; i++ {
			inRows[i] = in[i*w.Cols : (i+1)*w.Cols]
			outRows[i] = out[i*w.Rows : (i+1)*w.Rows]
		}
		matvecBatchWithQ8AndSlabs(w, inRows[:n], outRows[:n], false, in, out)
		first += n
	}
	return nil
}

type qi21PrefixKV struct{ k, v []float32 }

// A state belongs to one prompt and latent grid. Its cache contains only
// prefix K/V; it never retains full image-token projections from a prior step.
// It is used serially by the enclosing generation call.
type qi21TransformerState struct {
	model                       *qi21Transformer
	height, width, prefix       int
	text                        []float32
	cache                       []qi21PrefixKV
	valid                       bool
	cosine, sine                []float32
	zeroMod                     []float32
	x, norm, q, k, v, attn, out []float32
	gate, up                    []float32
}

func (m *qi21Transformer) newState(text []float32, height, width int) (*qi21TransformerState, error) {
	if m == nil || len(text) == 0 || len(text)%m.cfg.ContextDim != 0 || height < 1 || width < 1 || height > 256 || width > 256 {
		return nil, errors.New("qwen image: invalid prompt or latent grid")
	}
	prefix := len(text) / m.cfg.ContextDim
	if prefix > 8192 || prefix+height*width > 65536 {
		return nil, errors.New("qwen image: prompt or latent grid exceeds supported limits")
	}
	// Token limits alone do not bound memory: prefix K/V scales with every
	// layer. Include the first step's six full-width slabs, rotary tables,
	// chunked FFN, conditioning copies and temporary timestep projections.
	// Weight mappings have their own lifetime and are not expanded here.
	p, target, d := uint64(prefix), uint64(height*width), uint64(m.dim)
	n := p + target
	elements := 2*uint64(len(m.layers))*p*d + 6*n*d + n*uint64(m.cfg.HeadDim)
	elements += 2*min(uint64(qi21TokenTile), n)*d*uint64(m.cfg.MLPRatio) + target*uint64(m.cfg.OutChannels)
	elements += 3*p*uint64(m.cfg.ContextDim) + uint64(m.cfg.ContextDim) + 16*d + 512
	const projectionScratch = 32 << 20
	bytes := elements*4 + projectionScratch
	if bytes > 2<<30 {
		return nil, fmt.Errorf("qwen image: estimated transformer state %.1f MiB exceeds 2 GiB; shorten the prompt or reduce image dimensions", float64(bytes)/(1<<20))
	}
	for _, v := range text {
		if !finite32(v) {
			return nil, errors.New("qwen image: non-finite prompt embedding")
		}
	}
	s := &qi21TransformerState{model: m, height: height, width: width, prefix: prefix, text: append([]float32(nil), text...), cache: make([]qi21PrefixKV, len(m.layers))}
	s.cosine, s.sine = qi21TextImageRoPE(m.cfg.AxesDims, prefix, height, width)
	return s, nil
}

func qi21TextImageRoPE(axes []int, prefix, height, width int) ([]float32, []float32) {
	pairs := (axes[0] + axes[1] + axes[2]) / 2
	n := prefix + height*width
	cosine, sine := make([]float32, n*pairs), make([]float32, n*pairs)
	for token := 0; token < n; token++ {
		positions := [3]int{token, token, token}
		if token >= prefix {
			i := token - prefix
			positions = [3]int{prefix, i/width - (height - height/2), i%width - (width - width/2)}
		}
		offset := 0
		for axis, dim := range axes {
			for j := 0; j < dim/2; j++ {
				frequency := float32(1 / math.Pow(10000, float64(2*j)/float64(dim)))
				angle := float32(positions[axis]) * frequency
				cosine[token*pairs+offset+j] = float32(math.Cos(float64(angle)))
				sine[token*pairs+offset+j] = float32(math.Sin(float64(angle)))
			}
			offset += dim / 2
		}
	}
	return cosine, sine
}

func (s *qi21TransformerState) timeParameters(ctx context.Context, t float32) (mod, scale []float32, err error) {
	m := s.model
	d := m.dim
	input := make([]float32, 256)
	for i := 0; i < 128; i++ {
		freq := float32(math.Exp(-math.Log(10000) * float64(i) / 128))
		angle := (t * 1000) * freq
		input[i] = float32(math.Cos(float64(angle)))
		input[i+128] = float32(math.Sin(float64(angle)))
	}
	a, b := make([]float32, d), make([]float32, d)
	if err = qi21Project(ctx, m.timeIn, input, a); err != nil {
		return nil, nil, err
	}
	for i := range a {
		a[i] = siluExact(a[i])
	}
	if err = qi21Project(ctx, m.timeOut, a, b); err != nil {
		return nil, nil, err
	}
	for i := range b {
		b[i] = siluExact(b[i])
	}
	mod, scale = make([]float32, 4*d), make([]float32, d)
	if err = qi21Project(ctx, m.modulation, b, mod); err != nil {
		return nil, nil, err
	}
	// Gates depend on timestep, not token position. Evaluate each tanh once
	// per step instead of once per channel of every image token and layer.
	for _, base := range []int{d, 3 * d} {
		for i := 0; i < d; i++ {
			mod[base+i] = float32(math.Tanh(float64(mod[base+i])))
		}
	}
	err = qi21Project(ctx, m.normOut, b, scale)
	return mod, scale, err
}

func (s *qi21TransformerState) predict(ctx context.Context, latents []float32, t float32) ([]float32, error) {
	if s == nil || s.model == nil {
		return nil, errors.New("qwen image: missing transformer state")
	}
	m := s.model
	d := m.dim
	target := s.height * s.width
	if len(latents) != target*m.cfg.InChannels || !finite32(t) || t < 0 || t > 1 {
		return nil, errors.New("qwen image: invalid latents or timestep")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, v := range latents {
		if !finite32(v) {
			return nil, errors.New("qwen image: non-finite latent")
		}
	}
	mod, scale, err := s.timeParameters(ctx, t)
	if err != nil {
		return nil, err
	}
	if s.zeroMod == nil {
		s.zeroMod, _, err = s.timeParameters(ctx, 0)
		if err != nil {
			return nil, err
		}
	}
	n, offset := target, s.prefix
	if !s.valid {
		n += s.prefix
		offset = 0
	}
	ensureLenNoClear(&s.x, n*d)
	ensureLenNoClear(&s.norm, n*d)
	ensureLenNoClear(&s.q, n*d)
	ensureLenNoClear(&s.k, (s.prefix+target)*d)
	ensureLenNoClear(&s.v, (s.prefix+target)*d)
	ensureLenNoClear(&s.attn, n*d)
	ensureLenNoClear(&s.out, target*m.cfg.OutChannels)
	if !s.valid {
		textNorm := make([]float32, len(s.text))
		weights := make([]float32, m.cfg.ContextDim)
		for i, v := range m.textNorm {
			weights[i] = v + 1
		}
		for i := 0; i < s.prefix; i++ {
			row := textNorm[i*m.cfg.ContextDim : (i+1)*m.cfg.ContextDim]
			rmsNormInto(s.text[i*m.cfg.ContextDim:(i+1)*m.cfg.ContextDim], weights, m.cfg.Epsilon, &row)
		}
		projected := s.norm[:s.prefix*d]
		if err = qi21Project(ctx, m.textIn, textNorm, projected); err != nil {
			return nil, err
		}
		for i, v := range projected {
			projected[i] = geluTanh(v)
		}
		if err = qi21Project(ctx, m.textOut, projected, s.x[:s.prefix*d]); err != nil {
			return nil, err
		}
	}
	firstTarget := n - target
	if err = qi21Project(ctx, m.image, latents, s.x[firstTarget*d:]); err != nil {
		return nil, err
	}
	for layer := range m.layers {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		l := &m.layers[layer]
		s.normalizeModulate(s.norm, s.x, mod, 0, firstTarget)
		if err = qi21Project(ctx, l.q, s.norm, s.q); err != nil {
			return nil, err
		}
		if s.valid {
			copy(s.k, s.cache[layer].k)
			copy(s.v, s.cache[layer].v)
		}
		keys := s.k[offset*d:]
		values := s.v[offset*d:]
		if err = qi21Project(ctx, l.k, s.norm, keys); err != nil {
			return nil, err
		}
		if err = qi21Project(ctx, l.v, s.norm, values); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			q, k := s.q[i*d:(i+1)*d], keys[i*d:(i+1)*d]
			perHeadRMSNormInPlace(q, m.cfg.HeadDim, m.cfg.Heads, l.qNorm, m.cfg.Epsilon)
			perHeadRMSNormInPlace(k, m.cfg.HeadDim, m.cfg.Heads, l.kNorm, m.cfg.Epsilon)
			s.rotate(q, i+offset)
			s.rotate(k, i+offset)
		}
		if !s.valid {
			// Full slices would retain the image scratch; own only this compact prefix.
			kv := &s.cache[layer]
			ensureLenNoClear(&kv.k, s.prefix*d)
			ensureLenNoClear(&kv.v, s.prefix*d)
			copy(kv.k, s.k[:s.prefix*d])
			copy(kv.v, s.v[:s.prefix*d])
		}
		s.attention(ctx, n, offset)
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		if err = qi21Project(ctx, l.out, s.attn, s.norm); err != nil {
			return nil, err
		}
		s.addGated(s.x, s.norm, mod, d, firstTarget)
		// FFN rows are independent. A small slab avoids two full image-sized
		// hidden projections (over 1.5 GiB at the checkpoint's 2K resolution).
		hidden := d * m.cfg.MLPRatio
		ensureLenNoClear(&s.gate, min(qi21TokenTile, n)*hidden)
		ensureLenNoClear(&s.up, min(qi21TokenTile, n)*hidden)
		s.normalizeModulate(s.norm, s.x, mod, 2*d, firstTarget)
		for begin := 0; begin < n; {
			count := qi21BatchRows(n-begin, d)
			input := s.norm[begin*d : (begin+count)*d]
			gate, up := s.gate[:count*hidden], s.up[:count*hidden]
			if err = qi21Project(ctx, l.gate, input, gate); err != nil {
				return nil, err
			}
			if err = qi21Project(ctx, l.up, input, up); err != nil {
				return nil, err
			}
			siluMulF32(gate, up, gate)
			output := s.attn[begin*d : (begin+count)*d]
			if err = qi21Project(ctx, l.down, gate, output); err != nil {
				return nil, err
			}
			begin += count
		}
		s.addGated(s.x, s.attn, mod, 3*d, firstTarget)
	}
	targetX := s.x[firstTarget*d:]
	normalized := s.norm[:target*d]
	for i := 0; i < target; i++ {
		row := normalized[i*d : (i+1)*d]
		layerNormInto(targetX[i*d:(i+1)*d], nil, nil, m.cfg.Epsilon, &row)
		for j := range row {
			row[j] *= 1 + scale[j]
		}
	}
	if err = qi21Project(ctx, m.output, normalized, s.out); err != nil {
		return nil, err
	}
	for _, v := range s.out {
		if !finite32(v) {
			return nil, errors.New("qwen image: non-finite transformer output")
		}
	}
	s.valid = true
	return s.out, nil
}

func (s *qi21TransformerState) normalizeModulate(dst, x, mod []float32, base, firstTarget int) {
	d := s.model.dim
	for i := 0; i < len(x)/d; i++ {
		params := mod
		if i < firstTarget {
			params = s.zeroMod
		}
		out := dst[i*d : (i+1)*d]
		layerNormInto(x[i*d:(i+1)*d], nil, nil, s.model.cfg.Epsilon, &out)
		for j := range out {
			out[j] *= 1 + params[base+j]
		}
	}
}

func (s *qi21TransformerState) addGated(dst, add, mod []float32, base, firstTarget int) {
	d := s.model.dim
	for i := 0; i < len(dst)/d; i++ {
		params := mod
		if i < firstTarget {
			params = s.zeroMod
		}
		for j := 0; j < d; j++ {
			dst[i*d+j] += params[base+j] * add[i*d+j]
		}
	}
}

func (s *qi21TransformerState) rotate(x []float32, position int) {
	hd := s.model.cfg.HeadDim
	pairs := hd / 2
	cosine, sine := s.cosine[position*pairs:(position+1)*pairs], s.sine[position*pairs:(position+1)*pairs]
	for head := 0; head < s.model.cfg.Heads; head++ {
		v := x[head*hd : (head+1)*hd]
		for j := 0; j < pairs; j++ {
			a, b := v[2*j], v[2*j+1]
			v[2*j] = a*cosine[j] - b*sine[j]
			v[2*j+1] = a*sine[j] + b*cosine[j]
		}
	}
}

func (s *qi21TransformerState) attention(ctx context.Context, n, offset int) {
	m := s.model
	hd := m.cfg.HeadDim
	scale := float32(1 / math.Sqrt(float64(hd)))
	parallelRows(n*m.cfg.Heads, func(begin, end int) {
		for row := begin; row < end; row++ {
			if ctx.Err() != nil {
				return
			}
			token, head := row/m.cfg.Heads, row%m.cfg.Heads
			endKey := s.prefix + s.height*s.width
			if token+offset < s.prefix {
				endKey = token + offset + 1
			}
			start := token*m.dim + head*hd
			clear(s.attn[start : start+hd])
			onlineAttention(s.q[start:start+hd], s.k[head*hd:], s.v[head*hd:], m.dim, m.dim, hd, hd, 0, endKey-1, scale, 0, s.attn[start:start+hd])
		}
	})
}
