package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Native BFL tensor names, from flux-3-image-klein at
// bfc8889a4497f846416f253a3972a1941ac698b9. Empty reference-image streams
// have no attention edges and need no phantom projections for text-to-image.
type flux3Block struct {
	q, k, v, out, mlpIn, mlpOut Weight
	qNorm, kNorm                []float32
}

type flux3Transformer struct {
	dim, heads, headDim, contextDim, hidden     int
	image, text, timeIn, timeOut, refIn, refOut Weight
	earlyMod, singleMod                         [2]Weight // image, text
	finalMod, output                            Weight
	early                                       [2][]flux3Block
	single                                      []flux3Block
	flowMap                                     bool
}

func flux3BlockCount(s *qi21Weights, prefix string) (int, error) {
	indices := map[int]bool{}
	for name := range s.tensors {
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		index, _, ok := strings.Cut(strings.TrimPrefix(name, prefix), ".")
		i, err := strconv.Atoi(index)
		if !ok || err != nil || i < 0 || i >= 64 || strconv.Itoa(i) != index {
			return 0, fmt.Errorf("flux3: invalid block name %s", name)
		}
		indices[i] = true
	}
	if len(indices) == 0 {
		return 0, fmt.Errorf("flux3: missing %s blocks", prefix)
	}
	for i := range len(indices) {
		if !indices[i] {
			return 0, fmt.Errorf("flux3: noncontiguous %s blocks", prefix)
		}
	}
	return len(indices), nil
}

func newFlux3Transformer(ctx context.Context, s *qi21Weights) (*flux3Transformer, error) {
	image := s.tensors["emb_in.image.weight"].shape
	text := s.tensors["txt_in.weight"].shape
	norm := s.tensors["txt_mode_blocks.0.norm.query_norm.scale"].shape
	mlp := s.tensors["txt_mode_blocks.0.mlp_out.weight"].shape
	if len(image) != 2 || len(text) != 2 || len(norm) != 1 || len(mlp) != 2 {
		return nil, errors.New("flux3: expected BFL FLUX 3 Image Klein tensors")
	}
	d, hd := image[0], norm[0]
	if d < 8 || d > 8192 || hd < 8 || hd > 256 || hd%8 != 0 || d%hd != 0 || image[1] != 128 || text[1] < 8 || text[1] > 65536 || mlp[1] != 3*d {
		return nil, errors.New("flux3: unsupported transformer dimensions")
	}
	m := &flux3Transformer{dim: d, heads: d / hd, headDim: hd, contextDim: text[1], hidden: 3 * d}
	_, m.flowMap = s.tensors["ref_time_in.in_layer.weight"]
	var err error
	matrix := func(name string, rows, cols int) Weight {
		if err != nil {
			return Weight{}
		}
		err = ctx.Err()
		if err != nil {
			return Weight{}
		}
		var w Weight
		w, err = s.matrix(name, rows, cols)
		return w
	}
	floats := func(name string, size int) []float32 {
		if err != nil {
			return nil
		}
		var out []float32
		out, err = s.floats(name, size)
		return out
	}
	m.image, m.text = matrix("emb_in.image.weight", d, 128), matrix("txt_in.weight", d, m.contextDim)
	m.timeIn, m.timeOut = matrix("time_in.in_layer.weight", d, 256), matrix("time_in.out_layer.weight", d, d)
	if m.flowMap {
		m.refIn, m.refOut = matrix("ref_time_in.in_layer.weight", d, 256), matrix("ref_time_in.out_layer.weight", d, d)
	} else if _, exists := s.tensors["ref_time_in.out_layer.weight"]; exists {
		return nil, errors.New("flux3: incomplete flow-map conditioning")
	}
	for i, mode := range []string{"image", "txt"} {
		m.earlyMod[i] = matrix("early_stream_modulations."+mode+".lin.weight", 3*d, d)
		m.singleMod[i] = matrix("single_stream_modulations."+mode+".lin.weight", 3*d, d)
	}
	m.finalMod = matrix("final_layer.image.adaLN_modulation.1.weight", 2*d, d)
	m.output = matrix("final_layer.image.linear.weight", 128, d)
	for stream, prefix := range []string{"content_mode_blocks.image.", "txt_mode_blocks.", "single_blocks."} {
		count, e := flux3BlockCount(s, prefix)
		if e != nil {
			return nil, e
		}
		blocks := make([]flux3Block, count)
		for i := range blocks {
			p := prefix + strconv.Itoa(i) + "."
			blocks[i] = flux3Block{
				q: matrix(p+"q_proj.weight", d, d), k: matrix(p+"k_proj.weight", d, d), v: matrix(p+"v_proj.weight", d, d), out: matrix(p+"attn_out.weight", d, d),
				mlpIn: matrix(p+"mlp_in.weight", 2*m.hidden, d), mlpOut: matrix(p+"mlp_out.weight", d, m.hidden),
				qNorm: floats(p+"norm.query_norm.scale", hd), kNorm: floats(p+"norm.key_norm.scale", hd),
			}
		}
		if stream < 2 {
			m.early[stream] = blocks
		} else {
			m.single = blocks
		}
	}
	if len(m.early[0]) != len(m.early[1]) {
		return nil, errors.New("flux3: inconsistent early stream depths")
	}
	if err != nil {
		return nil, fmt.Errorf("flux3 transformer: %w", err)
	}
	return m, nil
}

type flux3State struct {
	model                               *flux3Transformer
	imageTokens, textTokens             int
	text, cosine, sine                  []float32
	textProjection                      []float32
	x, norm, q, k, v, attn, mlp, scores []float32
	velocity                            []float32
}

func (m *flux3Transformer) newState(text []float32, h, w int) (*flux3State, error) {
	if h < 1 || w < 1 || h > 128 || w > 128 || len(text) == 0 || len(text)%m.contextDim != 0 || len(text)/m.contextDim > 4096 {
		return nil, errors.New("flux3: invalid latent grid or text features")
	}
	tokens := len(text) / m.contextDim
	n := h*w + tokens
	// Six full-width slabs, RoPE, prompt projection and a tiled FFN.
	bytes := uint64(6*n*m.dim+n*m.headDim+tokens*m.dim+len(text)+2*min(n, qi21TokenTile)*m.hidden+h*w*128)*4 + 32<<20
	if bytes > 2<<30 {
		return nil, errors.New("flux3: transformer workspace exceeds 2 GiB; reduce dimensions or prompt length")
	}
	for _, f := range text {
		if !finite32(f) {
			return nil, errors.New("flux3: non-finite conditioning")
		}
	}
	s := &flux3State{model: m, imageTokens: h * w, textTokens: tokens, text: text}
	s.cosine, s.sine = flux3RoPE(h, w, tokens, m.headDim)
	return s, nil
}

func flux3RoPE(h, w, text, hd int) ([]float32, []float32) {
	pairs := hd / 2
	cosine, sine := make([]float32, (h*w+text)*pairs), make([]float32, (h*w+text)*pairs)
	axisDim := hd / 4
	for i := 0; i < h*w+text; i++ {
		pos := [4]float64{}
		if i < h*w {
			pos[1], pos[2] = float64(i/w)*0.5, float64(i%w)*0.5
		} else {
			pos[3] = float64(i - h*w)
		}
		for axis := range 4 {
			for j := 0; j < axisDim/2; j++ {
				a := pos[axis] / math.Pow(10000, float64(2*j)/float64(axisDim))
				k := i*pairs + axis*(axisDim/2) + j
				sn, cs := math.Sincos(a)
				sine[k], cosine[k] = float32(sn), float32(cs)
			}
		}
	}
	return cosine, sine
}

func (s *flux3State) rotate(x []float32, start int) {
	d, hd := s.model.dim, s.model.headDim
	for i := 0; i < len(x)/d; i++ {
		for head := 0; head < d; head += hd {
			for j := 0; j < hd/2; j++ {
				k := (start+i)*(hd/2) + j
				p := i*d + head + 2*j
				a, b := x[p], x[p+1]
				x[p], x[p+1] = a*s.cosine[k]-b*s.sine[k], a*s.sine[k]+b*s.cosine[k]
			}
		}
	}
}

func flux3Time(ctx context.Context, t float32, in, out Weight) ([]float32, error) {
	embed := make([]float32, 256)
	for j := range 128 {
		freq := float32(math.Exp(-math.Log(10000) * float64(j) / 128))
		a := float64(t * 1000 * freq)
		embed[j], embed[j+128] = float32(math.Cos(a)), float32(math.Sin(a))
	}
	x := make([]float32, in.Rows)
	if err := qi21Project(ctx, in, embed, x); err != nil {
		return nil, err
	}
	for i := range x {
		x[i] = siluExact(x[i])
	}
	result := make([]float32, out.Rows)
	if err := qi21Project(ctx, out, x, result); err != nil {
		return nil, err
	}
	return result, nil
}

func flux3Modulate(ctx context.Context, w Weight, vec []float32) ([]float32, error) {
	x := make([]float32, len(vec))
	for i, v := range vec {
		x[i] = siluExact(v)
	}
	out := make([]float32, w.Rows)
	if err := qi21Project(ctx, w, x, out); err != nil {
		return nil, err
	}
	return out, nil
}

func flux3CompactSwiGLU(combined []float32, width int) []float32 {
	out := combined[:len(combined)/2]
	for i := 0; i < len(out)/width; i++ {
		row := combined[i*2*width : (i+1)*2*width]
		// Compact in row order: the destination is either the gate itself
		// (row zero) or lies entirely before this row's unread inputs.
		siluMulF32(row[:width], row[width:], out[i*width:(i+1)*width])
	}
	return out
}

func (s *flux3State) block(ctx context.Context, b *flux3Block, begin, end int, mods [2][]float32) error {
	m, d, n := s.model, s.model.dim, end-begin
	x := s.x[begin*d : end*d]
	norm, q, k, v, attn := s.norm[:n*d], s.q[:n*d], s.k[:n*d], s.v[:n*d], s.attn[:n*d]
	for i := range n {
		mode := 0
		if begin+i >= s.imageTokens {
			mode = 1
		}
		mod := mods[mode]
		row := norm[i*d : (i+1)*d]
		layerNormInto(x[i*d:(i+1)*d], nil, nil, 1e-6, &row)
		for j := range row {
			row[j] = row[j]*(1+mod[d+j]) + mod[j]
		}
	}
	for _, p := range []struct {
		w   Weight
		out []float32
	}{{b.q, q}, {b.k, k}, {b.v, v}} {
		if err := qi21Project(ctx, p.w, norm, p.out); err != nil {
			return err
		}
	}
	for i := range n {
		perHeadRMSNormInPlace(q[i*d:(i+1)*d], m.headDim, m.heads, b.qNorm, 1e-6)
		perHeadRMSNormInPlace(k[i*d:(i+1)*d], m.headDim, m.heads, b.kNorm, 1e-6)
	}
	s.rotate(q, begin)
	s.rotate(k, begin)
	imageAttention(ctx, q, k, v, attn, n, n, m.heads, m.headDim, 0, 0, &s.scores)
	// Attention has finished reading Q, so its slab can hold the projection.
	projected := q
	if err := qi21Project(ctx, b.out, attn, projected); err != nil {
		return err
	}
	for first := 0; first < n; {
		count := qi21BatchRows(n-first, d)
		combined := s.mlp[:count*2*m.hidden]
		if err := qi21Project(ctx, b.mlpIn, norm[first*d:(first+count)*d], combined); err != nil {
			return err
		}
		hidden := flux3CompactSwiGLU(combined, m.hidden)
		if err := qi21Project(ctx, b.mlpOut, hidden, attn[first*d:(first+count)*d]); err != nil {
			return err
		}
		first += count
	}
	for i := range n {
		mode := 0
		if begin+i >= s.imageTokens {
			mode = 1
		}
		gate := mods[mode][2*d:]
		for j := range d {
			p := i*d + j
			x[p] += gate[j] * (projected[p] + attn[p])
		}
	}
	return ctx.Err()
}

func (s *flux3State) predict(ctx context.Context, latents []float32, t, next float32) ([]float32, error) {
	m, d := s.model, s.model.dim
	if len(latents) != s.imageTokens*128 || !finite32(t) || !finite32(next) || t < 0 || t > 1 || next < 0 || next > t {
		return nil, errors.New("flux3: invalid latents or timestep")
	}
	for _, v := range latents {
		if !finite32(v) {
			return nil, errors.New("flux3: non-finite latents")
		}
	}
	n := s.imageTokens + s.textTokens
	for _, p := range []*[]float32{&s.x, &s.norm, &s.q, &s.k, &s.v, &s.attn} {
		ensureLenNoClear(p, n*d)
	}
	ensureLenNoClear(&s.mlp, min(n, qi21TokenTile)*2*m.hidden)
	ensureLenNoClear(&s.velocity, len(latents))
	if err := qi21Project(ctx, m.image, latents, s.x[:s.imageTokens*d]); err != nil {
		return nil, err
	}
	if s.textProjection == nil {
		projected := make([]float32, s.textTokens*d)
		if err := qi21Project(ctx, m.text, s.text, projected); err != nil {
			return nil, err
		}
		s.textProjection, s.text = projected, nil
	}
	copy(s.x[s.imageTokens*d:], s.textProjection)
	var early, single [2][]float32
	var final []float32
	var ref []float32
	var err error
	if m.flowMap {
		ref, err = flux3Time(ctx, next, m.refIn, m.refOut)
		if err != nil {
			return nil, err
		}
	}
	for mode, time := range []float32{t, 0} {
		vec, e := flux3Time(ctx, time, m.timeIn, m.timeOut)
		if e != nil {
			return nil, e
		}
		if m.flowMap {
			addInPlace(vec, ref)
		}
		early[mode], err = flux3Modulate(ctx, m.earlyMod[mode], vec)
		if err != nil {
			return nil, err
		}
		single[mode], err = flux3Modulate(ctx, m.singleMod[mode], vec)
		if err != nil {
			return nil, err
		}
		if mode == 0 {
			final, err = flux3Modulate(ctx, m.finalMod, vec)
			if err != nil {
				return nil, err
			}
		}
	}
	for mode, blocks := range m.early {
		begin, end := 0, s.imageTokens
		if mode == 1 {
			begin, end = s.imageTokens, n
		}
		for i := range blocks {
			if err := s.block(ctx, &blocks[i], begin, end, early); err != nil {
				return nil, err
			}
		}
	}
	for i := range m.single {
		if err := s.block(ctx, &m.single[i], 0, n, single); err != nil {
			return nil, err
		}
	}
	out := s.norm[:s.imageTokens*d]
	for i := range s.imageTokens {
		row := out[i*d : (i+1)*d]
		layerNormInto(s.x[i*d:(i+1)*d], nil, nil, 1e-6, &row)
		for j := range row {
			row[j] = row[j]*(1+final[d+j]) + final[j]
		}
	}
	if err := qi21Project(ctx, m.output, out, s.velocity); err != nil {
		return nil, err
	}
	for _, v := range s.velocity {
		if !finite32(v) {
			return nil, errors.New("flux3: non-finite prediction")
		}
	}
	return s.velocity, ctx.Err()
}
