package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// FLUX concatenates Qwen3-VL-4B hidden states 4,8,...,32, before the unused
// layers 33..36 and final RMSNorm. Padding rows participate in the DiT and cannot be
// dropped. They all have position 1 and attend only to real prompt tokens.
type flux3TextEncoder struct {
	core *qi21TextEncoder
	pad  uint32
}

func newFlux3TextEncoder(config, tokenizer []byte, source qi21WeightSource) (*flux3TextEncoder, error) {
	var raw struct {
		Text struct {
			Layers int `json:"num_hidden_layers"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(config, &raw); err != nil {
		return nil, err
	}
	if raw.Text.Layers != 36 {
		return nil, errors.New("flux3: expected the 36-layer Qwen3-VL-4B text encoder")
	}
	e, err := newQwenVLTextEncoder(config, tokenizer, source, 32)
	if err != nil {
		return nil, err
	}
	if e.config.MaxSeqLen < 4096 {
		return nil, errors.New("flux3: expected at least 4096 text encoder positions")
	}
	pad, err := e.tokenizer.Encode("<|endoftext|>")
	if err != nil || len(pad) != 1 {
		return nil, errors.New("flux3: missing Qwen padding token")
	}
	return &flux3TextEncoder{e, pad[0]}, nil
}

func (e *flux3TextEncoder) encode(ctx context.Context, prompt string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(prompt) > 64<<10 || strings.Contains(prompt, "<|image_pad|>") || strings.Contains(prompt, "<|video_pad|>") {
		return nil, errors.New("flux3: expected a text prompt of at most 64 KiB")
	}
	tokens, err := e.core.tokenizer.Encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")
	if err != nil {
		return nil, err
	}
	// Match the reference's right truncation, including the chat template.
	tokens = tokens[:min(len(tokens), 4096)]
	real := len(tokens)
	padded := min((real+79)/80*80, 4096)
	if padded > real {
		// All padded queries have identical embeddings, positions and visible
		// keys. Compute one and duplicate its eight captured states below.
		tokens = append(tokens, e.pad)
	}
	n, c := len(tokens), e.core.config
	d, qd, kd := c.Dim, c.NHeads*c.HeadDim, c.NKVHeads*c.HeadDim
	bytes := uint64(8*padded*d+3*n*d+2*n*qd+2*n*kd+n*c.HeadDim+2*min(n, qi21TokenTile)*c.HiddenDim)*4 + 4<<20
	if bytes > 1<<30 {
		return nil, errors.New("flux3: text workspace exceeds 1 GiB")
	}
	x, norm, residual := make([]float32, n*d), make([]float32, n*d), make([]float32, n*d)
	q, k, v, attn := make([]float32, n*qd), make([]float32, n*kd), make([]float32, n*kd), make([]float32, n*qd)
	gate, up := make([]float32, min(n, qi21TokenTile)*c.HiddenDim), make([]float32, min(n, qi21TokenTile)*c.HiddenDim)
	cosine, sine := flux3TextRoPE(n, real, c.HeadDim, c.RopeTheta)
	var scores []float32
	result := make([]float32, padded*8*d)
	for i, id := range tokens {
		row := x[i*d : (i+1)*d]
		e.core.weights.TokenEmbd.RowInto(int(id), d, &row)
	}
	for layer, l := range e.core.weights.Layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for i := range n {
			row := norm[i*d : (i+1)*d]
			rmsNormInto(x[i*d:(i+1)*d], l.AttnNorm, c.RMSNormEps, &row)
		}
		for _, p := range []struct {
			w   Weight
			out []float32
		}{{l.WQ, q}, {l.WK, k}, {l.WV, v}} {
			if err := qi21Project(ctx, p.w, norm, p.out); err != nil {
				return nil, err
			}
		}
		for i := range n {
			qr, kr := q[i*qd:(i+1)*qd], k[i*kd:(i+1)*kd]
			perHeadRMSNormInPlace(qr, c.HeadDim, c.NHeads, l.AttnQNorm, c.RMSNormEps)
			perHeadRMSNormInPlace(kr, c.HeadDim, c.NKVHeads, l.AttnKNorm, c.RMSNormEps)
			half := c.HeadDim / 2
			cs, sn := cosine[i*half:(i+1)*half], sine[i*half:(i+1)*half]
			applyPreparedRope(qr, c.HeadDim, c.NHeads, half, half, sn, cs, false)
			applyPreparedRope(kr, c.HeadDim, c.NKVHeads, half, half, sn, cs, false)
		}
		imageGroupedAttention(ctx, q, k, v, attn, n, real, c.NHeads, c.NKVHeads, c.HeadDim, 0, real, &scores)
		if err := qi21Project(ctx, l.WO, attn, residual); err != nil {
			return nil, err
		}
		addInPlace(x, residual)
		for i := range n {
			row := norm[i*d : (i+1)*d]
			rmsNormInto(x[i*d:(i+1)*d], l.FFNNorm, c.RMSNormEps, &row)
		}
		for start := 0; start < n; {
			count := qi21BatchRows(n-start, d)
			input := norm[start*d : (start+count)*d]
			g, u := gate[:count*c.HiddenDim], up[:count*c.HiddenDim]
			if err := qi21Project(ctx, l.W1, input, g); err != nil {
				return nil, err
			}
			if err := qi21Project(ctx, l.W3, input, u); err != nil {
				return nil, err
			}
			siluMulF32(g, u, g)
			if err := qi21Project(ctx, l.W2, g, residual[start*d:(start+count)*d]); err != nil {
				return nil, err
			}
			start += count
		}
		addInPlace(x, residual)
		if (layer+1)%4 == 0 {
			slot := (layer+1)/4 - 1
			for i := range padded {
				row := x[min(i, n-1)*d : (min(i, n-1)+1)*d]
				out := result[(i*8+slot)*d : (i*8+slot+1)*d]
				copy(out, row)
			}
		}
	}
	for _, f := range result {
		if !finite32(f) {
			return nil, fmt.Errorf("flux3: non-finite text features")
		}
	}
	return result, ctx.Err()
}

// Positions are unchanged across all encoder layers and Q/K heads. Preserve
// the reference's float64 angle calculation when preparing their shared table.
func flux3TextRoPE(tokens, real, dim int, theta float32) (cosine, sine []float32) {
	cosine, sine = make([]float32, tokens*dim/2), make([]float32, tokens*dim/2)
	for j := 0; j < dim/2; j++ {
		denom := math.Pow(float64(theta), float64(2*j)/float64(dim))
		for i := range tokens {
			position := i
			if i >= real {
				position = 1
			}
			sn, cs := math.Sincos(float64(position) / denom)
			k := i*(dim/2) + j
			cosine[k], sine[k] = float32(cs), float32(sn)
		}
	}
	return cosine, sine
}
