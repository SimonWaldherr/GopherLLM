package gopherllm

import (
	"context"
	"fmt"
	"math"
)

func (m *TTSModel) conv(ctx context.Context, p string, x ttsMatrix, dilation, padding, groups int) (ttsMatrix, error) {
	w := m.weights[p+".weight"]
	if len(w.shape) == 2 {
		w.shape = []int{w.shape[0], w.shape[1], 1}
	}
	return ttsConv1D(ctx, x, w, m.weights[p+".bias"].data, dilation, padding, groups)
}

func (m *TTSModel) norm(p string, x ttsMatrix, eps float32) ttsMatrix {
	w, b := m.weights[p+".weight"].data, m.weights[p+".bias"].data
	for t := 0; t < x.length; t++ {
		var mean float64
		for c := 0; c < x.channels; c++ {
			mean += float64(x.data[c*x.length+t])
		}
		mean /= float64(x.channels)
		var variance float64
		for c := 0; c < x.channels; c++ {
			v := float64(x.data[c*x.length+t]) - mean
			variance += v * v
		}
		variance /= float64(x.channels)
		inv := float32(1 / math.Sqrt(variance+float64(eps)))
		for c := 0; c < x.channels; c++ {
			i := c*x.length + t
			x.data[i] = (x.data[i]-float32(mean))*inv*w[c] + b[c]
		}
	}
	return x
}

func ttsAdd(x, y ttsMatrix) ttsMatrix {
	for i := range x.data {
		x.data[i] += y.data[i]
	}
	return x
}

func (m *TTSModel) encode(ctx context.Context, ids []int) (ttsMatrix, ttsMatrix, ttsMatrix, error) {
	c := m.config
	n := len(ids)
	d := c.HiddenSize
	x := ttsMatrix{data: make([]float32, d*n), channels: d, length: n}
	emb := m.weights["text_encoder.embed_tokens.weight"].data
	scale := float32(math.Sqrt(float64(d)))
	for t, id := range ids {
		for j := 0; j < d; j++ {
			x.data[j*n+t] = emb[id*d+j] * scale
		}
	}
	for i := 0; i < c.NumHiddenLayers; i++ {
		p := fmt.Sprintf("text_encoder.encoder.layers.%d", i)
		a, err := m.attention(ctx, p+".attention", x)
		if err != nil {
			return ttsMatrix{}, ttsMatrix{}, ttsMatrix{}, err
		}
		x = m.norm(p+".layer_norm", ttsAdd(a, x), c.LayerNormEps)
		h, err := m.conv(ctx, p+".feed_forward.conv_1", x, 1, (c.FFNKernelSize-1)/2, 1)
		if err != nil {
			return ttsMatrix{}, ttsMatrix{}, ttsMatrix{}, err
		}
		for j, v := range h.data {
			if v < 0 {
				h.data[j] = 0
			}
		}
		h, err = m.conv(ctx, p+".feed_forward.conv_2", h, 1, (c.FFNKernelSize-1)/2, 1)
		if err != nil {
			return ttsMatrix{}, ttsMatrix{}, ttsMatrix{}, err
		}
		x = m.norm(p+".final_layer_norm", ttsAdd(h, x), c.LayerNormEps)
	}
	stats, err := m.conv(ctx, "text_encoder.project", x, 1, 0, 1)
	if err != nil {
		return ttsMatrix{}, ttsMatrix{}, ttsMatrix{}, err
	}
	means := ttsMatrix{data: stats.data[:c.FlowSize*n], channels: c.FlowSize, length: n}
	logs := ttsMatrix{data: stats.data[c.FlowSize*n:], channels: c.FlowSize, length: n}
	return x, means, logs, nil
}

func (m *TTSModel) attention(ctx context.Context, p string, x ttsMatrix) (ttsMatrix, error) {
	q, err := m.conv(ctx, p+".q_proj", x, 1, 0, 1)
	if err != nil {
		return ttsMatrix{}, err
	}
	k, err := m.conv(ctx, p+".k_proj", x, 1, 0, 1)
	if err != nil {
		return ttsMatrix{}, err
	}
	v, err := m.conv(ctx, p+".v_proj", x, 1, 0, 1)
	if err != nil {
		return ttsMatrix{}, err
	}
	head := x.channels / m.config.NumAttentionHeads
	n := x.length
	window := m.config.WindowSize
	scale := float32(1 / math.Sqrt(float64(head)))
	for i := range q.data {
		q.data[i] *= scale
	}
	relk, relv := m.weights[p+".emb_rel_k"].data, m.weights[p+".emb_rel_v"].data
	y := ttsMatrix{data: make([]float32, len(x.data)), channels: x.channels, length: n}
	scores := make([]float32, n)
	for h := 0; h < m.config.NumAttentionHeads; h++ {
		for t := 0; t < n; t++ {
			if err := ctx.Err(); err != nil {
				return ttsMatrix{}, err
			}
			for s := 0; s < n; s++ {
				var dot, relative float32
				offset := s - t
				for j := 0; j < head; j++ {
					a := q.data[(h*head+j)*n+t]
					dot += a * k.data[(h*head+j)*n+s]
					if offset >= -window && offset <= window {
						relative += a * relk[(offset+window)*head+j]
					}
				}
				scores[s] = dot + relative
			}
			ttsSoftmax(scores)
			for j := 0; j < head; j++ {
				var sum, relative float32
				for s, a := range scores {
					sum += a * v.data[(h*head+j)*n+s]
					offset := s - t
					if offset >= -window && offset <= window {
						relative += a * relv[(offset+window)*head+j]
					}
				}
				y.data[(h*head+j)*n+t] = sum + relative
			}
		}
	}
	return m.conv(ctx, p+".out_proj", y, 1, 0, 1)
}

func ttsSoftmax(x []float32) {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - max)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}

func (m *TTSModel) dds(ctx context.Context, p string, x ttsMatrix, conditioning *ttsMatrix) (ttsMatrix, error) {
	if conditioning != nil {
		x = ttsAdd(x, *conditioning)
	}
	dilation := 1
	for i := 0; i < m.config.DepthSeparableNumLayers; i++ {
		pad := (m.config.DurationPredictorKernelSize - 1) * dilation / 2
		h, err := m.conv(ctx, fmt.Sprintf("%s.convs_dilated.%d", p, i), x, dilation, pad, x.channels)
		if err != nil {
			return ttsMatrix{}, err
		}
		h = m.norm(fmt.Sprintf("%s.norms_1.%d", p, i), h, 1e-5)
		ttsGELU(h.data)
		h, err = m.conv(ctx, fmt.Sprintf("%s.convs_pointwise.%d", p, i), h, 1, 0, 1)
		if err != nil {
			return ttsMatrix{}, err
		}
		h = m.norm(fmt.Sprintf("%s.norms_2.%d", p, i), h, 1e-5)
		ttsGELU(h.data)
		x = ttsAdd(h, x)
		dilation *= m.config.DurationPredictorKernelSize
	}
	return x, nil
}

func ttsGELU(x []float32) {
	for i, v := range x {
		x[i] = geluExact(v)
	}
}
