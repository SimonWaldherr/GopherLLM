package gopherllm

import (
	"context"
	"fmt"
	"math"
)

func (m *TTSModel) duration(ctx context.Context, hidden ttsMatrix, noise []float32) ([]float32, error) {
	if len(noise) != 2*hidden.length {
		return nil, fmt.Errorf("tts: invalid duration noise length")
	}
	h, err := m.conv(ctx, "duration_predictor.conv_pre", hidden, 1, 0, 1)
	if err != nil {
		return nil, err
	}
	h, err = m.dds(ctx, "duration_predictor.conv_dds", h, nil)
	if err != nil {
		return nil, err
	}
	h, err = m.conv(ctx, "duration_predictor.conv_proj", h, 1, 0, 1)
	if err != nil {
		return nil, err
	}
	x := ttsMatrix{data: append([]float32(nil), noise...), channels: 2, length: hidden.length}
	// The first convolutional duration flow is omitted in the inference graph,
	// matching the reverse composition in MMS/VITS (4, 3, 2, affine 0).
	for i := m.config.DurationPredictorNumFlows; i >= 2; i-- {
		ttsFlipChannels(x)
		p := fmt.Sprintf("duration_predictor.flows.%d", i)
		first := ttsMatrix{data: x.data[:x.length], channels: 1, length: x.length}
		z, e := m.conv(ctx, p+".conv_pre", first, 1, 0, 1)
		if e != nil {
			return nil, e
		}
		z, e = m.dds(ctx, p+".conv_dds", z, &h)
		if e != nil {
			return nil, e
		}
		z, e = m.conv(ctx, p+".conv_proj", z, 1, 0, 1)
		if e != nil {
			return nil, e
		}
		bins := m.config.DurationPredictorFlowBins
		scale := float32(math.Sqrt(float64(m.config.HiddenSize)))
		widths, heights, derivatives := make([]float32, bins), make([]float32, bins), make([]float32, bins-1)
		for t := 0; t < x.length; t++ {
			for j := 0; j < bins; j++ {
				widths[j] = z.data[j*z.length+t] / scale
				heights[j] = z.data[(bins+j)*z.length+t] / scale
			}
			for j := 0; j < bins-1; j++ {
				derivatives[j] = z.data[(2*bins+j)*z.length+t]
			}
			y, e := ttsInverseSpline(x.data[x.length+t], widths, heights, derivatives, m.config.DurationPredictorTailBound)
			if e != nil {
				return nil, e
			}
			x.data[x.length+t] = y
		}
	}
	ttsFlipChannels(x)
	shift, scale := m.weights["duration_predictor.flows.0.translate"].data, m.weights["duration_predictor.flows.0.log_scale"].data
	for c := 0; c < 2; c++ {
		s := float32(math.Exp(float64(-scale[c])))
		for t := 0; t < x.length; t++ {
			i := c*x.length + t
			x.data[i] = (x.data[i] - shift[c]) * s
		}
	}
	return x.data[:x.length], nil
}

// ttsInverseSpline inverts the monotonic rational-quadratic duration flow.
// Tail derivatives are exactly one, making the transform the identity outside
// [-bound,bound]. Parameters follow the MMS/nflows minimum-bin conventions.
func ttsInverseSpline(value float32, widthLogits, heightLogits, derivativeLogits []float32, bound float32) (float32, error) {
	n := len(widthLogits)
	if n < 2 || n > 64 || len(heightLogits) != n || len(derivativeLogits) != n-1 || !finite32(value) || !finite32(bound) || bound <= 0 {
		return 0, fmt.Errorf("tts: invalid spline input")
	}
	if value < -bound || value > bound {
		return value, nil
	}
	widths, heights := append([]float32(nil), widthLogits...), append([]float32(nil), heightLogits...)
	for _, a := range [][]float32{widths, heights, derivativeLogits} {
		for _, v := range a {
			if !finite32(v) {
				return 0, fmt.Errorf("tts: non-finite spline parameter")
			}
		}
	}
	ttsSoftmax(widths)
	ttsSoftmax(heights)
	cw, ch := make([]float32, n+1), make([]float32, n+1)
	cw[0], ch[0] = -bound, -bound
	var sw, sh float32
	for i := 0; i < n; i++ {
		sw += .001 + (1-.001*float32(n))*widths[i]
		sh += .001 + (1-.001*float32(n))*heights[i]
		cw[i+1] = 2*bound*sw - bound
		ch[i+1] = 2*bound*sh - bound
	}
	cw[n], ch[n] = bound, bound
	bin := 0
	for bin < n-1 && value >= ch[bin+1] {
		bin++
	}
	w, h := cw[bin+1]-cw[bin], ch[bin+1]-ch[bin]
	delta := h / w
	derivative := func(i int) float32 {
		if i == 0 || i == n {
			return 1
		}
		v := derivativeLogits[i-1]
		if v > 20 {
			return .001 + v
		}
		return .001 + float32(math.Log1p(math.Exp(float64(v))))
	}
	d0, d1 := derivative(bin), derivative(bin+1)
	u := value - ch[bin]
	common := d0 + d1 - 2*delta
	a := h*(delta-d0) + u*common
	b := h*d0 - u*common
	c := -delta * u
	discriminant := b*b - 4*a*c
	if discriminant < 0 || !finite32(discriminant) {
		return 0, fmt.Errorf("tts: invalid inverse spline discriminant")
	}
	denominator := -b - float32(math.Sqrt(float64(discriminant)))
	if denominator == 0 {
		return 0, fmt.Errorf("tts: singular inverse spline")
	}
	root := 2 * c / denominator
	out := root*w + cw[bin]
	if !finite32(out) {
		return 0, fmt.Errorf("tts: non-finite inverse spline output")
	}
	return out, nil
}

func ttsFlipChannels(x ttsMatrix) {
	for c := 0; c < x.channels/2; c++ {
		for t := 0; t < x.length; t++ {
			a, b := c*x.length+t, (x.channels-1-c)*x.length+t
			x.data[a], x.data[b] = x.data[b], x.data[a]
		}
	}
}

func (m *TTSModel) flow(ctx context.Context, x ttsMatrix) (ttsMatrix, error) {
	for i := m.config.PriorEncoderNumFlows - 1; i >= 0; i-- {
		ttsFlipChannels(x)
		p := fmt.Sprintf("flow.flows.%d", i)
		half := x.channels / 2
		first := ttsMatrix{data: x.data[:half*x.length], channels: half, length: x.length}
		h, err := m.conv(ctx, p+".conv_pre", first, 1, 0, 1)
		if err != nil {
			return ttsMatrix{}, err
		}
		h, err = m.wavenet(ctx, p+".wavenet", h)
		if err != nil {
			return ttsMatrix{}, err
		}
		h, err = m.conv(ctx, p+".conv_post", h, 1, 0, 1)
		if err != nil {
			return ttsMatrix{}, err
		}
		for j, v := range h.data {
			x.data[half*x.length+j] -= v
		}
	}
	return x, nil
}

func (m *TTSModel) wavenet(ctx context.Context, p string, x ttsMatrix) (ttsMatrix, error) {
	out := ttsMatrix{data: make([]float32, len(x.data)), channels: x.channels, length: x.length}
	for i := 0; i < m.config.PriorEncoderNumWavenetLayers; i++ {
		h, err := m.conv(ctx, fmt.Sprintf("%s.in_layers.%d", p, i), x, 1, (m.config.WavenetKernelSize-1)/2, 1)
		if err != nil {
			return ttsMatrix{}, err
		}
		for j := range x.data {
			h.data[j] = float32(math.Tanh(float64(h.data[j]))) * float32(1/(1+math.Exp(float64(-h.data[len(x.data)+j]))))
		}
		h.data = h.data[:len(x.data)]
		h.channels = x.channels
		h, err = m.conv(ctx, fmt.Sprintf("%s.res_skip_layers.%d", p, i), h, 1, 0, 1)
		if err != nil {
			return ttsMatrix{}, err
		}
		if i < m.config.PriorEncoderNumWavenetLayers-1 {
			for j := range x.data {
				x.data[j] += h.data[j]
				out.data[j] += h.data[len(x.data)+j]
			}
		} else {
			for j, v := range h.data {
				out.data[j] += v
			}
		}
	}
	return out, nil
}
