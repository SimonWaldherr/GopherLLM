package gopherllm

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

func ttsLoadWeights(ctx context.Context, file *SafetensorsFile, c ttsVITSConfig) (map[string]ttsWeight, error) {
	expected := make(map[string][]int)
	add := func(name string, shape ...int) { expected[name] = shape }
	conv := func(name string, in, out, kernel int, bias bool) {
		add(name+".weight", out, in, kernel)
		if bias {
			add(name+".bias", out)
		}
	}
	norm := func(name string, n int) { add(name+".weight", n); add(name+".bias", n) }
	dds := func(p string) {
		for j := 0; j < c.DepthSeparableNumLayers; j++ {
			q := fmt.Sprintf("%s.convs_dilated.%d", p, j)
			conv(q, 1, c.HiddenSize, c.DurationPredictorKernelSize, true)
			conv(fmt.Sprintf("%s.convs_pointwise.%d", p, j), c.HiddenSize, c.HiddenSize, 1, true)
			norm(fmt.Sprintf("%s.norms_1.%d", p, j), c.HiddenSize)
			norm(fmt.Sprintf("%s.norms_2.%d", p, j), c.HiddenSize)
		}
	}
	add("text_encoder.embed_tokens.weight", c.VocabSize, c.HiddenSize)
	for i := 0; i < c.NumHiddenLayers; i++ {
		p := fmt.Sprintf("text_encoder.encoder.layers.%d", i)
		for _, name := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
			add(p+".attention."+name+".weight", c.HiddenSize, c.HiddenSize)
			add(p+".attention."+name+".bias", c.HiddenSize)
		}
		add(p+".attention.emb_rel_k", 1, 2*c.WindowSize+1, c.HiddenSize/c.NumAttentionHeads)
		add(p+".attention.emb_rel_v", 1, 2*c.WindowSize+1, c.HiddenSize/c.NumAttentionHeads)
		norm(p+".layer_norm", c.HiddenSize)
		norm(p+".final_layer_norm", c.HiddenSize)
		conv(p+".feed_forward.conv_1", c.HiddenSize, c.FFNDim, c.FFNKernelSize, true)
		conv(p+".feed_forward.conv_2", c.FFNDim, c.HiddenSize, c.FFNKernelSize, true)
	}
	conv("text_encoder.project", c.HiddenSize, 2*c.FlowSize, 1, true)
	conv("duration_predictor.conv_pre", c.HiddenSize, c.HiddenSize, 1, true)
	conv("duration_predictor.conv_proj", c.HiddenSize, c.HiddenSize, 1, true)
	dds("duration_predictor.conv_dds")
	add("duration_predictor.flows.0.translate", 2, 1)
	add("duration_predictor.flows.0.log_scale", 2, 1)
	// Flow 1 cancels out during reverse duration prediction; neither it nor
	// training-only posterior flows need resident weights.
	for i := 2; i <= c.DurationPredictorNumFlows; i++ {
		p := fmt.Sprintf("duration_predictor.flows.%d", i)
		conv(p+".conv_pre", 1, c.HiddenSize, 1, true)
		dds(p + ".conv_dds")
		conv(p+".conv_proj", c.HiddenSize, 3*c.DurationPredictorFlowBins-1, 1, true)
	}
	for i := 0; i < c.PriorEncoderNumFlows; i++ {
		p := fmt.Sprintf("flow.flows.%d", i)
		conv(p+".conv_pre", c.FlowSize/2, c.HiddenSize, 1, true)
		conv(p+".conv_post", c.HiddenSize, c.FlowSize/2, 1, true)
		for j := 0; j < c.PriorEncoderNumWavenetLayers; j++ {
			conv(fmt.Sprintf("%s.wavenet.in_layers.%d", p, j), c.HiddenSize, 2*c.HiddenSize, c.WavenetKernelSize, true)
			out := 2 * c.HiddenSize
			if j == c.PriorEncoderNumWavenetLayers-1 {
				out = c.HiddenSize
			}
			conv(fmt.Sprintf("%s.wavenet.res_skip_layers.%d", p, j), c.HiddenSize, out, 1, true)
		}
	}
	conv("decoder.conv_pre", c.FlowSize, c.UpsampleInitialChannel, 7, true)
	channels := c.UpsampleInitialChannel
	for i, k := range c.UpsampleKernelSizes {
		p := fmt.Sprintf("decoder.upsampler.%d", i)
		add(p+".weight", channels, channels/2, k)
		add(p+".bias", channels/2)
		channels /= 2
		for j, kernel := range c.ResblockKernelSizes {
			for d := range c.ResblockDilationSizes[j] {
				p := fmt.Sprintf("decoder.resblocks.%d", i*len(c.ResblockKernelSizes)+j)
				conv(fmt.Sprintf("%s.convs1.%d", p, d), channels, channels, kernel, true)
				conv(fmt.Sprintf("%s.convs2.%d", p, d), channels, channels, kernel, true)
			}
		}
	}
	conv("decoder.conv_post", channels, 1, 7, false)
	weights := make(map[string]ttsWeight, len(expected))
	read := func(name string, shape []int) (ttsWeight, error) {
		info, ok := file.Tensors[name]
		if !ok {
			return ttsWeight{}, fmt.Errorf("tts: missing tensor %s", name)
		}
		if !slices.Equal(info.Shape, shape) {
			return ttsWeight{}, fmt.Errorf("tts: %s shape %v, expected %v", name, info.Shape, shape)
		}
		// Validate before the general format reader allocates or converts data.
		n := 1
		for _, d := range shape {
			if d <= 0 || n > (64<<20)/d {
				return ttsWeight{}, fmt.Errorf("tts: invalid tensor size for %s", name)
			}
			n *= d
		}
		width := 4
		if info.DType == SafetensorsF16 || info.DType == SafetensorsBF16 {
			width = 2
		} else if info.DType != SafetensorsF32 {
			return ttsWeight{}, fmt.Errorf("tts: unsupported tensor dtype for %s", name)
		}
		if info.Numel != n || info.Begin < 0 || info.End < info.Begin || info.End-info.Begin != n*width || info.End > len(file.Data)-file.DataOffset {
			return ttsWeight{}, fmt.Errorf("tts: invalid data range for %s", name)
		}
		data, err := file.F32(name)
		if err != nil {
			return ttsWeight{}, err
		}
		if !numeric.AllFiniteF32(data) {
			return ttsWeight{}, fmt.Errorf("tts: non-finite tensor %s", name)
		}
		return ttsWeight{data: data, shape: shape}, nil
	}
	for name, shape := range expected {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		_, present := file.Tensors[name]
		if present {
			w, err := read(name, shape)
			if err != nil {
				return nil, err
			}
			weights[name] = w
			continue
		}
		// HF's MMS checkpoint stores WaveNet convolutions in weight-normalized
		// form. Fold g/||v|| once; no runtime normalization or duplicate copy.
		if len(name) < 7 || name[len(name)-7:] != ".weight" || len(shape) != 3 {
			return nil, fmt.Errorf("tts: missing tensor %s", name)
		}
		v, err := read(name+"_v", shape)
		if err != nil {
			return nil, err
		}
		g, err := read(name+"_g", []int{shape[0], 1, 1})
		if err != nil {
			return nil, err
		}
		stride := shape[1] * shape[2]
		for row := 0; row < shape[0]; row++ {
			var sum float64
			for _, x := range v.data[row*stride : (row+1)*stride] {
				sum += float64(x) * float64(x)
			}
			if sum == 0 {
				return nil, fmt.Errorf("tts: zero weight norm in %s", name)
			}
			scale := g.data[row] / float32(math.Sqrt(sum))
			if !finite32(scale) {
				return nil, fmt.Errorf("tts: non-finite weight norm in %s", name)
			}
			for j := row * stride; j < (row+1)*stride; j++ {
				v.data[j] *= scale
				if !finite32(v.data[j]) {
					return nil, fmt.Errorf("tts: non-finite normalized weight in %s", name)
				}
			}
		}
		weights[name] = v
	}
	return weights, nil
}
