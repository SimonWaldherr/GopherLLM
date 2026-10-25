package gopherllm

import (
	"context"
	"fmt"
	"math"
)

type ttsVocoderWork struct {
	conv                                             ttsConvScratch
	upsampled, sum, first, second, hidden, activated []float32
}

func ttsVocoderConvInto(ctx context.Context, weights map[string]ttsWeight, name string, x ttsMatrix, dilation, padding int, dst *[]float32, scratch *ttsConvScratch) (ttsMatrix, error) {
	w, ok := weights[name+".weight"]
	if !ok {
		return ttsMatrix{}, fmt.Errorf("tts vocoder: missing %s.weight", name)
	}
	var bias []float32
	if b, ok := weights[name+".bias"]; ok {
		bias = b.data
	}
	y, err := ttsConv1DInto(ctx, x, w, bias, dilation, padding, 1, dst, scratch)
	if err != nil {
		return ttsMatrix{}, fmt.Errorf("tts vocoder %s: %w", name, err)
	}
	return y, nil
}

func ttsLeakyReluInPlace(data []float32, slope float32) {
	for i, v := range data {
		if v < 0 {
			data[i] = v * slope
		}
	}
}

func ttsVocoderResidual(ctx context.Context, weights map[string]ttsWeight, name string, x ttsMatrix, kernel int, dilations []int, slope float32, work *ttsVocoderWork) (ttsMatrix, error) {
	h := x
	for i, dilation := range dilations {
		if err := ctx.Err(); err != nil {
			return ttsMatrix{}, err
		}
		// Preserve the residual in a separate buffer until its final addition.
		// Activate once per input value, before im2col repeats kernel offsets.
		residual := h.data
		ensureLenNoClear(&work.activated, len(residual))
		copy(work.activated, residual)
		ttsLeakyReluInPlace(work.activated, slope)
		activated := ttsMatrix{data: work.activated, channels: h.channels, length: h.length}
		var err error
		h, err = ttsVocoderConvInto(ctx, weights, fmt.Sprintf("%s.convs1.%d", name, i), activated, dilation, (kernel-1)*dilation/2, &work.hidden, &work.conv)
		if err != nil {
			return ttsMatrix{}, err
		}
		ttsLeakyReluInPlace(h.data, slope)
		dst := &work.first
		if i%2 != 0 {
			dst = &work.second
		}
		h, err = ttsVocoderConvInto(ctx, weights, fmt.Sprintf("%s.convs2.%d", name, i), h, 1, (kernel-1)/2, dst, &work.conv)
		if err != nil {
			return ttsMatrix{}, err
		}
		if len(h.data) != len(residual) {
			return ttsMatrix{}, fmt.Errorf("tts vocoder: residual shape mismatch")
		}
		for j, v := range residual {
			h.data[j] += v
		}
	}
	return h, nil
}

func ttsVocode(ctx context.Context, weights map[string]ttsWeight, cfg ttsVITSConfig, latent ttsMatrix) ([]float32, error) {
	if len(cfg.UpsampleRates) == 0 || len(cfg.UpsampleRates) != len(cfg.UpsampleKernelSizes) || len(cfg.ResblockKernelSizes) == 0 || len(cfg.ResblockKernelSizes) != len(cfg.ResblockDilationSizes) {
		return nil, fmt.Errorf("tts vocoder: invalid upsampling or residual configuration")
	}
	var work ttsVocoderWork
	h, err := ttsVocoderConvInto(ctx, weights, "decoder.conv_pre", latent, 1, 3, &work.sum, &work.conv)
	if err != nil {
		return nil, err
	}
	for stage, stride := range cfg.UpsampleRates {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ttsLeakyReluInPlace(h.data, cfg.LeakyReLUSlope)
		name := fmt.Sprintf("decoder.upsampler.%d", stage)
		weight, ok := weights[name+".weight"]
		if !ok {
			return nil, fmt.Errorf("tts vocoder: missing %s.weight", name)
		}
		var bias []float32
		if b, ok := weights[name+".bias"]; ok {
			bias = b.data
		}
		h, err = ttsConvTranspose1DInto(ctx, h, weight, bias, stride, (cfg.UpsampleKernelSizes[stage]-stride)/2, &work.upsampled, &work.conv)
		if err != nil {
			return nil, fmt.Errorf("tts vocoder %s: %w", name, err)
		}
		ensureLenNoClear(&work.sum, len(h.data))
		sum := ttsMatrix{data: work.sum, channels: h.channels, length: h.length}
		for branch, kernel := range cfg.ResblockKernelSizes {
			name := fmt.Sprintf("decoder.resblocks.%d", stage*len(cfg.ResblockKernelSizes)+branch)
			block, err := ttsVocoderResidual(ctx, weights, name, h, kernel, cfg.ResblockDilationSizes[branch], cfg.LeakyReLUSlope, &work)
			if err != nil {
				return nil, err
			}
			if branch == 0 {
				copy(sum.data, block.data)
			} else {
				if len(sum.data) != len(block.data) {
					return nil, fmt.Errorf("tts vocoder: residual branch shape mismatch")
				}
				for i, v := range block.data {
					sum.data[i] += v
				}
			}
		}
		for i := range sum.data {
			sum.data[i] /= float32(len(cfg.ResblockKernelSizes))
		}
		h = sum
	}
	// Transformers' final leaky_relu uses its default slope, not the
	// configured residual/upsampling slope (0.1 in MMS).
	ttsLeakyReluInPlace(h.data, .01)
	var waveform []float32
	h, err = ttsVocoderConvInto(ctx, weights, "decoder.conv_post", h, 1, 3, &waveform, &work.conv)
	if err != nil {
		return nil, err
	}
	if h.channels != 1 {
		return nil, fmt.Errorf("tts vocoder: expected mono output")
	}
	for i, v := range h.data {
		h.data[i] = float32(math.Tanh(float64(v)))
	}
	return h.data, ctx.Err()
}
