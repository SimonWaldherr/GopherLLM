package gopherllm

import (
	"context"
	"fmt"
	"math"
)

// Original per-frame matvec frontends, retained to check scheduling and scratch
// changes against every output bit, including normalization and padding.

func referenceComputeVoxtralRealtimeMelSpectrogramContext(ctx context.Context, cfg VoxtralRealtimeMelConfig, window []float32, filterbank Weight, samples []float32) (mel []float32, nFrames int, err error) {
	if len(window) != cfg.WinLength {
		return nil, 0, fmt.Errorf("mel spectrogram: window length %d, want win_length=%d", len(window), cfg.WinLength)
	}
	if cfg.WinLength != cfg.NFFT {
		return nil, 0, fmt.Errorf("mel spectrogram: this implementation assumes win_length==n_fft (got %d/%d)", cfg.WinLength, cfg.NFFT)
	}
	padded := samples
	if cfg.Center {
		padded, err = reflectPad(samples, cfg.NFFT/2)
		if err != nil {
			return nil, 0, fmt.Errorf("mel spectrogram: %w", err)
		}
	}
	if len(padded) < cfg.NFFT {
		return nil, 0, fmt.Errorf("mel spectrogram: padded input (%d samples) shorter than n_fft=%d", len(padded), cfg.NFFT)
	}
	totalFrames := (len(padded)-cfg.NFFT)/cfg.HopLength + 1
	// Whisper (and this checkpoint's frontend) discards the last centered
	// STFT frame -- see this function's doc comment.
	nFrames = totalFrames - 1
	if nFrames <= 0 {
		return nil, 0, fmt.Errorf("mel spectrogram: audio too short to produce any frames (%d samples)", len(samples))
	}

	dft := buildVoxtralDFTTables(cfg.NFFT)
	numMels := cfg.NumMels
	mel = make([]float32, numMels*nFrames)

	windowed := make([]float32, cfg.NFFT)
	cosOut := make([]float32, dft.nFreq)
	sinOut := make([]float32, dft.nFreq)
	power := make([]float32, dft.nFreq)
	melFrame := make([]float32, numMels)
	globalFloor := cfg.GlobalLogMelMax - 8.0

	for t := range nFrames {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		start := t * cfg.HopLength
		frame := padded[start : start+cfg.NFFT]
		for i, s := range frame {
			windowed[i] = s * window[i]
		}
		if cfg.PreEmphasis != 0 {
			prev := float32(0)
			for i, v := range windowed {
				windowed[i] = v - cfg.PreEmphasis*prev
				prev = v
			}
		}
		dft.cos.MatvecInto(windowed, &cosOut)
		dft.sin.MatvecInto(windowed, &sinOut)
		for k := range power {
			power[k] = cosOut[k]*cosOut[k] + sinOut[k]*sinOut[k]
		}
		filterbank.MatvecInto(power, &melFrame)
		for c := range numMels {
			v := melFrame[c]
			if v < 1e-10 {
				v = 1e-10
			}
			logv := float32(math.Log10(float64(v)))
			if logv < globalFloor {
				logv = globalFloor
			}
			mel[c*nFrames+t] = (logv + 4) / 4
		}
	}
	return mel, nFrames, nil
}

func referenceParakeetLogMelSpectrogram(cfg ParakeetPreprocessorConfig, filterbank []float32, samples []float32) (mel []float32, nFrames int, err error) {
	// Preemphasis first, on the raw signal -- NeMo applies this before
	// framing/windowing, not per-frame.
	pre := make([]float32, len(samples))
	if len(samples) > 0 {
		pre[0] = samples[0]
		for i := 1; i < len(samples); i++ {
			pre[i] = samples[i] - cfg.PreEmphasis*samples[i-1]
		}
	}

	padded, err := reflectPad(pre, cfg.NFFT/2)
	if err != nil {
		return nil, 0, err
	}
	if len(padded) < cfg.NFFT {
		return nil, 0, fmt.Errorf("parakeet mel spectrogram: padded input (%d samples) shorter than n_fft=%d", len(padded), cfg.NFFT)
	}
	nFrames = (len(padded)-cfg.NFFT)/cfg.HopLength + 1
	if nFrames <= 0 {
		return nil, 0, fmt.Errorf("parakeet mel spectrogram: audio too short to produce any frames")
	}

	window := parakeetHannSymmetric(cfg.WinLength)
	// The window is shorter than the FFT size, so NeMo (matching
	// torch.stft's win_length<n_fft behavior) centers it within a
	// zero-padded n_fft buffer rather than left-aligning it.
	padLeft := (cfg.NFFT - cfg.WinLength) / 2

	dft := buildVoxtralDFTTables(cfg.NFFT)
	numMels := cfg.NumMelBins
	mel = make([]float32, numMels*nFrames)

	frame := make([]float32, cfg.NFFT)
	cosOut := make([]float32, dft.nFreq)
	sinOut := make([]float32, dft.nFreq)
	power := make([]float32, dft.nFreq)
	melFrame := make([]float32, numMels)
	filterbankWeight := Weight{F32: filterbank}

	for t := range nFrames {
		start := t * cfg.HopLength
		for i := range frame {
			frame[i] = 0
		}
		for i, w := range window {
			frame[padLeft+i] = padded[start+i] * w
		}
		dft.cos.MatvecInto(frame, &cosOut)
		dft.sin.MatvecInto(frame, &sinOut)
		for k := range power {
			power[k] = cosOut[k]*cosOut[k] + sinOut[k]*sinOut[k]
		}
		filterbankWeight.MatvecInto(power, &melFrame)
		for c, v := range melFrame {
			mel[c*nFrames+t] = float32(math.Log(float64(v) + float64(cfg.LogZeroGuard)))
		}
	}

	// Per-feature (per-mel-channel) normalization: NeMo's FilterbankFeatures
	// normalize_type="per_feature" computes mean/std per channel over the
	// whole utterance's time axis, using the unbiased (N-1) estimator.
	for c := range numMels {
		row := mel[c*nFrames : (c+1)*nFrames]
		var sum float64
		for _, v := range row {
			sum += float64(v)
		}
		mean := sum / float64(nFrames)
		var sumSq float64
		for _, v := range row {
			d := float64(v) - mean
			sumSq += d * d
		}
		denom := float64(nFrames) - 1
		if denom < 1 {
			denom = 1
		}
		std := math.Sqrt(sumSq/denom) + 1e-5
		for i, v := range row {
			row[i] = float32((float64(v) - mean) / std)
		}
	}

	return mel, nFrames, nil
}

func referenceApplyCausalConv1dContext(ctx context.Context, x []float32, in, length int, conv VoxtralRealtimeEncoderConv) (out []float32, outLen int, err error) {
	if conv.In != in {
		return nil, 0, fmt.Errorf("causal conv1d: input has %d channels, weight expects %d", in, conv.In)
	}
	padLeft, padRight, outLen := causalConv1dOutputLen(length, conv.Kernel, conv.Stride)
	paddedLen := length + padLeft + padRight
	padded := make([]float32, in*paddedLen)
	for c := range in {
		copy(padded[c*paddedLen+padLeft:c*paddedLen+padLeft+length], x[c*length:(c+1)*length])
	}
	patch := make([]float32, in*conv.Kernel)
	out = make([]float32, conv.Out*outLen)
	for t := range outLen {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		base := t * conv.Stride
		for c := range in {
			copy(patch[c*conv.Kernel:(c+1)*conv.Kernel], padded[c*paddedLen+base:c*paddedLen+base+conv.Kernel])
		}
		for o := range conv.Out {
			v := DotF32(conv.Weight[o*in*conv.Kernel:(o+1)*in*conv.Kernel], patch)
			if len(conv.Bias) > o {
				v += conv.Bias[o]
			}
			out[o*outLen+t] = geluTanh(v)
		}
	}
	return out, outLen, nil
}
