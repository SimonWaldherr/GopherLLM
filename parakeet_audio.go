package gopherllm

import (
	"fmt"
	"math"
)

// This file implements Parakeet's log-mel frontend and FastConformer
// subsampling stem -- see parakeet.go's doc comment, points 1-2.
//
// The mel-spectrogram computation follows NVIDIA NeMo's
// FilterbankFeatures (nemo/collections/asr/parts/preprocessing/features.py,
// fetched and inspected directly, not guessed): preemphasis 0.97, a
// symmetric (non-periodic) Hann window of WinLength samples zero-padded to
// NFFT (centered), Slaney mel filterbank (same formula as Voxtral's,
// reused via voxtralSlaneyMelFilterbank -- this is a generic Slaney
// filterbank, not something Voxtral-specific despite the file it lives in),
// natural-log with an additive zero guard (2^-24), then per-utterance,
// per-mel-channel normalization (subtract the channel's mean over time,
// divide by its std + 1e-5). Dither is a training-only augmentation (NeMo
// gates it on self.training) and is not applied here.

// parakeetHannSymmetric computes a non-periodic (symmetric) Hann window --
// torch.hann_window(n, periodic=False): w[i] = 0.5 - 0.5*cos(2*pi*i/(n-1)).
// Distinct from voxtralPeriodicHannWindow (2*pi*i/n): Parakeet's checkpoint
// (asr.preprocessor.hann_periodic=false) uses the symmetric form, Voxtral's
// (verified byte-exact against a shipped window tensor) uses periodic.
func parakeetHannSymmetric(n int) []float32 {
	w := make([]float32, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1)))
	}
	return w
}

// parakeetLogMelSpectrogram computes NeMo-convention log-mel features for
// one utterance. Returns channel-major [NumMelBins][nFrames] flattened as
// mel[c*nFrames+t], matching the layout the subsampling conv stem consumes
// (mirroring computeVoxtralRealtimeMelSpectrogramContext's own return
// convention).
func parakeetLogMelSpectrogram(cfg ParakeetPreprocessorConfig, filterbank []float32, samples []float32) (mel []float32, nFrames int, err error) {
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

// parakeetConv2D applies one Conv2d stage of the dw_striding subsampling
// stem over a [inChannels][H][W] input (row-major, W fastest), producing
// [OutChannels][outH][outW]. Groups is inChannels/conv.InPerGroup;
// inPerGroup==1 with OutChannels==inChannels means a depthwise stage,
// inPerGroup==inChannels means a regular/pointwise (groups=1) stage --
// exactly the two shapes NeMo's dw_striding subsampling uses.
func parakeetConv2D(x []float32, inChannels, h, w int, conv ParakeetSubsamplingConv) (out []float32, outH, outW int) {
	outH = (h+2*conv.PadH-conv.KH)/conv.StrideH + 1
	outW = (w+2*conv.PadW-conv.KW)/conv.StrideW + 1
	groups := inChannels / conv.InPerGroup
	outPerGroup := conv.OutChannels / groups
	out = make([]float32, conv.OutChannels*outH*outW)
	for oc := 0; oc < conv.OutChannels; oc++ {
		g := oc / outPerGroup
		bias := float32(0)
		if len(conv.Bias) > oc {
			bias = conv.Bias[oc]
		}
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				var sum float32
				for ic := 0; ic < conv.InPerGroup; ic++ {
					inCh := g*conv.InPerGroup + ic
					for kh := 0; kh < conv.KH; kh++ {
						ih := oh*conv.StrideH - conv.PadH + kh
						if ih < 0 || ih >= h {
							continue
						}
						for kw := 0; kw < conv.KW; kw++ {
							iw := ow*conv.StrideW - conv.PadW + kw
							if iw < 0 || iw >= w {
								continue
							}
							xv := x[inCh*h*w+ih*w+iw]
							wv := conv.Weight[oc*conv.InPerGroup*conv.KH*conv.KW+ic*conv.KH*conv.KW+kh*conv.KW+kw]
							sum += xv * wv
						}
					}
				}
				out[oc*outH*outW+oh*outW+ow] = sum + bias
			}
		}
	}
	return out, outH, outW
}

func reluInPlace(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

// parakeetSubsample runs the dw_striding conv stem over the log-mel
// features [NumMelBins][nFrames] and projects the result to DModel,
// producing one embedding per (roughly SubsamplingFactor-th) output frame,
// row-major [outT][DModel] flattened.
func parakeetSubsample(cfg ParakeetConfig, w ParakeetEncoderWeights, mel []float32, nFrames int) (out []float32, outT int, err error) {
	// The mel frontend returns channel-major [mel][time]; NeMo's conv
	// subsampling instead treats the input as a single-channel 2D image
	// [time][mel] (H=time, W=mel) -- transpose before the first conv.
	h, wDim := nFrames, cfg.Preprocessor.NumMelBins
	x := make([]float32, h*wDim)
	for c := 0; c < wDim; c++ {
		for t := 0; t < nFrames; t++ {
			x[t*wDim+c] = mel[c*nFrames+t]
		}
	}

	convs := w.SubsamplingConvs
	x, h, wDim = parakeetConv2D(x, 1, h, wDim, convs[0])
	reluInPlace(x)
	x, h, wDim = parakeetConv2D(x, convs[0].OutChannels, h, wDim, convs[1])
	x, h, wDim = parakeetConv2D(x, convs[1].OutChannels, h, wDim, convs[2])
	reluInPlace(x)
	x, h, wDim = parakeetConv2D(x, convs[2].OutChannels, h, wDim, convs[3])
	x, h, wDim = parakeetConv2D(x, convs[3].OutChannels, h, wDim, convs[4])
	reluInPlace(x)

	channels := convs[4].OutChannels
	outT = h
	flatDim := channels * wDim
	proj := Weight{F32: w.SubsamplingOut.F32}
	out = make([]float32, outT*cfg.Encoder.DModel)
	frame := make([]float32, flatDim)
	var projected []float32
	for t := 0; t < outT; t++ {
		// x is [channels][h][w]; one output timestep gathers every
		// channel's row t, concatenated over the mel axis -- matching
		// PyTorch's transpose(1,2).reshape(B,T,channels*mel) before the
		// Linear projection.
		for c := 0; c < channels; c++ {
			copy(frame[c*wDim:(c+1)*wDim], x[c*h*wDim+t*wDim:c*h*wDim+(t+1)*wDim])
		}
		proj.MatvecInto(frame, &projected)
		copy(out[t*cfg.Encoder.DModel:(t+1)*cfg.Encoder.DModel], projected)
		for i := range cfg.Encoder.DModel {
			out[t*cfg.Encoder.DModel+i] += w.SubsamplingOutBias[i]
		}
	}
	return out, outT, nil
}
