package gopherllm

import (
	"context"
	"fmt"
	"math"
)

// This file implements the offline (non-streaming) forward pass from raw
// 16kHz mono PCM samples through the mel-spectrogram frontend, the causal
// audio encoder, and the audio-to-text adapter, producing one embedding per
// AudioLengthPerTok worth of audio in the decoder's hidden size. It does not
// decode arbitrary audio file formats. The incremental frontend and session
// live in voxtral_stream_audio.go and voxtral_realtime_session.go; see
// voxtral_realtime.go's doc comment for the overall architecture and how
// each stage's math was verified.

// voxtralDFTTables holds the precomputed real-DFT basis used to turn a
// windowed frame into a power spectrum. NFFT=400 is not a power of two, so
// this is a direct O(freq*n) DFT (as two matvecs) rather than an FFT --
// cheap enough here since it runs once per ~10ms frame, not per decoded
// token.
type voxtralDFTTables struct {
	cos, sin Weight // each Cols=NFFT, rows=NFFT/2+1 (inferred from len(F32)/Cols)
	nFreq    int
	nfft     int
}

func buildVoxtralDFTTables(nfft int) voxtralDFTTables {
	nFreq := nfft/2 + 1
	cos := make([]float32, nFreq*nfft)
	sin := make([]float32, nFreq*nfft)
	for k := range nFreq {
		w := 2 * math.Pi * float64(k) / float64(nfft)
		row := k * nfft
		for n := range nfft {
			s, c := math.Sincos(w * float64(n))
			cos[row+n] = float32(c)
			sin[row+n] = float32(-s) // conjugate: real DFT convention X[k] = sum x[n]*(cos - i*sin)
		}
	}
	return voxtralDFTTables{
		cos:   Weight{F32: cos, Cols: nfft},
		sin:   Weight{F32: sin, Cols: nfft},
		nFreq: nFreq,
		nfft:  nfft,
	}
}

// reflectPad mirrors numpy/torch "reflect" padding: the boundary sample is
// not duplicated (pad=2 on [a,b,c,d,e] gives [c,b, a,b,c,d,e, d,c]), matching
// torch.stft(center=True, pad_mode="reflect")'s default. Requires
// len(samples) > pad; real audio clips are always far longer than
// NFFT/2 (200 samples = 12.5ms at 16kHz).
func reflectPad(samples []float32, pad int) ([]float32, error) {
	n := len(samples)
	if pad <= 0 {
		return samples, nil
	}
	if n <= pad {
		return nil, fmt.Errorf("reflect-padding: %d samples is too short for pad=%d", n, pad)
	}
	out := make([]float32, n+2*pad)
	for i := range pad {
		out[i] = samples[pad-i]
		out[pad+n+i] = samples[n-2-i]
	}
	copy(out[pad:pad+n], samples)
	return out, nil
}

// computeVoxtralRealtimeMelSpectrogram implements the exact frontend from
// mistralai's reference feature extractor (verified against
// antirez/voxtral.c's python_simple_implementation.py, which is
// byte-for-byte OpenAI Whisper's log_mel_spectrogram with the one
// deliberate change a streaming/causal model needs: the log-clamp floor is
// the checkpoint's fixed Mel.GlobalLogMelMax instead of Whisper's per-clip
// dynamic maximum, since a causal model can't know a future max):
//
//	stft   = STFT(pad_reflect(samples, NFFT/2), NFFT, hop, hann_window)
//	power  = |stft[..., :-1]|^2                          // drop the last frame
//	mel    = filterbank.T @ power                        // filterbank already Slaney-normalized on disk
//	logmel = log10(max(mel, 1e-10))
//	logmel = max(logmel, GlobalLogMelMax - 8.0)
//	logmel = (logmel + 4.0) / 4.0
//
// Returns channel-major [NumMels][nFrames] flattened as mel[c*nFrames+t], the
// layout the causal conv stem consumes directly.
func computeVoxtralRealtimeMelSpectrogram(cfg VoxtralRealtimeMelConfig, window []float32, filterbank Weight, samples []float32) (mel []float32, nFrames int, err error) {
	return computeVoxtralRealtimeMelSpectrogramContext(context.Background(), cfg, window, filterbank, samples)
}

func computeVoxtralRealtimeMelSpectrogramContext(ctx context.Context, cfg VoxtralRealtimeMelConfig, window []float32, filterbank Weight, samples []float32) (mel []float32, nFrames int, err error) {
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

// causalConv1dOutputLen mirrors python_simple_implementation.py's
// causal_conv1d padding arithmetic: pad left by kernel-stride (zeros), then
// just enough zeros on the right to make the strided output cover every
// input sample under ceiling division, rather than truncating floor-style.
func causalConv1dOutputLen(length, kernel, stride int) (padLeft, padRight, outLen int) {
	padLeft = kernel - stride
	nFrames := float64(length-kernel+padLeft)/float64(stride) + 1
	outLen = int(math.Ceil(nFrames - 1e-9))
	targetLength := (outLen-1)*stride + (kernel - padLeft)
	padRight = targetLength - length
	if padRight < 0 {
		padRight = 0
	}
	return
}

// applyCausalConv1d runs one of the encoder's two conv-stem layers. x is
// channel-major [In][Length] flattened as x[i*length+t]; the result is
// channel-major [Out][outLen], GELU-activated with the tanh approximation
// (verified against antirez/voxtral.c's vox_gelu, not PyTorch's default
// exact/erf GELU -- every GELU site in this file and voxtral_realtime_decoder.go
// uses the same tanh approximation for this reason).
func applyCausalConv1d(x []float32, in, length int, conv VoxtralRealtimeEncoderConv) (out []float32, outLen int, err error) {
	return applyCausalConv1dContext(context.Background(), x, in, length, conv)
}

func applyCausalConv1dContext(ctx context.Context, x []float32, in, length int, conv VoxtralRealtimeEncoderConv) (out []float32, outLen int, err error) {
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

// standardRopeInvFreqSlice builds plain (non-YaRN) RoPE inverse frequencies
// for pairs = headDim/2, decoupled from the Config-based buildRopeInvFreq
// (the encoder/decoder here aren't loaded through the generic text-model
// path this turn).
func standardRopeInvFreqSlice(headDim int, theta float32) []float32 {
	pairs := headDim / 2
	inv := make([]float32, pairs)
	for pair := range pairs {
		i := float32(pair * 2)
		inv[pair] = 1 / float32(math.Pow(float64(theta), float64(i/float32(headDim))))
	}
	return inv
}

// EncodeAudioVoxtralRealtime runs the mel frontend, causal conv stem, and
// the NLayers-block causal audio-encoder transformer over one offline
// (complete, non-streaming) audio clip, then the adapter, producing one
// DecoderHiddenSize-wide embedding per group of Projector.DownsampleFactor
// encoder frames -- ready to be summed into the decoder's text token
// embeddings at matching sequence positions (see voxtral_realtime.go's doc
// comment, point 4).
func EncodeAudioVoxtralRealtime(cfg VoxtralRealtimeConfig, weights VoxtralRealtimeWeights, samples []float32) (audioEmbeds [][]float32, err error) {
	return EncodeAudioVoxtralRealtimeContext(context.Background(), cfg, weights, samples)
}

// EncodeAudioVoxtralRealtimeContext is the cancellable variant of EncodeAudioVoxtralRealtime.
func EncodeAudioVoxtralRealtimeContext(ctx context.Context, cfg VoxtralRealtimeConfig, weights VoxtralRealtimeWeights, samples []float32) (audioEmbeds [][]float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	mel, melFrames, err := computeVoxtralRealtimeMelSpectrogramContext(ctx, cfg.Mel, weights.MelWindow, weights.MelFilterbank, samples)
	if err != nil {
		return nil, fmt.Errorf("encoding audio: %w", err)
	}

	h, hLen, err := applyCausalConv1dContext(ctx, mel, cfg.Mel.NumMels, melFrames, weights.Encoder.Conv[0])
	if err != nil {
		return nil, fmt.Errorf("encoding audio: conv0: %w", err)
	}
	h, hLen, err = applyCausalConv1dContext(ctx, h, cfg.Encoder.DModel, hLen, weights.Encoder.Conv[1])
	if err != nil {
		return nil, fmt.Errorf("encoding audio: conv1: %w", err)
	}
	x, err := forwardVoxtralEncoderChunk(ctx, cfg, weights, h, hLen, nil)
	if err != nil {
		return nil, err
	}
	return projectAudioVoxtralRealtime(cfg, weights, x)
}

// A nil state evaluates an offline clip; a state retains only the causal
// attention window, with absolute positions independent of cache eviction.
type voxtralEncoderState struct {
	pos  int
	k, v [][][]float32
}

func forwardVoxtralEncoderChunk(ctx context.Context, cfg VoxtralRealtimeConfig, weights VoxtralRealtimeWeights, h []float32, hLen int, state *voxtralEncoderState) ([][]float32, error) {
	startPos := 0
	if state != nil {
		startPos = state.pos
		if state.k == nil {
			state.k = make([][][]float32, len(weights.Encoder.Layers))
			state.v = make([][][]float32, len(weights.Encoder.Layers))
		}
	}

	dim := cfg.Encoder.DModel
	heads := cfg.Encoder.NHeads
	headDim := cfg.Encoder.HeadDim
	// Q/K/V project DModel (1280) up to heads*headDim (2048) -- verified
	// against the real checkpoint's attn.q/k/v.weight dims [1280,2048] and
	// attn.out.weight dims [2048,1280]. Unlike most decoder-only text
	// architectures in this codebase, the audio encoder's attention width
	// is NOT tied to its residual-stream width, so qkvDim must be tracked
	// separately from dim rather than asserted equal to it.
	qkvDim := heads * headDim
	if qkvDim <= 0 {
		return nil, fmt.Errorf("encoding audio: invalid encoder heads=%d headDim=%d", heads, headDim)
	}
	n := hLen

	// h is channel-major [DModel][n]; the transformer blocks below want
	// time-major [n][DModel] rows (matching EncodeImagePixtral's convention,
	// so the same rmsNormInto/matvecBatch/DotF32/AxpyF32 helpers apply).
	x := make([][]float32, n)
	{
		flat := make([]float32, n*dim)
		for t := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			x[t] = flat[t*dim : (t+1)*dim : (t+1)*dim]
			for c := range dim {
				x[t][c] = h[c*n+t]
			}
		}
	}

	invFreq := standardRopeInvFreqSlice(headDim, cfg.Encoder.RopeTheta)
	scale := float32(1 / math.Sqrt(float64(headDim)))

	q := make([][]float32, n)
	k := make([][]float32, n)
	v := make([][]float32, n)
	attnOut := make([][]float32, n)
	outProj := make([][]float32, n)
	normed := make([][]float32, n)
	ffnGateOut := make([][]float32, n)
	ffnUpOut := make([][]float32, n)
	ffnActOut := make([][]float32, n)
	ffnDownOut := make([][]float32, n)
	for t := range n {
		q[t] = make([]float32, qkvDim)
		k[t] = make([]float32, qkvDim)
		v[t] = make([]float32, qkvDim)
		attnOut[t] = make([]float32, qkvDim)
		outProj[t] = make([]float32, dim)
		normed[t] = make([]float32, dim)
		ffnGateOut[t] = make([]float32, cfg.Encoder.FFNDim)
		ffnUpOut[t] = make([]float32, cfg.Encoder.FFNDim)
		ffnActOut[t] = make([]float32, cfg.Encoder.FFNDim)
		ffnDownOut[t] = make([]float32, dim)
	}

	var sinScratch, cosScratch []float32
	ropeSin := make([]float32, n*(headDim/2))
	ropeCos := make([]float32, n*(headDim/2))
	ropeHalf := 0
	for t := range n {
		half, nCache := prepareRopeScratch(startPos+t, headDim, headDim, invFreq, 1, &sinScratch, &cosScratch)
		ropeHalf = half
		if nCache > 0 {
			copy(ropeSin[t*half:t*half+nCache], sinScratch[:nCache])
			copy(ropeCos[t*half:t*half+nCache], cosScratch[:nCache])
		}
	}

	window := cfg.Encoder.SlidingWindow
	var kHead, vHead, scores []float32

	for li := range weights.Encoder.Layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		layer := &weights.Encoder.Layers[li]
		for t := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			rmsNormInto(x[t], layer.AttnNorm, cfg.Encoder.Epsilon, &normed[t])
		}
		voxtralMatvecBatch(layer.Q, normed, q)
		voxtralMatvecBatch(layer.K, normed, k)
		voxtralMatvecBatch(layer.V, normed, v)
		for t := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if len(layer.QB) >= qkvDim {
				addInPlace(q[t], layer.QB[:qkvDim])
			}
			if len(layer.VB) >= qkvDim {
				addInPlace(v[t], layer.VB[:qkvDim])
			}
			if ropeHalf > 0 {
				sin := ropeSin[t*ropeHalf : (t+1)*ropeHalf]
				cos := ropeCos[t*ropeHalf : (t+1)*ropeHalf]
				// GGUF Q/K (and Q bias) already use split-half row order.
				applyPreparedRope(q[t], headDim, heads, ropeHalf, ropeHalf, sin, cos, false)
				applyPreparedRope(k[t], headDim, heads, ropeHalf, ropeHalf, sin, cos, false)
			}
		}

		allK, allV := k, v
		past := 0
		if state != nil {
			past = len(state.k[li])
			allK = append(append([][]float32(nil), state.k[li]...), k...)
			allV = append(append([][]float32(nil), state.v[li]...), v...)
		}
		total := past + n
		ensureLenNoClear(&kHead, heads*total*headDim)
		ensureLenNoClear(&vHead, heads*total*headDim)
		for j := range total {
			kj, vj := allK[j], allV[j]
			for hh := range heads {
				off := hh * headDim
				base := hh*total*headDim + j*headDim
				copy(kHead[base:base+headDim], kj[off:off+headDim])
				copy(vHead[base:base+headDim], vj[off:off+headDim])
			}
		}

		if !voxtralAttentionBatch(q, kHead, vHead, attnOut, heads, headDim, past, window, scale) {
			ensureLenNoClear(&scores, total)
			for t := range n {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				lo := 0
				end := past + t
				if window > 0 && end-window+1 > 0 {
					lo = end - window + 1
				}
				for hh := range heads {
					off := hh * headDim
					query := q[t][off : off+headDim]
					kh := kHead[hh*total*headDim : (hh+1)*total*headDim]
					vh := vHead[hh*total*headDim : (hh+1)*total*headDim]
					maxScore := negMaxF32
					for j := lo; j <= end; j++ {
						s := DotF32(query, kh[j*headDim:(j+1)*headDim]) * scale
						scores[j] = s
						maxScore = max(maxScore, s)
					}
					denom := float32(0)
					for j := lo; j <= end; j++ {
						scores[j] = fastExpF32(scores[j] - maxScore)
						denom += scores[j]
					}
					out := attnOut[t][off : off+headDim]
					clear(out)
					if denom > 0 {
						inv := 1 / denom
						for j := lo; j <= end; j++ {
							AxpyF32(out, scores[j]*inv, vh[j*headDim:(j+1)*headDim])
						}
					}
				}
			}
		}
		if state != nil {
			for t := range n {
				appendVoxtralHistory(&state.k[li], k[t], window)
				appendVoxtralHistory(&state.v[li], v[t], window)
			}
		}
		voxtralMatvecBatch(layer.Out, attnOut, outProj)
		for t := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if len(layer.OutB) >= dim {
				addInPlace(outProj[t], layer.OutB[:dim])
			}
			addInPlace(x[t], outProj[t])
		}

		for t := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			rmsNormInto(x[t], layer.FFNNorm, cfg.Encoder.Epsilon, &normed[t])
		}
		voxtralMatvecBatch(layer.FFNGate, normed, ffnGateOut)
		voxtralMatvecBatch(layer.FFNUp, normed, ffnUpOut)
		for t := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			siluMulF32(ffnGateOut[t], ffnUpOut[t], ffnActOut[t])
		}
		voxtralMatvecBatch(layer.FFNDown, ffnActOut, ffnDownOut)
		for t := range n {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if len(layer.FFNDownB) >= dim {
				addInPlace(ffnDownOut[t], layer.FFNDownB[:dim])
			}
			addInPlace(x[t], ffnDownOut[t])
		}
	}

	for t := range n {
		rmsNormInto(x[t], weights.Encoder.FinalNorm, cfg.Encoder.Epsilon, &x[t])
	}

	if state != nil {
		state.pos += n
	}
	return x, nil
}

// projectAudioVoxtralRealtime is the adapter: concatenate DownsampleFactor
// consecutive encoder frames (in time order, most-recent-dims-last -- see
// voxtral_realtime.go's doc comment point 3) into one InputDim vector, then
// linear -> GELU -> linear (no biases) into the decoder's hidden size.
// Trailing encoder frames that don't fill a complete group are dropped, same
// as PyTorch's reshape-based downsample would refuse a ragged remainder.
func projectAudioVoxtralRealtime(cfg VoxtralRealtimeConfig, weights VoxtralRealtimeWeights, encoderOut [][]float32) ([][]float32, error) {
	ds := cfg.Projector.DownsampleFactor
	dim := cfg.Encoder.DModel
	if ds*dim != cfg.Projector.InputDim {
		return nil, fmt.Errorf("projecting audio: downsample_factor*DModel (%d*%d) != projector input_dim (%d)", ds, dim, cfg.Projector.InputDim)
	}
	groups := len(encoderOut) / ds
	if groups == 0 {
		return nil, fmt.Errorf("projecting audio: only %d encoder frames, need at least %d for one downsample group", len(encoderOut), ds)
	}
	grouped := make([][]float32, groups)
	concatBuf := make([]float32, cfg.Projector.InputDim)
	var hidden []float32 // sized by MatvecInto from Linear1.Rows
	for g := range groups {
		for j := range ds {
			copy(concatBuf[j*dim:(j+1)*dim], encoderOut[g*ds+j])
		}
		weights.Projector.Linear1.MatvecInto(concatBuf, &hidden)
		for i, v := range hidden {
			hidden[i] = geluTanh(v)
		}
		out := make([]float32, cfg.Decoder.HiddenSize)
		weights.Projector.Linear2.MatvecInto(hidden, &out)
		grouped[g] = out
	}
	return grouped, nil
}
