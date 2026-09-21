package gopherllm

import (
	"context"
	"fmt"
	"math"
)

// voxtralStreamAudio retains STFT overlap, causal convolution tails and the
// encoder's sliding attention window. Every sample/frame is evaluated once.
type voxtralStreamAudio struct {
	cfg                          VoxtralRealtimeConfig
	dft                          voxtralDFTTables
	samples                      []float32
	total, frames                int
	conv                         [2]voxtralStreamConv
	encoder                      voxtralEncoderState
	residual                     [][]float32
	windowed, re, im, power, mel []float32
}

func newVoxtralStreamAudio(cfg VoxtralRealtimeConfig) (*voxtralStreamAudio, error) {
	m := cfg.Mel
	if m.NFFT <= 0 || m.NFFT != m.WinLength || m.HopLength <= 0 || m.HopLength > m.NFFT || !m.Center || cfg.Projector.DownsampleFactor <= 0 || cfg.Encoder.SlidingWindow <= 0 {
		return nil, fmt.Errorf("unsupported Voxtral streaming frontend or attention window")
	}
	raw := m.HopLength * 2 * cfg.Projector.DownsampleFactor
	left := voxtralLeftPadTokens * raw
	return &voxtralStreamAudio{cfg: cfg, dft: buildVoxtralDFTTables(m.NFFT), samples: make([]float32, left+m.NFFT/2), total: left}, nil
}

func (s *voxtralStreamAudio) push(ctx context.Context, w VoxtralRealtimeWeights, pcm []float32, final bool) ([][]float32, error) {
	m := s.cfg.Mel
	s.samples = append(s.samples, pcm...)
	s.total += len(pcm)
	if final {
		raw := m.HopLength * 2 * s.cfg.Projector.DownsampleFactor
		padding := (raw-s.total%raw)%raw + voxtralRightPadTokens*raw
		s.samples = append(s.samples, make([]float32, padding+m.NFFT/2)...)
		s.total += padding
	}
	count := 0
	if len(s.samples) >= m.NFFT {
		count = (len(s.samples)-m.NFFT)/m.HopLength + 1
	}
	if final {
		count = min(count, s.total/m.HopLength-s.frames)
	} // drop final STFT frame
	if count <= 0 {
		return nil, nil
	}
	rows := make([][]float32, count)
	ensureLenNoClear(&s.windowed, m.NFFT)
	ensureLenNoClear(&s.power, s.dft.nFreq)
	for t := range count {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for i := range s.windowed {
			s.windowed[i] = s.samples[t*m.HopLength+i] * w.MelWindow[i]
		}
		if m.PreEmphasis != 0 {
			prev := float32(0)
			for i, v := range s.windowed {
				s.windowed[i] = v - m.PreEmphasis*prev
				prev = v
			}
		}
		s.dft.cos.MatvecInto(s.windowed, &s.re)
		s.dft.sin.MatvecInto(s.windowed, &s.im)
		for k := range s.power {
			s.power[k] = s.re[k]*s.re[k] + s.im[k]*s.im[k]
		}
		w.MelFilterbank.MatvecInto(s.power, &s.mel)
		rows[t] = make([]float32, m.NumMels)
		for c, v := range s.mel {
			log := float32(math.Log10(float64(max(v, 1e-10))))
			rows[t][c] = (max(log, m.GlobalLogMelMax-8) + 4) / 4
		}
	}
	used := count * m.HopLength
	copy(s.samples, s.samples[used:])
	s.samples = s.samples[:len(s.samples)-used]
	s.frames += count
	var err error
	for i := range s.conv {
		rows, err = s.conv[i].push(ctx, w.Encoder.Conv[i], rows)
		if err != nil {
			return nil, err
		}
	}
	if len(rows) == 0 {
		return nil, nil
	}
	// The shared batched transformer consumes channel-major convolution output.
	n, dim := len(rows), s.cfg.Encoder.DModel
	h := make([]float32, n*dim)
	for t, row := range rows {
		for c, v := range row {
			h[c*n+t] = v
		}
	}
	encoded, err := forwardVoxtralEncoderChunk(ctx, s.cfg, w, h, n, &s.encoder)
	if err != nil {
		return nil, err
	}
	all := append(s.residual, encoded...)
	ds := s.cfg.Projector.DownsampleFactor
	usable := len(all) / ds * ds
	s.residual = append([][]float32(nil), all[usable:]...)
	if usable == 0 {
		return nil, nil
	}
	return projectAudioVoxtralRealtime(s.cfg, w, all[:usable])
}

type voxtralStreamConv struct {
	initialized bool
	rows        [][]float32
}

// Stride alignment is carried across calls: no right padding is inserted at
// chunk boundaries, including when a chunk ends with an odd mel-frame count.
func (s *voxtralStreamConv) push(ctx context.Context, w VoxtralRealtimeEncoderConv, rows [][]float32) ([][]float32, error) {
	if w.Kernel < w.Stride || w.Stride <= 0 {
		return nil, fmt.Errorf("unsupported streaming convolution")
	}
	if !s.initialized {
		s.rows = make([][]float32, w.Kernel-w.Stride)
		for i := range s.rows {
			s.rows[i] = make([]float32, w.In)
		}
		s.initialized = true
	}
	s.rows = append(s.rows, rows...)
	n := 0
	if len(s.rows) >= w.Kernel {
		n = (len(s.rows)-w.Kernel)/w.Stride + 1
	}
	out := make([][]float32, n)
	patches := make([][]float32, n)
	width := w.In * w.Kernel
	flat := make([]float32, n*width)
	for t := range n {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		patches[t] = flat[t*width : (t+1)*width]
		for c := range w.In {
			for k := range w.Kernel {
				patches[t][c*w.Kernel+k] = s.rows[t*w.Stride+k][c]
			}
		}
		out[t] = make([]float32, w.Out)
	}
	blasMatvecBatch(Weight{F32: w.Weight, Rows: w.Out, Cols: width}, patches, out)
	for t := range n {
		for o := range w.Out {
			v := out[t][o]
			if o < len(w.Bias) {
				v += w.Bias[o]
			}
			out[t][o] = geluTanh(v)
		}
	}
	s.rows = append([][]float32(nil), s.rows[n*w.Stride:]...)
	return out, nil
}
