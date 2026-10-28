package engine

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"sync"
	"testing"
)

func checkSpeechMelBits(t *testing.T, got []float32, n int, err error, want []float32, frames int) {
	t.Helper()
	if err != nil || n != frames || len(got) != len(want) {
		t.Fatalf("mel shape/error: %d/%d, %d/%d, %v", n, frames, len(got), len(want), err)
	}
	for i, value := range want {
		if math.Float32bits(got[i]) != math.Float32bits(value) {
			t.Fatalf("mel[%d]=%g, want %g", i, got[i], value)
		}
	}
}

func TestSpeechFrontendMatchesSerialMatvecs(t *testing.T) {
	for _, workers := range []int{1, 3, 8} {
		setPerformanceThreads(t, workers)
		v, p := speechFrontendConfigs()
		for _, count := range []int{513, 2401} {
			for _, silence := range []bool{false, true} {
				samples := make([]float32, count)
				if !silence {
					for i := range samples {
						samples[i] = float32(math.Sin(float64(i*i)*.017)) * .3
					}
				}
				for _, pre := range []float32{0, .97} {
					v.PreEmphasis, p.PreEmphasis = pre, pre
					for _, center := range []bool{false, true} {
						v.Center = center
						window := voxtralPeriodicHannWindow(v.NFFT)
						bank := Weight{F32: voxtralSlaneyMelFilterbank(16000, v.NFFT, v.NumMels, 0, 8000)}
						for _, raw := range []bool{false, true} {
							if raw {
								data := make([]byte, 4*len(bank.F32))
								for i, f := range bank.F32 {
									binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(f))
								}
								bank = Weight{Raw: data, Rows: v.NumMels, Cols: v.NFFT/2 + 1, Type: GGMLTypeF32}
							}
							want, frames, err := referenceComputeVoxtralRealtimeMelSpectrogramContext(context.Background(), v, window, bank, samples)
							got, n, gotErr := computeVoxtralRealtimeMelSpectrogramContext(context.Background(), v, window, bank, samples)
							// A short uncentered clip can have no retained STFT frames.
							if err != nil {
								if gotErr == nil {
									t.Fatal("lost short-audio error")
								}
								continue
							}
							checkSpeechMelBits(t, got, n, gotErr, want, frames)
						}
					}
					bank := voxtralSlaneyMelFilterbank(16000, p.NFFT, p.NumMelBins, 0, 8000)
					want, frames, err := referenceParakeetLogMelSpectrogram(p, bank, samples)
					if err != nil {
						t.Fatal(err)
					}
					got, n, err := parakeetLogMelSpectrogram(p, bank, samples)
					checkSpeechMelBits(t, got, n, err, want, frames)
				}
			}
		}
	}
}

func TestSpeechMelConcurrentScratchAndCancellation(t *testing.T) {
	setPerformanceThreads(t, 4)
	v, _ := speechFrontendConfigs()
	samples, window := benchTone(.31), voxtralPeriodicHannWindow(v.NFFT)
	bank := Weight{F32: voxtralSlaneyMelFilterbank(16000, v.NFFT, v.NumMels, 0, 8000)}
	want, frames, err := referenceComputeVoxtralRealtimeMelSpectrogramContext(context.Background(), v, window, bank, samples)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 5 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, n, err := computeVoxtralRealtimeMelSpectrogramContext(context.Background(), v, window, bank, samples)
			if err != nil || n != frames || len(got) != len(want) {
				t.Errorf("concurrent frontend: frames=%d, error=%v", n, err)
				return
			}
			for i, value := range want {
				if math.Float32bits(got[i]) != math.Float32bits(value) {
					t.Errorf("concurrent frontend differs at %d", i)
					return
				}
			}
		}()
	}
	wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := computeVoxtralRealtimeMelSpectrogramContext(ctx, v, window, bank, samples); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled frontend: %v", err)
	}
	for _, ctx := range []context.Context{context.Background(), ctx} {
		task := speechMelTask{ctx: ctx, dft: cachedSpeechDFT(v.NFFT), samples: samples,
			window: window, filterbank: bank, frames: 17, hop: v.HopLength, mel: make([]float32, 17*v.NumMels)}
		_ = task.run()
		if task.ctx != nil || task.samples != nil || task.window != nil || task.mel != nil || task.filterbank.F32 != nil || task.dft.cos.F32 != nil {
			t.Fatal("completed task retained request buffers")
		}
	}
}

func TestSpeechMelPaddedPreemphasisDoesNotCarryBetweenFrames(t *testing.T) {
	const nfft, windowLen, windowLeft, frames, hop, mels = 128, 96, 16, 7, 3, 5
	dft := cachedSpeechDFT(nfft)
	window := voxtralPeriodicHannWindow(windowLen)
	samples := randomExpertInput(windowLen+(frames-1)*hop, 4891)
	bank := Weight{F32: randomExpertInput(mels*dft.nFreq, 4892)}
	want := make([]float32, frames*mels)
	for pos := range frames {
		frame := make([]float32, nfft)
		for i, value := range window {
			frame[windowLeft+i] = samples[pos*hop+i] * value
		}
		previous := float32(0)
		for i, value := range frame {
			frame[i] = value - .97*previous
			previous = value
		}
		re, im := dft.cos.Matvec(frame), dft.sin.Matvec(frame)
		power := make([]float32, dft.nFreq)
		for i := range power {
			power[i] = re[i]*re[i] + im[i]*im[i]
		}
		copy(want[pos*mels:], bank.Matvec(power))
	}
	for _, workers := range []int{1, 3} {
		setPerformanceThreads(t, workers)
		got := make([]float32, len(want))
		task := speechMelTask{ctx: context.Background(), dft: dft, samples: samples,
			window: window, windowLeft: windowLeft, preEmphasis: .97, filterbank: bank,
			mel: got, frames: frames, hop: hop, rowMajor: true}
		if err := task.run(); err != nil {
			t.Fatal(err)
		}
		for i, value := range want {
			if math.Float32bits(got[i]) != math.Float32bits(value) {
				t.Fatalf("workers%d frame%d mel%d: %g, want %g", workers, i/mels, i%mels, got[i], value)
			}
		}
	}
}

// Real frontend dimensions with a tiny downstream encoder keep streaming
// boundary checks quick while exercising parallel 400-point frame processing.
func speechTestVoxtralWeights() (VoxtralRealtimeConfig, VoxtralRealtimeWeights) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	cfg.Mel, _ = speechFrontendConfigs()
	cfg.Encoder.NumMelBins = cfg.Mel.NumMels
	w.MelWindow = voxtralPeriodicHannWindow(cfg.Mel.WinLength)
	w.MelFilterbank = Weight{F32: voxtralSlaneyMelFilterbank(16000, cfg.Mel.NFFT, cfg.Mel.NumMels, 0, 8000)}
	c := &w.Encoder.Conv[0]
	c.In = cfg.Mel.NumMels
	c.Weight = randomExpertInput(c.Out*c.In*c.Kernel, 82736)
	return cfg, w
}

func TestSpeechStreamingFrontendBoundaries(t *testing.T) {
	setPerformanceThreads(t, 4)
	cfg, w := speechTestVoxtralWeights()
	samples := benchTone(.31)
	raw := cfg.Mel.HopLength * 2 * cfg.Projector.DownsampleFactor
	want, err := EncodeAudioVoxtralRealtime(cfg, w, padVoxtralAudioOffline(samples, raw))
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range []int{37, 320, 2600} {
		s, err := newVoxtralStreamAudio(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var got [][]float32
		for off := 0; off < len(samples); off += chunk {
			rows, err := s.push(context.Background(), w, samples[off:min(off+chunk, len(samples))], false)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, rows...)
		}
		last, err := s.push(context.Background(), w, nil, true)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, last...)
		if len(got) != len(want) {
			t.Fatalf("chunk%d frames %d/%d", chunk, len(got), len(want))
		}
		for i := range want {
			for j, value := range want[i] {
				if math.IsNaN(float64(got[i][j])) || math.Abs(float64(got[i][j]-value)) > 2e-5 {
					t.Fatalf("chunk%d row%d col%d: %g, want %g", chunk, i, j, got[i][j], value)
				}
			}
		}
	}
}

func TestSpeechCausalConvolutionMatchesSerial(t *testing.T) {
	for _, workers := range []int{1, 3, 8} {
		setPerformanceThreads(t, workers)
		for _, frames := range []int{1, 7, 97} {
			for _, stride := range []int{1, 2, 3} {
				const in, out, kernel = 33, 64, 3
				conv := VoxtralRealtimeEncoderConv{In: in, Out: out, Kernel: kernel, Stride: stride,
					Weight: randomExpertInput(out*in*kernel, 72314), Bias: randomExpertInput(out, 12341)}
				x := randomExpertInput(in*frames, 81238)
				want, length, err := referenceApplyCausalConv1dContext(context.Background(), x, in, frames, conv)
				if err != nil {
					t.Fatal(err)
				}
				got, n, err := applyCausalConv1dContext(context.Background(), x, in, frames, conv)
				checkSpeechMelBits(t, got, n, err, want, length)
			}
		}
	}
}
