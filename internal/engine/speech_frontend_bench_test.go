package engine

import (
	"context"
	"fmt"
	"testing"
)

func speechFrontendConfigs() (VoxtralRealtimeMelConfig, ParakeetPreprocessorConfig) {
	return VoxtralRealtimeMelConfig{SampleRate: 16000, NFFT: 400, WinLength: 400, HopLength: 160,
			NumMels: 128, Center: true, GlobalLogMelMax: 1.5},
		ParakeetPreprocessorConfig{SampleRate: 16000, NFFT: 512, WinLength: 400, HopLength: 160,
			NumMelBins: 128, PreEmphasis: .97, LogZeroGuard: 1.0 / (1 << 24)}
}

func BenchmarkSpeechCausalConvolution(b *testing.B) {
	setPerformanceThreads(b, 8)
	for _, in := range []int{128, 1280} {
		b.Run(fmt.Sprint(in), func(b *testing.B) {
			const frames, out, kernel = 700, 1280, 3
			stride := 1
			if in == 1280 {
				stride = 2
			}
			conv := VoxtralRealtimeEncoderConv{In: in, Out: out, Kernel: kernel, Stride: stride,
				Weight: randomExpertInput(out*in*kernel, 72314), Bias: randomExpertInput(out, 12341)}
			x := randomExpertInput(in*frames, 81238)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, _, err := applyCausalConv1dContext(context.Background(), x, in, frames, conv); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Production mel dimensions, tiny downstream weights: isolates streaming
// frontend overhead while exercising the actual push and overlap lifecycle.
func BenchmarkSpeechStreamingFrontend(b *testing.B) {
	setPerformanceThreads(b, 8)
	cfg, w := speechTestVoxtralWeights()
	for _, size := range []int{320, 2560} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			s, err := newVoxtralStreamAudio(cfg)
			if err != nil {
				b.Fatal(err)
			}
			pcm := benchTone(float64(size) / 16000)
			if _, err := s.push(context.Background(), w, pcm, false); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := s.push(context.Background(), w, pcm, false); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Includes padding, windowing, spectra, mel projection and normalization.
// Setup/warmup is outside timing; model loading and encoding are excluded.
func BenchmarkSpeechFrontend(b *testing.B) {
	setPerformanceThreads(b, 8)
	v, p := speechFrontendConfigs()
	window := voxtralPeriodicHannWindow(v.WinLength)
	vbank := Weight{F32: voxtralSlaneyMelFilterbank(16000, v.NFFT, v.NumMels, 0, 8000)}
	pbank := voxtralSlaneyMelFilterbank(16000, p.NFFT, p.NumMelBins, 0, 8000)
	for _, clip := range []struct {
		name    string
		seconds float64
	}{{"160ms", .16}, {"1s", 1}, {"7s", 7}} {
		samples := benchTone(clip.seconds)
		for _, model := range []string{"voxtral", "parakeet"} {
			b.Run(model+"/"+clip.name, func(b *testing.B) {
				run := func() {
					var err error
					if model == "voxtral" {
						_, _, err = computeVoxtralRealtimeMelSpectrogramContext(context.Background(), v, window, vbank, samples)
					} else {
						_, _, err = parakeetLogMelSpectrogram(p, pbank, samples)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					run()
				}
			})
		}
	}
}
