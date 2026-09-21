package gopherllm

import (
	"context"
	"testing"
)

// buildBenchVoxtralRealtimeWeights hand-builds a Voxtral Realtime encoder at
// the real mistralai/Voxtral-Mini-4B-Realtime-2602 checkpoint's own
// dimensions (DModel=1280, FFNDim=5120, NHeads=32, HeadDim=64,
// SlidingWindow=750, NumMels=128), unlike buildTinyVoxtralRealtimeWeights
// (voxtral_realtime_audio_test.go), which is deliberately tiny for fast
// correctness tests. BenchmarkVoxtralEncode below needs the real shapes:
// that's what blasMatvecBatch's Accelerate GEMM path (and the portable
// matvecBatch fallback) actually get called with in production, and a
// too-small benchmark can hide or invert a real regression (e.g. GEMM
// call overhead dominating at tiny sizes when the opposite is true at
// real ones). The decoder is left empty; ParakeetEncode's counterpart
// benchmark similarly only builds the pieces its own encode path touches.
func buildBenchVoxtralRealtimeWeights(nLayers int) (VoxtralRealtimeConfig, VoxtralRealtimeWeights) {
	const (
		numMels = 128
		nfft    = 400
		hop     = 160
		winLen  = 400

		dModel  = 1280
		ffnDim  = 5120
		heads   = 32
		headDim = 64

		downsample   = 4
		decHidden    = 3072
		projInputDim = dModel * downsample
	)

	fill := func(n, seed int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*7+seed)%13-6) / 20
		}
		return v
	}
	fillWeight := func(rows, cols, seed int) Weight { return Weight{F32: fill(rows*cols, seed)} }
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}

	cfg := VoxtralRealtimeConfig{
		Mel: VoxtralRealtimeMelConfig{
			SampleRate: 16000, NumMels: numMels, NFFT: nfft, HopLength: hop, WinLength: winLen,
			FMin: 0, FMax: 8000, Center: true, PadMode: "reflect", Normalize: "global",
			MelNorm: "slaney", GlobalLogMelMax: 1.5,
		},
		Encoder: VoxtralRealtimeEncoderConfig{
			DModel: dModel, FFNDim: ffnDim, HeadDim: headDim, NHeads: heads, NKVHeads: heads,
			NLayers: nLayers, NumMelBins: numMels, SlidingWindow: 750, RopeTheta: 10000, Epsilon: 1e-5,
		},
		Projector: VoxtralRealtimeProjectorConfig{
			InputDim: projInputDim, DownsampleFactor: downsample, AudioLengthPerTok: hop * 2 * downsample,
		},
		Decoder: VoxtralRealtimeDecoderConfig{HiddenSize: decHidden},
	}

	nfreq := nfft/2 + 1
	w := VoxtralRealtimeWeights{
		MelWindow:     fill(winLen, 1),
		MelFilterbank: Weight{F32: fill(numMels*nfreq, 2), Cols: nfreq},
	}
	w.Encoder.Conv[0] = VoxtralRealtimeEncoderConv{Weight: fill(3*numMels*dModel, 3), Bias: fill(dModel, 4), Kernel: 3, In: numMels, Out: dModel, Stride: 1}
	w.Encoder.Conv[1] = VoxtralRealtimeEncoderConv{Weight: fill(3*dModel*dModel, 5), Bias: fill(dModel, 6), Kernel: 3, In: dModel, Out: dModel, Stride: 2}
	w.Encoder.FinalNorm = ones(dModel)
	qkv := heads * headDim
	for l := 0; l < nLayers; l++ {
		seed := 1000 * (l + 1)
		w.Encoder.Layers = append(w.Encoder.Layers, VoxtralRealtimeEncoderLayer{
			AttnNorm: ones(dModel),
			Q:        fillWeight(qkv, dModel, seed+1), K: fillWeight(qkv, dModel, seed+2), V: fillWeight(qkv, dModel, seed+3), Out: fillWeight(dModel, qkv, seed+4),
			VB: fill(qkv, seed+5), OutB: fill(dModel, seed+6),
			FFNNorm: ones(dModel),
			FFNGate: fillWeight(ffnDim, dModel, seed+7), FFNUp: fillWeight(ffnDim, dModel, seed+8), FFNDown: fillWeight(dModel, ffnDim, seed+9),
			FFNDownB: fill(dModel, seed+10),
		})
	}
	w.Projector.Linear1 = fillWeight(decHidden, projInputDim, 2001)
	w.Projector.Linear2 = fillWeight(decHidden, decHidden, 2002)
	return cfg, w
}

// benchVoxtralTone synthesizes seconds of a 440Hz tone as float32 samples.
func benchVoxtralTone(seconds float64) []float32 {
	const sr = 16000
	n := int(seconds * sr)
	samples := make([]float32, n)
	for i := range samples {
		v := int16(3000.0 * sinApprox(2*3.14159265*440*float64(i)/sr))
		samples[i] = float32(v) / 32768.0
	}
	return samples
}

// BenchmarkVoxtralEncode covers the full offline encoder (mel frontend,
// causal conv stem, all NLayers transformer blocks, adapter projection) at
// the real checkpoint's 32-layer depth, on a 7-second synthetic clip -- the
// same shape a real transcription request hits. Weights here are built
// already-F32 (buildBenchVoxtralRealtimeWeights), i.e. already in the state
// prepareVoxtralStreamWeights produces, so this benchmarks blasMatvecBatch's
// fast path specifically; it would not have caught
// TranscribeVoxtralRealtime's missing-prepare regression (see that
// function's fix), which only manifests when the weights are still
// quantized. Guards against a regression in the fast path itself, e.g. a
// change that accidentally defeats Accelerate's GEMM batching.
func BenchmarkVoxtralEncode(b *testing.B) {
	cfg, w := buildBenchVoxtralRealtimeWeights(32)
	samples := benchVoxtralTone(7.0)
	b.ReportAllocs()
	b.ResetTimer()
	ctx := context.Background()
	for i := 0; i < b.N; i++ {
		if _, err := EncodeAudioVoxtralRealtimeContext(ctx, cfg, w, samples); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVoxtralEncoderLayer isolates one transformer block's cost at the
// real n=372 timesteps a 7-second clip's conv stem produces, without paying
// the mel/conv-stem cost on every iteration.
func BenchmarkVoxtralEncoderLayer(b *testing.B) {
	cfg, w := buildBenchVoxtralRealtimeWeights(1)
	const n = 372
	h := make([]float32, cfg.Encoder.DModel*n)
	for i := range h {
		h[i] = float32((i*31)%23-11) / 17
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := forwardVoxtralEncoderChunk(ctx, cfg, w, h, n, nil); err != nil {
			b.Fatal(err)
		}
	}
}
