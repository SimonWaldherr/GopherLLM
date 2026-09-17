package gopherllm

import (
	"math"
	"testing"
)

// buildTinyVoxtralRealtimeWeights hand-builds a tiny but structurally
// faithful Voxtral Realtime model directly (skipping the GGUF encode/decode
// round trip, which LoadVoxtralRealtimeModel's real-checkpoint verification
// already covers -- see voxtral_realtime.go's doc comment). The encoder's
// heads*headDim (qkvDim) is deliberately kept different from DModel, the
// same way the real 4B checkpoint's 1280-wide residual stream vs. 2048-wide
// Q/K/V projection is: a regression that re-conflates the two (as an earlier
// draft of EncodeAudioVoxtralRealtime did) fails immediately here instead of
// only against the real 3.6GB file.
func buildTinyVoxtralRealtimeWeights() (VoxtralRealtimeConfig, VoxtralRealtimeWeights) {
	const (
		numMels = 8
		nfft    = 32
		hop     = 8
		winLen  = 32

		dModel   = 6
		ffnDim   = 10
		heads    = 2
		headDim  = 4 // qkvDim = heads*headDim = 8, deliberately != dModel
		encLayer = 2

		hidden       = 12
		decFFN       = 16
		decHeads     = 2
		decKVHeads   = 1
		decHeadDim   = 6 // decHeads*decHeadDim = 12 == hidden, the ordinary (tied) case
		decLayers    = 2
		vocab        = 32
		adaHidden    = 3
		downsample   = 2
		projInputDim = dModel * downsample
	)

	fill := func(n, seed int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*7+seed)%13-6) / 20
		}
		return v
	}
	fillW := func(rows, cols, seed int) Weight { return Weight{F32: fill(rows*cols, seed)} }
	fillV := func(n, seed int) []float32 { return fill(n, seed) }
	onesNorm := func(n int) []float32 {
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
			NLayers: encLayer, NumMelBins: numMels, SlidingWindow: 6, RopeTheta: 10000, Epsilon: 1e-5,
		},
		Projector: VoxtralRealtimeProjectorConfig{
			InputDim: projInputDim, DownsampleFactor: downsample, AudioLengthPerTok: hop * downsample,
		},
		Decoder: VoxtralRealtimeDecoderConfig{
			HiddenSize: hidden, IntermediateSize: decFFN, HeadDim: decHeadDim, NHeads: decHeads,
			NKVHeads: decKVHeads, NLayers: decLayers, VocabSize: vocab, SlidingWindow: 16,
			RopeTheta: 10000, Epsilon: 1e-5, MaxPositionEmbeddings: 1024, TieWordEmbeddings: true,
		},
		Time: VoxtralRealtimeTimeConfig{EmbedDim: hidden, AdaHidden: adaHidden, EmbedTheta: 10000, DefaultNumDelayTokens: 2},
	}

	nfreq := nfft/2 + 1
	w := VoxtralRealtimeWeights{
		MelWindow:     fillV(winLen, 1),
		MelFilterbank: Weight{F32: fill(numMels*nfreq, 2), Cols: nfreq},
	}
	w.Encoder.Conv[0] = VoxtralRealtimeEncoderConv{Weight: fillV(3*numMels*dModel, 3), Bias: fillV(dModel, 4), Kernel: 3, In: numMels, Out: dModel, Stride: 1}
	w.Encoder.Conv[1] = VoxtralRealtimeEncoderConv{Weight: fillV(3*dModel*dModel, 5), Bias: fillV(dModel, 6), Kernel: 3, In: dModel, Out: dModel, Stride: 2}
	w.Encoder.FinalNorm = onesNorm(dModel)
	qkv := heads * headDim
	for l := range encLayer {
		w.Encoder.Layers = append(w.Encoder.Layers, VoxtralRealtimeEncoderLayer{
			AttnNorm: onesNorm(dModel),
			Q:        fillW(qkv, dModel, 10+l), K: fillW(qkv, dModel, 11+l), V: fillW(qkv, dModel, 12+l), Out: fillW(dModel, qkv, 13+l),
			VB: fillV(qkv, 14+l), OutB: fillV(dModel, 15+l),
			FFNNorm: onesNorm(dModel),
			FFNGate: fillW(ffnDim, dModel, 16+l), FFNUp: fillW(ffnDim, dModel, 17+l), FFNDown: fillW(dModel, ffnDim, 18+l),
			FFNDownB: fillV(dModel, 19+l),
		})
	}
	w.Projector.Linear1 = fillW(hidden, projInputDim, 20)
	w.Projector.Linear2 = fillW(hidden, hidden, 21)

	w.Decoder.TokenEmbd = fillW(vocab, hidden, 22)
	w.Decoder.OutputNorm = onesNorm(hidden)
	w.Decoder.TimeEmbedInvFreq = fillV(hidden/2, 23)
	for l := range decLayers {
		w.Decoder.Layers = append(w.Decoder.Layers, VoxtralRealtimeDecoderLayer{
			AttnNorm: onesNorm(hidden),
			Q:        fillW(decHeads*decHeadDim, hidden, 30+l), K: fillW(decKVHeads*decHeadDim, hidden, 31+l),
			V: fillW(decKVHeads*decHeadDim, hidden, 32+l), O: fillW(hidden, decHeads*decHeadDim, 33+l),
			FFNNorm:    onesNorm(hidden),
			AdaLinear1: fillW(adaHidden, hidden, 34+l), AdaLinear2: fillW(hidden, adaHidden, 35+l),
			FFNGate: fillW(decFFN, hidden, 36+l), FFNUp: fillW(decFFN, hidden, 37+l), FFNDown: fillW(hidden, decFFN, 38+l),
		})
	}
	return cfg, w
}

func TestEncodeAudioVoxtralRealtimeShapesAndFinite(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()

	samples := make([]float32, 4000) // 250ms at 16kHz
	for i := range samples {
		samples[i] = float32(0.1 * math.Sin(2*math.Pi*220*float64(i)/16000))
	}

	embeds, err := EncodeAudioVoxtralRealtime(cfg, w, samples)
	if err != nil {
		t.Fatalf("EncodeAudioVoxtralRealtime: %v", err)
	}
	if len(embeds) == 0 {
		t.Fatalf("expected at least one audio embedding group, got 0")
	}
	for gi, g := range embeds {
		if len(g) != cfg.Decoder.HiddenSize {
			t.Fatalf("group %d: len=%d, want HiddenSize=%d", gi, len(g), cfg.Decoder.HiddenSize)
		}
		for i, v := range g {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("group %d value %d is %v", gi, i, v)
			}
		}
	}
}

func TestEncodeAudioVoxtralRealtimeTooShortErrors(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	if _, err := EncodeAudioVoxtralRealtime(cfg, w, make([]float32, 4)); err == nil {
		t.Fatalf("expected an error for audio far shorter than one mel frame")
	}
}

func TestCausalConv1dOutputLenMatchesReference(t *testing.T) {
	// Values hand-checked against python_simple_implementation.py's
	// causal_conv1d padding formula (see voxtral_realtime_audio.go's doc
	// comment): stride=1 is a pure causal left-pad with the output length
	// unchanged; stride=2 halves the length via ceiling division.
	if padLeft, padRight, outLen := causalConv1dOutputLen(199, 3, 1); padLeft != 2 || padRight != 0 || outLen != 199 {
		t.Fatalf("stride=1: got padLeft=%d padRight=%d outLen=%d", padLeft, padRight, outLen)
	}
	if padLeft, _, outLen := causalConv1dOutputLen(199, 3, 2); padLeft != 1 || outLen != 100 {
		t.Fatalf("stride=2: got padLeft=%d outLen=%d, want padLeft=1 outLen=100", padLeft, outLen)
	}
}
