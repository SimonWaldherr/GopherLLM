package gopherllm

import "testing"

// buildBenchParakeetWeights hand-builds a FastConformer encoder at the real
// nvidia/parakeet-tdt-0.6b-v3 checkpoint's own dimensions (extracted via
// loadParakeetConfig against the real GGUF: DModel=1024, DFF=4096, NHeads=8,
// HeadDim=128, ConvKernelSize=9, SubsamplingConvChans=256, NumMelBins=128,
// NFFT=512, WinLength=400, HopLength=160, PosEncMaxLen=9999) rather than a
// toy size, so BenchmarkParakeetEncode and its per-stage benchmarks below
// measure the same matrix shapes production traffic hits -- the whole point
// of comparing before/after an optimization pass. Only the encoder's own
// weights are populated (decoder/joint/vocab are untouched by ParakeetEncode).
func buildBenchParakeetWeights(nLayers int) (ParakeetConfig, ParakeetWeights) {
	const (
		numMelBins = 128
		nfft       = 512
		winLen     = 400
		hop        = 160

		dModel  = 1024
		dff     = 4096
		heads   = 8
		headDim = 128
		kernel  = 9
		chans   = 256

		posEncMaxLen = 9999
	)

	fill := func(n, seed int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*7+seed)%13-6) / 20
		}
		return v
	}
	fillW := func(rows, cols, seed int) Weight { return Weight{F32: fill(rows*cols, seed)} }
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	zeros := func(n int) []float32 { return make([]float32, n) }

	cfg := ParakeetConfig{
		Preprocessor: ParakeetPreprocessorConfig{
			SampleRate: 16000, NumMelBins: numMelBins, NFFT: nfft, WinLength: winLen, HopLength: hop,
			PreEmphasis: 0.97, LogZeroGuard: 1.0 / (1 << 24),
		},
		Encoder: ParakeetEncoderConfig{
			DModel: dModel, DFF: dff, NHeads: heads, HeadDim: headDim, NLayers: nLayers,
			ConvKernelSize: kernel, SubsamplingConvChans: chans, SubsamplingFactor: 8, Epsilon: 1e-5,
		},
	}

	w := ParakeetWeights{
		MelFilterbank: voxtralSlaneyMelFilterbank(cfg.Preprocessor.SampleRate, nfft, numMelBins, 0, float64(cfg.Preprocessor.SampleRate)/2),
	}

	convSpecs := []struct{ outCh, inPerGroup int }{
		{chans, 1},
		{chans, 1},
		{chans, chans},
		{chans, 1},
		{chans, chans},
	}
	for i, spec := range convSpecs {
		KH, KW, sh, sw, ph, pw := 3, 3, 2, 2, 1, 1
		if spec.inPerGroup == chans { // pointwise 1x1 stride-1 stage
			KH, KW, sh, sw, ph, pw = 1, 1, 1, 1, 0, 0
		}
		w.Encoder.SubsamplingConvs[i] = ParakeetSubsamplingConv{
			Weight:      fill(spec.outCh*spec.inPerGroup*KH*KW, 100+i),
			Bias:        fill(spec.outCh, 200+i),
			OutChannels: spec.outCh, InPerGroup: spec.inPerGroup,
			KH: KH, KW: KW, StrideH: sh, StrideW: sw, PadH: ph, PadW: pw,
		}
	}
	// 3 stride-2 stages over NumMelBins=128 -> 64 -> 32 -> 16 (the pointwise
	// 1x1 stride-1 stages don't change the mel axis), matching NeMo's own
	// dw_striding arithmetic (parakeetConv2D computes this identically at
	// runtime; hardcoded here only to size SubsamplingOut up front).
	finalMelWidth := 16
	flatDim := chans * finalMelWidth
	w.Encoder.SubsamplingOut = fillW(dModel, flatDim, 300)
	w.Encoder.SubsamplingOutBias = fill(dModel, 301)
	w.Encoder.PosEnc = fill(posEncMaxLen*dModel, 400)
	w.Encoder.PosEncMaxLen = posEncMaxLen

	for l := 0; l < nLayers; l++ {
		seed := 1000 * (l + 1)
		w.Encoder.Layers = append(w.Encoder.Layers, ParakeetEncoderLayer{
			NormFeedForward1Weight: ones(dModel), NormFeedForward1Bias: zeros(dModel),
			FeedForward1Linear1: fillW(dff, dModel, seed+1), FeedForward1Linear2: fillW(dModel, dff, seed+2),
			NormConvWeight: ones(dModel), NormConvBias: zeros(dModel),
			Conv: ParakeetEncoderConvLayer{
				PointwiseConv1:  fillW(2*dModel, dModel, seed+3),
				DepthwiseConv:   fill(dModel*kernel, seed+4),
				BatchNormWeight: ones(dModel),
				BatchNormBias:   zeros(dModel),
				BatchNormMean:   zeros(dModel),
				BatchNormVar:    ones(dModel),
				PointwiseConv2:  fillW(dModel, dModel, seed+5),
			},
			NormSelfAttWeight: ones(dModel), NormSelfAttBias: zeros(dModel),
			SelfAttn: ParakeetEncoderAttention{
				LinearQ: fillW(dModel, dModel, seed+6), LinearK: fillW(dModel, dModel, seed+7),
				LinearV: fillW(dModel, dModel, seed+8), LinearOut: fillW(dModel, dModel, seed+9),
				LinearPos: fillW(dModel, dModel, seed+10),
				PosBiasU:  fill(dModel, seed+11), PosBiasV: fill(dModel, seed+12),
			},
			NormFeedForward2Weight: ones(dModel), NormFeedForward2Bias: zeros(dModel),
			FeedForward2Linear1: fillW(dff, dModel, seed+13), FeedForward2Linear2: fillW(dModel, dff, seed+14),
			NormOutWeight: ones(dModel), NormOutBias: zeros(dModel),
		})
	}
	return cfg, w
}

// benchTone synthesizes seconds of a 440Hz PCM16 tone directly as float32
// samples (no WAV framing needed -- callers here call ParakeetEncode
// directly, not through the WAV-decoding transcription entry points).
func benchTone(seconds float64) []float32 {
	const sr = 16000
	n := int(seconds * sr)
	samples := make([]float32, n)
	for i := range samples {
		v := int16(3000.0 * sinApprox(2*3.14159265*440*float64(i)/sr))
		samples[i] = float32(v) / 32768.0
	}
	return samples
}

// BenchmarkParakeetEncode covers the full FastConformer encoder (mel
// frontend, dw_striding subsampling, all NLayers Conformer blocks) at the
// real checkpoint's 24-layer depth, on a 7-second synthetic clip -- close to
// the two real fixtures used elsewhere in this package's tests. Compare
// ns/op and B/op across a change the same way BenchmarkVoxtralEncode does
// for the Voxtral encoder.
func BenchmarkParakeetEncode(b *testing.B) {
	cfg, w := buildBenchParakeetWeights(24)
	samples := benchTone(7.0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ParakeetEncode(cfg, w, samples); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkParakeetEncoderLayer isolates one Conformer block's cost (FFN ->
// attention -> conv -> FFN) at the real T=44 frames a 7-second clip
// subsamples to, without paying the mel/subsampling cost on every
// iteration -- the more precise signal for changes scoped to the encoder
// block itself.
func BenchmarkParakeetEncoderLayer(b *testing.B) {
	cfg, w := buildBenchParakeetWeights(1)
	layer := &w.Encoder.Layers[0]
	const t = 44
	xs := make([][]float32, t)
	for i := range xs {
		xs[i] = make([]float32, cfg.Encoder.DModel)
		for j := range xs[i] {
			xs[i][j] = float32((i*31+j*7)%23-11) / 17
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		xs2 := parakeetFeedForwardBatch(xs, layer.NormFeedForward1Weight, layer.NormFeedForward1Bias, layer.FeedForward1Linear1, layer.FeedForward1Linear2, cfg.Encoder.Epsilon)
		xs3, err := parakeetRelPosSelfAttention(xs2, layer.SelfAttn, layer.NormSelfAttWeight, layer.NormSelfAttBias, cfg.Encoder.NHeads, cfg.Encoder.HeadDim, w.Encoder.PosEnc, w.Encoder.PosEncMaxLen, cfg.Encoder.Epsilon)
		if err != nil {
			b.Fatal(err)
		}
		xs4 := parakeetConvModule(xs3, layer.Conv, layer.NormConvWeight, layer.NormConvBias, cfg.Encoder.DModel, cfg.Encoder.ConvKernelSize, cfg.Encoder.Epsilon)
		_ = parakeetFeedForwardBatch(xs4, layer.NormFeedForward2Weight, layer.NormFeedForward2Bias, layer.FeedForward2Linear1, layer.FeedForward2Linear2, cfg.Encoder.Epsilon)
	}
}

// BenchmarkParakeetRelPosSelfAttention isolates the relative-position
// attention block alone (the DotF32/AxpyF32 rewrite's direct target) at
// T=44.
func BenchmarkParakeetRelPosSelfAttention(b *testing.B) {
	cfg, w := buildBenchParakeetWeights(1)
	layer := &w.Encoder.Layers[0]
	const t = 44
	xs := make([][]float32, t)
	for i := range xs {
		xs[i] = make([]float32, cfg.Encoder.DModel)
		for j := range xs[i] {
			xs[i][j] = float32((i*31+j*7)%23-11) / 17
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := parakeetRelPosSelfAttention(xs, layer.SelfAttn, layer.NormSelfAttWeight, layer.NormSelfAttBias, cfg.Encoder.NHeads, cfg.Encoder.HeadDim, w.Encoder.PosEnc, w.Encoder.PosEncMaxLen, cfg.Encoder.Epsilon); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkParakeetFeedForwardBatch isolates one macaron half-step FFN
// module (the blasMatvecBatch rewrite's direct target) at T=44.
func BenchmarkParakeetFeedForwardBatch(b *testing.B) {
	cfg, w := buildBenchParakeetWeights(1)
	layer := &w.Encoder.Layers[0]
	const t = 44
	xs := make([][]float32, t)
	for i := range xs {
		xs[i] = make([]float32, cfg.Encoder.DModel)
		for j := range xs[i] {
			xs[i][j] = float32((i*31+j*7)%23-11) / 17
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = parakeetFeedForwardBatch(xs, layer.NormFeedForward1Weight, layer.NormFeedForward1Bias, layer.FeedForward1Linear1, layer.FeedForward1Linear2, cfg.Encoder.Epsilon)
	}
}
