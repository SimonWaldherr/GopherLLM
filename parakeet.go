package gopherllm

import "fmt"

// This file (and parakeet_audio.go, parakeet_conformer.go, parakeet_tdt.go,
// parakeet_transcribe.go) load and run NVIDIA's Parakeet-TDT: a FastConformer
// encoder feeding a Transducer (RNN-T) decoder using the TDT (Token-and-
// Duration Transducer) variant, from a GGUF whose general.architecture is
// "asr" (verified against nvidia/parakeet-tdt-0.6b-v3's community GGUF
// conversion via --list-metadata/--list-tensors; NVIDIA's own release ships
// a .nemo bundle -- a tarball of a YAML config plus a PyTorch checkpoint --
// which this loader does not read directly).
//
// This is architecturally unrelated to every other model family in this
// codebase: encoder attention uses Transformer-XL-style RELATIVE positional
// scoring (not RoPE or absolute position embeddings), the encoder's conv
// module operates on a genuine 2D [time, mel] grid for subsampling (not the
// 1D causal convs Voxtral's encoder uses), and the decoder is a small LSTM
// "prediction network" plus a joint network -- not an autoregressive
// transformer at all. See each file's own doc comment for its piece.
//
// Architecture (FastConformer encoder + TDT decoder), by tensor family:
//
//  1. Preprocessor: log-mel spectrogram, NeMo's own convention -- STFT with
//     a symmetric (non-periodic) Hann window shorter than n_fft (zero-padded
//     centered within the FFT frame), Slaney mel filterbank, natural-log
//     with an additive zero guard, then per-utterance per-mel-channel
//     normalization (mean/std over time). No learned weights.
//  2. Subsampling ("encoder.pre_encode.*"): NeMo's "dw_striding" conv stem --
//     one regular Conv2d (1->256 channels, 3x3, stride 2), then two
//     depthwise-separable stages (Conv2d groups=256, 3x3, stride 2, then a
//     1x1 pointwise Conv2d to mix channels), each pair halving both the
//     time and mel axes -- 2^3 = 8x downsampling total (subsampling_factor).
//     ReLU between stages. The result is flattened over [channels, mel] and
//     projected (pre_encode.out, a Linear) to d_model.
//  3. Encoder ("encoder.layers.N.*", 24 layers): a Conformer block --
//     half-step feed-forward (linear1 -> Swish -> linear2, residual scaled
//     0.5), relative-position multi-head self-attention (Transformer-XL
//     style: separate learned biases pos_bias_u/pos_bias_v, a "matrix_bd"
//     term built from encoder.pos_enc.pe via linear_pos, and the
//     compute_rel_shift trick to align it with absolute position pairs),
//     a convolution module (pointwise_conv1 -> GLU -> depthwise_conv ->
//     batch_norm -> Swish -> pointwise_conv2), a second half-step
//     feed-forward, then a final LayerNorm (norm_out).
//  4. Decoder ("decoder.prediction.*"): the RNN-T "prediction network" --
//     an embedding table (vocab_size+1 for the RNNT blank) feeding a
//     2-layer LSTM (dec_rnn.lstm.{ih,hh}_l{0,1}), autoregressive over
//     emitted labels only (not time), producing one hidden state per label
//     regardless of how many encoder frames it took to emit it.
//  5. Joint ("joint.*"): combines one encoder frame and the current
//     prediction-network state (joint.enc / joint.pred, each projecting to
//     a shared width), sums them, applies an activation, and projects
//     (joint.joint_net.2) to vocab_size+1+len(durations) logits: the
//     RNNT/TDT token distribution and TDT's duration distribution in one
//     shot. Greedy TDT decoding picks the best (token, duration) pair per
//     step and advances the encoder-frame cursor by duration (0 means
//     "stay and emit another token from the same frame"), which is what
//     lets TDT skip frames a plain RNN-T must visit one at a time.
//  6. Tokenizer ("asr.tokenizer.vocab" + implicit blank_id): a plain
//     SentencePiece piece list (ID -> string only; decoding a transducer's
//     output never needs the encode-side algorithm a text LLM tokenizer
//     needs, since there is no text prompt to tokenize, only token IDs to
//     render back to text) -- see parakeet_transcribe.go.

// ParakeetPreprocessorConfig is the log-mel frontend's configuration.
type ParakeetPreprocessorConfig struct {
	SampleRate   int
	NumMelBins   int
	NFFT         int
	WinLength    int // < NFFT; the Hann window is zero-padded, centered within each FFT frame
	HopLength    int
	PreEmphasis  float32
	LogZeroGuard float32
}

// ParakeetEncoderConfig is the FastConformer encoder's configuration.
type ParakeetEncoderConfig struct {
	DModel               int
	DFF                  int
	NHeads               int
	HeadDim              int
	NLayers              int
	ConvKernelSize       int
	SubsamplingConvChans int
	SubsamplingFactor    int // 2^(number of stride-2 stages); 8 for this checkpoint
	Epsilon              float32
}

// ParakeetRNNTConfig is the prediction network + joint network's
// configuration, including TDT's duration-bin extension over plain RNN-T.
type ParakeetRNNTConfig struct {
	VocabSize         int // real vocabulary size, NOT counting the blank
	BlankID           int // == VocabSize; the transducer's blank/no-emission symbol
	PredHidden        int
	PredEmbedDim      int
	PredNumLayers     int
	JointDim          int
	Durations         []int
	MaxSymbolsPerStep int
}

type ParakeetConfig struct {
	Preprocessor ParakeetPreprocessorConfig
	Encoder      ParakeetEncoderConfig
	RNNT         ParakeetRNNTConfig
}

// ParakeetEncoderConvLayer is one Conformer block's convolution module.
type ParakeetEncoderConvLayer struct {
	PointwiseConv1                                              Weight    // DModel -> 2*DModel (split into GLU gate/value)
	DepthwiseConv                                               []float32 // flat [DModel][ConvKernelSize], one 1D kernel per channel (groups==DModel)
	BatchNormWeight, BatchNormBias, BatchNormMean, BatchNormVar []float32
	PointwiseConv2                                              Weight // DModel -> DModel
}

// ParakeetEncoderAttention is one Conformer block's relative-position
// multi-head self-attention (Transformer-XL style).
type ParakeetEncoderAttention struct {
	LinearQ, LinearK, LinearV, LinearOut, LinearPos Weight
	PosBiasU, PosBiasV                              []float32 // [NHeads*HeadDim], the "u"/"v" content/position biases
}

// ParakeetEncoderLayer is one Conformer block: half-step FFN, relative
// self-attention, conv module, half-step FFN, final norm.
type ParakeetEncoderLayer struct {
	NormFeedForward1Weight, NormFeedForward1Bias []float32
	FeedForward1Linear1, FeedForward1Linear2     Weight
	NormConvWeight, NormConvBias                 []float32
	Conv                                         ParakeetEncoderConvLayer
	NormSelfAttWeight, NormSelfAttBias           []float32
	SelfAttn                                     ParakeetEncoderAttention
	NormFeedForward2Weight, NormFeedForward2Bias []float32
	FeedForward2Linear1, FeedForward2Linear2     Weight
	NormOutWeight, NormOutBias                   []float32
}

// ParakeetSubsamplingConv is one Conv2d stage of the dw_striding
// subsampling stem. Weight is flat, row-major [OutChannels][InPerGroup][KH][KW]
// (PyTorch's native Conv2d layout; GGUF's own dims order needed reversing to
// get here -- see the loader). Groups is InChannels/InPerGroup implicitly:
// InPerGroup==1 and OutChannels==InChannels for a depthwise stage, or
// InPerGroup==InChannels for a regular/pointwise stage.
type ParakeetSubsamplingConv struct {
	Weight                               []float32
	Bias                                 []float32
	OutChannels, InPerGroup              int
	KH, KW, StrideH, StrideW, PadH, PadW int
}

type ParakeetEncoderWeights struct {
	SubsamplingConvs   [5]ParakeetSubsamplingConv // conv.0 (regular), conv.2+conv.3 (depthwise+pointwise), conv.5+conv.6 (depthwise+pointwise) -- indices kept sparse to mirror the checkpoint's own nn.Sequential numbering (see parakeet_audio.go)
	SubsamplingOut     Weight                     // pre_encode.out: flattened [channels*melsAfterSubsampling] -> DModel
	SubsamplingOutBias []float32
	PosEnc             []float32 // encoder.pos_enc.pe, flat [MaxLen][DModel], relative positional encoding table
	PosEncMaxLen       int
	Layers             []ParakeetEncoderLayer
}

type ParakeetLSTMLayer struct {
	WeightIH, WeightHH Weight    // [4*Hidden, Input] / [4*Hidden, Hidden], PyTorch gate order [i,f,g,o]
	BiasIH, BiasHH     []float32 // [4*Hidden]
}

type ParakeetDecoderWeights struct {
	Embed Weight // [VocabSize+1 (blank)][PredEmbedDim]
	LSTM  []ParakeetLSTMLayer
}

type ParakeetJointWeights struct {
	Pred, Enc         Weight // project prediction-network / encoder outputs into joint space
	PredBias, EncBias []float32
	Out               Weight // joint space -> vocab+1+len(durations) logits
	OutBias           []float32
}

type ParakeetWeights struct {
	MelFilterbank []float32 // flat [NumMelBins][NFFT/2+1], computed (see voxtralSlaneyMelFilterbank reuse)
	Encoder       ParakeetEncoderWeights
	Decoder       ParakeetDecoderWeights
	Joint         ParakeetJointWeights
	Vocab         []string // ID -> SentencePiece piece text; BlankID is one past the end, has no text
}

// ParakeetArchitectureSupported reports whether a GGUF's general.architecture
// names this loader's format ("asr"), for callers that dispatch on
// architecture before committing to a specific loader the way runner.go's
// ArchitectureSupported does for the main text-model list. Kept separate
// from that list because this is a wholly different loading/inference path,
// not a Runner-integrated architecture.
func ParakeetArchitectureSupported(arch string) bool { return arch == "asr" }

func loadParakeetConfig(gguf *GGUFFile) (ParakeetConfig, error) {
	arch, _ := gguf.GetString("general.architecture")
	if arch != "asr" {
		return ParakeetConfig{}, fmt.Errorf("loading parakeet model: general.architecture is %q, want \"asr\"", arch)
	}
	headType, _ := gguf.GetString("asr.head_type")
	if headType != "tdt" {
		return ParakeetConfig{}, fmt.Errorf("loading parakeet model: asr.head_type is %q, want \"tdt\" (only the TDT transducer head is supported)", headType)
	}

	pre := ParakeetPreprocessorConfig{
		SampleRate:   int(gguf.GetU32("asr.preprocessor.sample_rate", 16000)),
		NumMelBins:   int(gguf.GetU32("asr.encoder.feat_in", 0)),
		NFFT:         int(gguf.GetU32("asr.preprocessor.n_fft", 0)),
		HopLength:    0, // derived below from window_stride*sample_rate
		PreEmphasis:  gguf.GetF32("asr.preprocessor.preemph", 0),
		LogZeroGuard: 1.0 / (1 << 24),
	}
	windowSizeS := gguf.GetF32("asr.preprocessor.window_size", 0)
	windowStrideS := gguf.GetF32("asr.preprocessor.window_stride", 0)
	pre.WinLength = int(windowSizeS*float32(pre.SampleRate) + 0.5)
	pre.HopLength = int(windowStrideS*float32(pre.SampleRate) + 0.5)
	if pre.NumMelBins <= 0 || pre.NFFT <= 0 || pre.WinLength <= 0 || pre.HopLength <= 0 {
		return ParakeetConfig{}, fmt.Errorf("loading parakeet model: invalid preprocessor config %+v", pre)
	}

	enc := ParakeetEncoderConfig{
		DModel:               int(gguf.GetU32("asr.encoder.d_model", 0)),
		DFF:                  int(gguf.GetU32("asr.encoder.d_ff", 0)),
		NHeads:               int(gguf.GetU32("asr.encoder.n_heads", 0)),
		NLayers:              int(gguf.GetU32("asr.encoder.n_layers", 0)),
		ConvKernelSize:       int(gguf.GetU32("asr.encoder.conv_kernel_size", 0)),
		SubsamplingConvChans: int(gguf.GetU32("asr.encoder.subsampling_conv_channels", 0)),
		SubsamplingFactor:    int(gguf.GetU32("asr.encoder.subsampling_factor", 0)),
		Epsilon:              1e-5,
	}
	if enc.DModel <= 0 || enc.NHeads <= 0 || enc.NLayers <= 0 || enc.DFF <= 0 {
		return ParakeetConfig{}, fmt.Errorf("loading parakeet model: invalid encoder config %+v", enc)
	}
	if enc.DModel%enc.NHeads != 0 {
		return ParakeetConfig{}, fmt.Errorf("loading parakeet model: d_model %d not divisible by n_heads %d", enc.DModel, enc.NHeads)
	}
	enc.HeadDim = enc.DModel / enc.NHeads

	durationsU32, _ := gguf.GetU32Array("asr.tdt.durations")
	durations := make([]int, len(durationsU32))
	for i, d := range durationsU32 {
		durations[i] = int(d)
	}
	rnnt := ParakeetRNNTConfig{
		VocabSize:         int(gguf.GetU32("asr.rnnt.vocab_size", 0)) - 1, // metadata's vocab_size already counts the blank
		BlankID:           int(gguf.GetU32("asr.rnnt.blank_id", 0)),
		PredHidden:        int(gguf.GetU32("asr.rnnt.pred_hidden", 0)),
		PredEmbedDim:      int(gguf.GetU32("asr.rnnt.pred_embed_dim", 0)),
		PredNumLayers:     int(gguf.GetU32("asr.rnnt.pred_num_layers", 0)),
		JointDim:          int(gguf.GetU32("asr.rnnt.joint_dim", 0)),
		Durations:         durations,
		MaxSymbolsPerStep: int(gguf.GetU32("asr.rnnt.max_symbols_per_step", 10)),
	}
	if rnnt.VocabSize <= 0 || rnnt.PredHidden <= 0 || rnnt.PredNumLayers <= 0 || rnnt.JointDim <= 0 || len(rnnt.Durations) == 0 {
		return ParakeetConfig{}, fmt.Errorf("loading parakeet model: invalid RNNT/TDT config %+v", rnnt)
	}

	return ParakeetConfig{Preprocessor: pre, Encoder: enc, RNNT: rnnt}, nil
}
