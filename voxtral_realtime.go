package gopherllm

import (
	"fmt"
	"io"
)

// This file loads mistralai's Voxtral Realtime streaming speech-to-text
// model (general.architecture "voxtral_realtime") from a single
// self-contained GGUF. Unlike Pixtral's vision tower (a companion "mmproj"
// GGUF paired with a separate text-model GGUF, see pixtral_vision.go), a
// Voxtral Realtime GGUF bundles the mel-spectrogram frontend, the causal
// audio encoder, the audio-to-text adapter, and the LLM decoder in one file
// under four tensor-name prefixes: "frontend.*", "enc.*", "proj.*", "dec.*".
//
// Architecture (verified against a real
// mistralai/Voxtral-Mini-4B-Realtime-2602 Q6_K GGUF via --list-metadata /
// --list-tensors, cross-checked against huggingface/transformers'
// modeling_voxtral_realtime.py and antirez/voxtral.c's from-scratch C port):
//
//  1. Frontend: log-mel spectrogram. STFT with the model's own precomputed
//     Hann window (frontend.window) and mel filterbank (frontend.mel_filterbank,
//     [NFFT/2+1 -> NumMels]), then the stt.frontend.* normalize/global_log_mel_max
//     settings -- not a generic librosa-default mel, so both tensors must come
//     from the checkpoint, never be recomputed from scratch.
//  2. Causal audio encoder ("enc.*"): two causal Conv1d layers (kernel 3;
//     conv.0 stride 1, mel bins -> DModel; conv.1 stride 2, DModel -> DModel,
//     halving the frame rate) each followed by GELU, then NLayers pre-norm
//     transformer blocks -- full (non-GQA) attention with split-half
//     RoPE (GGUF Q/K rows and Q bias are permuted from the interleaved source) and a sliding-window causal mask, SwiGLU FFN. Biases
//     are present on q/v/out/ffn_down but not on k (verified: k.bias is absent
//     from the tensor inventory while q/v/out/ffn_down.bias are present).
//  3. Adapter ("proj.*"): concatenate DownsampleFactor consecutive encoder
//     output frames (DModel*DownsampleFactor == InputDim), linear -> GELU ->
//     linear, no biases, producing one embedding per AudioLengthPerTok output
//     tokens in the decoder's own hidden size.
//  4. Decoder ("dec.*"): a Ministral-style GQA transformer (split-half
//     RoPE for the GGUF's permuted Q/K rows, unlike the original safetensors,
//     sliding-window causal attention, SwiGLU FFN, tied token embedding /
//     output projection) with one addition: each block's FFN branch is
//     modulated by an Ada-RMSNorm-style scale gate conditioned on a fixed
//     "time" signal (dec.time_embed.inv_freq, sinusoidal in
//     DefaultNumDelayTokens) -- h_norm * (1 + ada_up(gelu(ada_down(t_cond))))
//     applied only before the FFN, never before attention. Audio embeddings
//     are summed elementwise into the text token embeddings at matching
//     sequence positions (no interleaving, no placeholder tokens).
//
// This file only loads and validates the weights; the forward pass (encoder
// attention/conv graph, adapter, ada-conditioned decoder graph) and Runner
// integration land separately.
type VoxtralRealtimeMelConfig struct {
	SampleRate      int     // stt.frontend.sample_rate
	NumMels         int     // stt.frontend.num_mels == enc.num_mel_bins
	NFFT            int     // stt.frontend.n_fft
	HopLength       int     // stt.frontend.hop_length
	WinLength       int     // stt.frontend.win_length
	FMin, FMax      float32 // stt.frontend.f_min / f_max
	Center          bool    // stt.frontend.center
	Dither          float32 // stt.frontend.dither
	PreEmphasis     float32 // stt.frontend.pre_emphasis
	PadMode         string  // stt.frontend.pad_mode, e.g. "reflect"
	Normalize       string  // stt.frontend.normalize, e.g. "global"
	MelNorm         string  // stt.frontend.mel_norm, e.g. "slaney" (informational -- already baked into the filterbank weights)
	GlobalLogMelMax float32 // stt.frontend.global_log_mel_max
}

type VoxtralRealtimeEncoderConfig struct {
	DModel        int
	FFNDim        int
	HeadDim       int
	NHeads        int
	NKVHeads      int
	NLayers       int
	NumMelBins    int
	SlidingWindow int
	RopeTheta     float32
	Epsilon       float32
}

type VoxtralRealtimeProjectorConfig struct {
	InputDim          int // encoder DModel * DownsampleFactor
	DownsampleFactor  int
	AudioLengthPerTok int
}

type VoxtralRealtimeDecoderConfig struct {
	HiddenSize            int
	IntermediateSize      int
	HeadDim               int
	NHeads                int
	NKVHeads              int
	NLayers               int
	VocabSize             int
	SlidingWindow         int
	RopeTheta             float32
	Epsilon               float32
	MaxPositionEmbeddings int
	TieWordEmbeddings     bool
}

type VoxtralRealtimeTimeConfig struct {
	EmbedDim              int     // stt.voxtral_realtime.time.embed_dim -- matches decoder HiddenSize
	AdaHidden             int     // stt.voxtral_realtime.time.ada_hidden -- the ada MLP's bottleneck width
	EmbedTheta            float32 // stt.voxtral_realtime.time.embed_theta
	DefaultNumDelayTokens int     // stt.voxtral_realtime.time.default_num_delay_tokens
}

type VoxtralRealtimeConfig struct {
	Mel                 VoxtralRealtimeMelConfig
	Encoder             VoxtralRealtimeEncoderConfig
	Projector           VoxtralRealtimeProjectorConfig
	Decoder             VoxtralRealtimeDecoderConfig
	Time                VoxtralRealtimeTimeConfig
	StreamingPadTokenID int
}

// VoxtralRealtimeEncoderConv is one of the two causal Conv1d layers in the
// encoder's conv stem. Weight is flattened [Kernel][In][Out] (fastest to
// slowest, matching the GGUF tensor's own dim order) rather than a Weight
// matvec type: a strided causal conv isn't a single matrix multiply.
type VoxtralRealtimeEncoderConv struct {
	Weight []float32
	Bias   []float32
	Kernel int
	In     int
	Out    int
	Stride int
}

// VoxtralRealtimeEncoderLayer is one pre-norm transformer block of the audio
// encoder: full (non-GQA) attention with split-half RoPE and a sliding
// causal mask, then a gated SwiGLU FFN. K carries no bias (see this file's
// doc comment); every other projection does.
type VoxtralRealtimeEncoderLayer struct {
	AttnNorm                []float32
	Q, K, V, Out            Weight
	QB, VB, OutB            []float32
	FFNNorm                 []float32
	FFNGate, FFNUp, FFNDown Weight
	FFNDownB                []float32
}

type VoxtralRealtimeEncoderWeights struct {
	Conv      [2]VoxtralRealtimeEncoderConv
	Layers    []VoxtralRealtimeEncoderLayer
	FinalNorm []float32
}

type VoxtralRealtimeProjectorWeights struct {
	Linear1, Linear2 Weight // no biases
}

// VoxtralRealtimeDecoderLayer is one decoder block: standard pre-norm GQA
// attention (split-half RoPE, sliding-window causal mask), then FFN preceded by
// the Ada-RMSNorm scale gate (AdaLinear1/2, no biases) computed from the
// fixed time-conditioning vector. No biases anywhere in this block.
type VoxtralRealtimeDecoderLayer struct {
	AttnNorm                []float32
	Q, K, V, O              Weight
	FFNNorm                 []float32
	AdaLinear1, AdaLinear2  Weight
	FFNGate, FFNUp, FFNDown Weight
}

type VoxtralRealtimeDecoderWeights struct {
	TokenEmbd        Weight // dec.token_embd.weight -- also the tied output projection
	Layers           []VoxtralRealtimeDecoderLayer
	OutputNorm       []float32
	TimeEmbedInvFreq []float32 // dec.time_embed.inv_freq, length HiddenSize/2
}

type VoxtralRealtimeWeights struct {
	MelWindow     []float32 // frontend.window, length Mel.WinLength
	MelFilterbank Weight    // frontend.mel_filterbank: Cols=NFFT/2+1, rows inferred as NumMels
	Encoder       VoxtralRealtimeEncoderWeights
	Projector     VoxtralRealtimeProjectorWeights
	Decoder       VoxtralRealtimeDecoderWeights
}

// releaseVoxtralRealtimeWeights releases every accelerator copy this model
// owns, mirroring releasePixtralVisionWeights.
func releaseVoxtralRealtimeWeights(weights *VoxtralRealtimeWeights) {
	if weights == nil {
		return
	}
	var releaser weightResourceReleaser
	releaser.release(&weights.MelFilterbank)
	for i := range weights.Encoder.Layers {
		l := &weights.Encoder.Layers[i]
		releaser.release(&l.Q)
		releaser.release(&l.K)
		releaser.release(&l.V)
		releaser.release(&l.Out)
		releaser.release(&l.FFNGate)
		releaser.release(&l.FFNUp)
		releaser.release(&l.FFNDown)
	}
	releaser.release(&weights.Projector.Linear1)
	releaser.release(&weights.Projector.Linear2)
	releaser.release(&weights.Decoder.TokenEmbd)
	for i := range weights.Decoder.Layers {
		l := &weights.Decoder.Layers[i]
		releaser.release(&l.Q)
		releaser.release(&l.K)
		releaser.release(&l.V)
		releaser.release(&l.O)
		releaser.release(&l.AdaLinear1)
		releaser.release(&l.AdaLinear2)
		releaser.release(&l.FFNGate)
		releaser.release(&l.FFNUp)
		releaser.release(&l.FFNDown)
	}
}

// LoadVoxtralRealtimeModel loads a self-contained Voxtral Realtime GGUF's
// config and weights. useMetal requests accelerator-resident copies of the
// large matmul weights where supported, matching LoadPixtralVisionModel's
// convention. See this file's doc comment for the computational graph these
// weights feed.
func LoadVoxtralRealtimeModel(data []byte, gguf *GGUFFile, useMetal bool, logw io.Writer) (VoxtralRealtimeConfig, VoxtralRealtimeWeights, error) {
	if logw == nil {
		logw = io.Discard
	}
	arch, _ := gguf.GetString("general.architecture")
	if arch != "voxtral_realtime" {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: general.architecture is %q, want \"voxtral_realtime\"", arch)
	}

	mel := VoxtralRealtimeMelConfig{
		SampleRate:      int(gguf.GetU32("stt.frontend.sample_rate", 16000)),
		NumMels:         int(gguf.GetU32("stt.frontend.num_mels", 0)),
		NFFT:            int(gguf.GetU32("stt.frontend.n_fft", 0)),
		HopLength:       int(gguf.GetU32("stt.frontend.hop_length", 0)),
		WinLength:       int(gguf.GetU32("stt.frontend.win_length", 0)),
		FMin:            gguf.GetF32("stt.frontend.f_min", 0),
		FMax:            gguf.GetF32("stt.frontend.f_max", 8000),
		Center:          gguf.GetBool("stt.frontend.center", true),
		Dither:          gguf.GetF32("stt.frontend.dither", 0),
		PreEmphasis:     gguf.GetF32("stt.frontend.pre_emphasis", 0),
		GlobalLogMelMax: gguf.GetF32("stt.frontend.global_log_mel_max", 0),
	}
	mel.PadMode, _ = gguf.GetString("stt.frontend.pad_mode")
	mel.Normalize, _ = gguf.GetString("stt.frontend.normalize")
	mel.MelNorm, _ = gguf.GetString("stt.frontend.mel_norm")
	if mel.NumMels <= 0 || mel.NFFT <= 0 || mel.HopLength <= 0 || mel.WinLength <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: invalid mel frontend config %+v", mel)
	}

	enc := VoxtralRealtimeEncoderConfig{
		DModel:        int(gguf.GetU32("stt.voxtral_realtime.encoder.d_model", 0)),
		FFNDim:        int(gguf.GetU32("stt.voxtral_realtime.encoder.ffn_dim", 0)),
		HeadDim:       int(gguf.GetU32("stt.voxtral_realtime.encoder.head_dim", 0)),
		NHeads:        int(gguf.GetU32("stt.voxtral_realtime.encoder.n_heads", 0)),
		NKVHeads:      int(gguf.GetU32("stt.voxtral_realtime.encoder.n_kv_heads", 0)),
		NLayers:       int(gguf.GetU32("stt.voxtral_realtime.encoder.n_layers", 0)),
		NumMelBins:    int(gguf.GetU32("stt.voxtral_realtime.encoder.num_mel_bins", 0)),
		SlidingWindow: int(gguf.GetU32("stt.voxtral_realtime.encoder.sliding_window", 0)),
		RopeTheta:     gguf.GetF32("stt.voxtral_realtime.encoder.rope_theta", 1e6),
		Epsilon:       gguf.GetF32("stt.voxtral_realtime.encoder.rms_norm_eps", 1e-5),
	}
	if enc.DModel <= 0 || enc.NHeads <= 0 || enc.NLayers <= 0 || enc.FFNDim <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: invalid encoder config %+v", enc)
	}
	if enc.HeadDim <= 0 {
		if enc.DModel%enc.NHeads != 0 {
			return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: encoder d_model %d not divisible by n_heads %d", enc.DModel, enc.NHeads)
		}
		enc.HeadDim = enc.DModel / enc.NHeads
	}
	if enc.NKVHeads <= 0 {
		enc.NKVHeads = enc.NHeads
	}

	proj := VoxtralRealtimeProjectorConfig{
		InputDim:          int(gguf.GetU32("stt.voxtral_realtime.projector.input_dim", 0)),
		DownsampleFactor:  int(gguf.GetU32("stt.voxtral_realtime.projector.downsample_factor", 0)),
		AudioLengthPerTok: int(gguf.GetU32("stt.voxtral_realtime.projector.audio_length_per_tok", 0)),
	}
	if proj.InputDim <= 0 || proj.DownsampleFactor <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: invalid projector config %+v", proj)
	}

	dec := VoxtralRealtimeDecoderConfig{
		HiddenSize:            int(gguf.GetU32("stt.voxtral_realtime.decoder.hidden_size", 0)),
		IntermediateSize:      int(gguf.GetU32("stt.voxtral_realtime.decoder.intermediate_size", 0)),
		HeadDim:               int(gguf.GetU32("stt.voxtral_realtime.decoder.head_dim", 0)),
		NHeads:                int(gguf.GetU32("stt.voxtral_realtime.decoder.n_heads", 0)),
		NKVHeads:              int(gguf.GetU32("stt.voxtral_realtime.decoder.n_kv_heads", 0)),
		NLayers:               int(gguf.GetU32("stt.voxtral_realtime.decoder.n_layers", 0)),
		VocabSize:             int(gguf.GetU32("stt.voxtral_realtime.decoder.vocab_size", 0)),
		SlidingWindow:         int(gguf.GetU32("stt.voxtral_realtime.decoder.sliding_window", 0)),
		RopeTheta:             gguf.GetF32("stt.voxtral_realtime.decoder.rope_theta", 1e6),
		Epsilon:               gguf.GetF32("stt.voxtral_realtime.decoder.rms_norm_eps", 1e-5),
		MaxPositionEmbeddings: int(gguf.GetU32("stt.voxtral_realtime.decoder.max_position_embeddings", 0)),
		TieWordEmbeddings:     gguf.GetBool("stt.voxtral_realtime.decoder.tie_word_embeddings", true),
	}
	if dec.HiddenSize <= 0 || dec.NHeads <= 0 || dec.NLayers <= 0 || dec.VocabSize <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: invalid decoder config %+v", dec)
	}
	if dec.HeadDim <= 0 {
		if dec.HiddenSize%dec.NHeads != 0 {
			return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: decoder hidden_size %d not divisible by n_heads %d", dec.HiddenSize, dec.NHeads)
		}
		dec.HeadDim = dec.HiddenSize / dec.NHeads
	}
	if dec.NKVHeads <= 0 {
		dec.NKVHeads = dec.NHeads
	}

	timeCfg := VoxtralRealtimeTimeConfig{
		EmbedDim:              int(gguf.GetU32("stt.voxtral_realtime.time.embed_dim", 0)),
		AdaHidden:             int(gguf.GetU32("stt.voxtral_realtime.time.ada_hidden", 0)),
		EmbedTheta:            gguf.GetF32("stt.voxtral_realtime.time.embed_theta", 10000),
		DefaultNumDelayTokens: int(gguf.GetU32("stt.voxtral_realtime.time.default_num_delay_tokens", 0)),
	}
	if timeCfg.EmbedDim <= 0 || timeCfg.AdaHidden <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: invalid time-conditioning config %+v", timeCfg)
	}

	config := VoxtralRealtimeConfig{
		Mel:                 mel,
		Encoder:             enc,
		Projector:           proj,
		Decoder:             dec,
		Time:                timeCfg,
		StreamingPadTokenID: int(gguf.GetU32("stt.voxtral_realtime.streaming_pad_token_id", 0)),
	}

	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, false, false, useMetal)
	}
	loadConv := func(name string, kernel, in, out, stride int) (VoxtralRealtimeEncoderConv, error) {
		info, ok := tensors[name]
		if !ok {
			return VoxtralRealtimeEncoderConv{}, fmt.Errorf("tensor %s not found", name)
		}
		if want := kernel * in * out; int(info.Numel()) != want {
			return VoxtralRealtimeEncoderConv{}, fmt.Errorf("tensor %s has %d elements, want %d (kernel=%d in=%d out=%d)", name, info.Numel(), want, kernel, in, out)
		}
		w, err := loadF32Vec(data, gguf.DataOffset, name, tensors, inferred)
		if err != nil {
			return VoxtralRealtimeEncoderConv{}, err
		}
		b, err := loadF32Vec(data, gguf.DataOffset, name[:len(name)-len("weight")]+"bias", tensors, inferred)
		if err != nil {
			return VoxtralRealtimeEncoderConv{}, err
		}
		return VoxtralRealtimeEncoderConv{Weight: w, Bias: b, Kernel: kernel, In: in, Out: out, Stride: stride}, nil
	}

	weights := VoxtralRealtimeWeights{}

	weights.MelWindow, _ = loadF32Vec(data, gguf.DataOffset, "frontend.window", tensors, inferred)
	if len(weights.MelWindow) != mel.WinLength {
		return config, weights, fmt.Errorf("loading voxtral realtime model: frontend.window has %d elements, want win_length=%d", len(weights.MelWindow), mel.WinLength)
	}
	melFilterbank, err := load("frontend.mel_filterbank")
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	weights.MelFilterbank = melFilterbank

	conv0, err := loadConv("enc.conv.0.weight", 3, mel.NumMels, enc.DModel, 1)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	conv1, err := loadConv("enc.conv.1.weight", 3, enc.DModel, enc.DModel, 2)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	weights.Encoder.Conv = [2]VoxtralRealtimeEncoderConv{conv0, conv1}

	weights.Encoder.FinalNorm, err = loadF32Vec(data, gguf.DataOffset, "enc.final_norm.weight", tensors, inferred)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}

	encLayers := make([]VoxtralRealtimeEncoderLayer, 0, enc.NLayers)
	for l := range enc.NLayers {
		p := fmt.Sprintf("enc.blocks.%d.", l)
		var layer VoxtralRealtimeEncoderLayer
		layer.AttnNorm, err = loadF32Vec(data, gguf.DataOffset, p+"norm_attn.weight", tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.Q, err = load(p + "attn.q.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.K, err = load(p + "attn.k.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.V, err = load(p + "attn.v.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.Out, err = load(p + "attn.out.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		qkvOutDim := enc.NHeads * enc.HeadDim
		layer.QB = loadOptionalF32Vec(data, gguf.DataOffset, p+"attn.q.bias", tensors, inferred, qkvOutDim)
		layer.VB = loadOptionalF32Vec(data, gguf.DataOffset, p+"attn.v.bias", tensors, inferred, qkvOutDim)
		layer.OutB = loadOptionalF32Vec(data, gguf.DataOffset, p+"attn.out.bias", tensors, inferred, enc.DModel)
		layer.FFNNorm, err = loadF32Vec(data, gguf.DataOffset, p+"norm_ffn.weight", tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNGate, err = load(p + "ffn.gate.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNUp, err = load(p + "ffn.up.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNDown, err = load(p + "ffn.down.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		layer.FFNDownB = loadOptionalF32Vec(data, gguf.DataOffset, p+"ffn.down.bias", tensors, inferred, enc.DModel)
		encLayers = append(encLayers, layer)
		if l == 0 || l+1 == enc.NLayers || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded voxtral encoder layer %d/%d\n", l+1, enc.NLayers)
		}
	}
	weights.Encoder.Layers = encLayers

	if weights.Projector.Linear1, err = load("proj.linear_1.weight"); err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	if weights.Projector.Linear2, err = load("proj.linear_2.weight"); err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}

	if weights.Decoder.TokenEmbd, err = load("dec.token_embd.weight"); err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	weights.Decoder.OutputNorm, err = loadF32Vec(data, gguf.DataOffset, "dec.output_norm.weight", tensors, inferred)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	weights.Decoder.TimeEmbedInvFreq, err = loadF32Vec(data, gguf.DataOffset, "dec.time_embed.inv_freq", tensors, inferred)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	if want := dec.HiddenSize / 2; len(weights.Decoder.TimeEmbedInvFreq) != want {
		return config, weights, fmt.Errorf("loading voxtral realtime model: dec.time_embed.inv_freq has %d elements, want hidden_size/2=%d", len(weights.Decoder.TimeEmbedInvFreq), want)
	}

	decLayers := make([]VoxtralRealtimeDecoderLayer, 0, dec.NLayers)
	for l := range dec.NLayers {
		p := fmt.Sprintf("dec.blocks.%d.", l)
		var layer VoxtralRealtimeDecoderLayer
		layer.AttnNorm, err = loadF32Vec(data, gguf.DataOffset, p+"norm_attn.weight", tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.Q, err = load(p + "attn.q.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.K, err = load(p + "attn.k.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.V, err = load(p + "attn.v.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.O, err = load(p + "attn.o.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		layer.FFNNorm, err = loadF32Vec(data, gguf.DataOffset, p+"norm_ffn.weight", tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.AdaLinear1, err = load(p + "ada.linear_1.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.AdaLinear2, err = load(p + "ada.linear_2.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNGate, err = load(p + "ffn.gate.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNUp, err = load(p + "ffn.up.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNDown, err = load(p + "ffn.down.weight"); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		decLayers = append(decLayers, layer)
		if l == 0 || l+1 == dec.NLayers || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded voxtral decoder layer %d/%d\n", l+1, dec.NLayers)
		}
	}
	weights.Decoder.Layers = decLayers

	return config, weights, nil
}
