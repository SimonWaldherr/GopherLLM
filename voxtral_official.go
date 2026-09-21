package gopherllm

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// This file loads mistralai's OFFICIAL (non-GGUF) Voxtral Realtime release:
// consolidated.safetensors (Mistral's own native "consolidated" checkpoint
// format -- dense BF16, no quantization; NOT model.safetensors, which is
// the same weights under transformers' own naming and is not read here)
// plus its sibling params.json (hyperparameters) and tekken.json
// (tokenizer). This is a THIRD tensor-naming convention for the identical
// checkpoint voxtral_realtime.go's GGUF loader already supports two
// conventions of -- see voxtralTensorNames' doc comment there for the other
// two, both produced by third-party GGUF conversions of this same release.
//
// Two things are simpler here than in the GGUF loader:
//   - No quantization: every tensor loads straight into Weight.F32,
//     reusing the same F32 matvec/conv path the GGUF loader's mel-filterbank
//     and conv tensors already exercise.
//   - No axis permutation: PyTorch's native tensor layout ([out,in] for a
//     Linear weight, [out,in,kernel] for a Conv1d weight, both row-major)
//     already matches what this codebase's F32 matvec/conv code expects
//     byte-for-byte (verified against applyCausalConv1dContext's own
//     indexing, conv.Weight[o*in*Kernel+c*Kernel+k]) -- unlike GGUF, whose
//     dims[0]/dims[1] convention needed an explicit transpose for the mel
//     filterbank tensor in the "voxtral.*" GGUF convention.
//
// One thing is harder: the official release ships no precomputed mel
// filterbank, STFT window, or time-embedding inverse-frequency table at
// all (unlike either known GGUF convention, which bakes at least the
// filterbank in for convenience) -- all three are computed from their
// standard formulas instead (voxtral_mel_filterbank.go,
// voxtralPeriodicHannWindow, voxtralSinusoidalInvFreq), verified byte-close
// against a working GGUF's shipped values.

// voxtralParamsFile is params.json's schema, Mistral's own native
// hyperparameter format (distinct from HF transformers' config.json, which
// describes the identical model under different field names and is not
// read here). Fetched and inspected directly from the official release.
type voxtralParamsFile struct {
	Dim                int     `json:"dim"`
	NLayers            int     `json:"n_layers"`
	HeadDim            int     `json:"head_dim"`
	HiddenDim          int     `json:"hidden_dim"`
	NHeads             int     `json:"n_heads"`
	NKVHeads           int     `json:"n_kv_heads"`
	RopeTheta          float64 `json:"rope_theta"`
	NormEps            float64 `json:"norm_eps"`
	VocabSize          int     `json:"vocab_size"`
	TiedEmbeddings     bool    `json:"tied_embeddings"`
	SlidingWindow      int     `json:"sliding_window"`
	ModelMaxLength     int     `json:"model_max_length"`
	AdaRMSNormTCondDim int     `json:"ada_rms_norm_t_cond_dim"`
	Multimodal         struct {
		WhisperModelArgs struct {
			EncoderArgs struct {
				AudioEncodingArgs struct {
					SamplingRate    int     `json:"sampling_rate"`
					FrameRate       float64 `json:"frame_rate"`
					NumMelBins      int     `json:"num_mel_bins"`
					HopLength       int     `json:"hop_length"`
					WindowSize      int     `json:"window_size"`
					GlobalLogMelMax float64 `json:"global_log_mel_max"`
				} `json:"audio_encoding_args"`
				Dim           int     `json:"dim"`
				NLayers       int     `json:"n_layers"`
				HeadDim       int     `json:"head_dim"`
				HiddenDim     int     `json:"hidden_dim"`
				NHeads        int     `json:"n_heads"`
				NKVHeads      int     `json:"n_kv_heads"`
				RopeTheta     float64 `json:"rope_theta"`
				NormEps       float64 `json:"norm_eps"`
				SlidingWindow int     `json:"sliding_window"`
			} `json:"encoder_args"`
			DownsampleArgs struct {
				DownsampleFactor int `json:"downsample_factor"`
			} `json:"downsample_args"`
		} `json:"whisper_model_args"`
	} `json:"multimodal"`
}

// voxtralTekkenAudioConfig is tekken.json's "audio" field: streaming timing
// parameters that live alongside the tokenizer in the official release
// rather than in params.json. DefaultNumDelayTokens has no field of its
// own anywhere in either file -- it's derived the same way the checkpoint
// itself defines it: delay (ms) / frame period (ms), verified to land on
// the same value (6) every known GGUF conversion carries explicitly.
type voxtralTekkenAudioConfig struct {
	FrameRate               float64 `json:"frame_rate"`
	TranscriptionDelayMs    float64 `json:"transcription_delay_ms"`
	StreamingNLeftPadTokens int     `json:"streaming_n_left_pad_tokens"`
}

// LoadVoxtralRealtimeModelFromSafetensors loads the official
// mistralai/Voxtral-Mini-4B-Realtime-2602 release from a directory
// containing consolidated.safetensors, params.json, and tekken.json -- the
// exact file set the official Hugging Face repo publishes for this model.
// Config, weights, and tokenizer are returned together (unlike the GGUF
// loader's split LoadVoxtralRealtimeModel/TokenizerFromMetadata) because
// StreamingPadTokenID can only be resolved after the tokenizer exists (it's
// the vocabulary ID of the literal "[STREAMING_PAD]" special token, not a
// small fixed metadata field the way it is in both GGUF conventions).
func LoadVoxtralRealtimeModelFromSafetensors(dir string, useMetal bool, logw io.Writer) (VoxtralRealtimeConfig, VoxtralRealtimeWeights, *Tokenizer, error) {
	if logw == nil {
		logw = io.Discard
	}

	paramsData, err := os.ReadFile(filepath.Join(dir, "params.json"))
	if err != nil {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: reading params.json: %w", err)
	}
	var params voxtralParamsFile
	if err := json.Unmarshal(paramsData, &params); err != nil {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: parsing params.json: %w", err)
	}

	tekkenData, err := os.ReadFile(filepath.Join(dir, "tekken.json"))
	if err != nil {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: reading tekken.json: %w", err)
	}
	tok, err := voxtralTokenizerFromTekkenJSON(tekkenData)
	if err != nil {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: building tokenizer: %w", err)
	}
	var tekkenWrap struct {
		Audio voxtralTekkenAudioConfig `json:"audio"`
	}
	if err := json.Unmarshal(tekkenData, &tekkenWrap); err != nil {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: parsing tekken.json audio config: %w", err)
	}

	aenc := params.Multimodal.WhisperModelArgs.EncoderArgs.AudioEncodingArgs
	mel := VoxtralRealtimeMelConfig{
		SampleRate:      aenc.SamplingRate,
		NumMels:         aenc.NumMelBins,
		NFFT:            aenc.WindowSize,
		HopLength:       aenc.HopLength,
		WinLength:       aenc.WindowSize,
		Center:          true,
		FMax:            float32(aenc.SamplingRate) / 2,
		GlobalLogMelMax: float32(aenc.GlobalLogMelMax),
	}
	if mel.NumMels <= 0 || mel.NFFT <= 0 || mel.HopLength <= 0 || mel.WinLength <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: invalid mel frontend config %+v", mel)
	}

	encArgs := params.Multimodal.WhisperModelArgs.EncoderArgs
	enc := VoxtralRealtimeEncoderConfig{
		DModel: encArgs.Dim, FFNDim: encArgs.HiddenDim, HeadDim: encArgs.HeadDim,
		NHeads: encArgs.NHeads, NKVHeads: encArgs.NKVHeads, NLayers: encArgs.NLayers,
		NumMelBins: aenc.NumMelBins, SlidingWindow: encArgs.SlidingWindow,
		RopeTheta: float32(encArgs.RopeTheta), Epsilon: float32(encArgs.NormEps),
	}
	if enc.DModel <= 0 || enc.NHeads <= 0 || enc.NLayers <= 0 || enc.FFNDim <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: invalid encoder config %+v", enc)
	}
	if enc.NKVHeads <= 0 {
		enc.NKVHeads = enc.NHeads
	}

	downsampleFactor := params.Multimodal.WhisperModelArgs.DownsampleArgs.DownsampleFactor
	proj := VoxtralRealtimeProjectorConfig{
		InputDim:         enc.DModel * downsampleFactor,
		DownsampleFactor: downsampleFactor,
	}
	if proj.InputDim <= 0 || proj.DownsampleFactor <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: invalid projector config %+v", proj)
	}

	dec := VoxtralRealtimeDecoderConfig{
		HiddenSize: params.Dim, IntermediateSize: params.HiddenDim, HeadDim: params.HeadDim,
		NHeads: params.NHeads, NKVHeads: params.NKVHeads, NLayers: params.NLayers, VocabSize: params.VocabSize,
		SlidingWindow: params.SlidingWindow, RopeTheta: float32(params.RopeTheta), Epsilon: float32(params.NormEps),
		MaxPositionEmbeddings: params.ModelMaxLength, TieWordEmbeddings: params.TiedEmbeddings,
	}
	if dec.HiddenSize <= 0 || dec.NHeads <= 0 || dec.NLayers <= 0 || dec.VocabSize <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: invalid decoder config %+v", dec)
	}
	if dec.NKVHeads <= 0 {
		dec.NKVHeads = dec.NHeads
	}

	delayTokens := 0
	if tekkenWrap.Audio.FrameRate > 0 {
		framePeriodMs := 1000.0 / tekkenWrap.Audio.FrameRate
		delayTokens = int(math.Round(tekkenWrap.Audio.TranscriptionDelayMs / framePeriodMs))
	}
	timeCfg := VoxtralRealtimeTimeConfig{
		EmbedDim: dec.HiddenSize, AdaHidden: params.AdaRMSNormTCondDim,
		EmbedTheta: 10000, DefaultNumDelayTokens: delayTokens,
	}
	if timeCfg.AdaHidden <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: invalid time-conditioning config %+v", timeCfg)
	}

	streamingPadID, ok := tok.TokenToID["[STREAMING_PAD]"]
	if !ok {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: tokenizer has no [STREAMING_PAD] token")
	}

	config := VoxtralRealtimeConfig{
		Mel: mel, Encoder: enc, Projector: proj, Decoder: dec, Time: timeCfg,
		StreamingPadTokenID: int(streamingPadID),
		// The raw PyTorch checkpoint's Q/K rows are in the reference
		// implementation's native interleaved-pair RoPE layout -- split-half
		// is a permutation GGUF conversion tools apply, not something
		// present in an unconverted safetensors release. Matches this
		// codebase's own ropeInterleaved("mistral3")==true for the same
		// underlying architecture family under mainline conversion, and
		// matches what the "voxtral.*" GGUF convention (itself apparently
		// produced by an unpermuting conversion path) needed too.
		RopeInterleaved: true,
	}

	stData, err := os.ReadFile(filepath.Join(dir, "consolidated.safetensors"))
	if err != nil {
		return config, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: reading consolidated.safetensors: %w", err)
	}
	st, err := ParseSafetensors(stData)
	if err != nil {
		return config, VoxtralRealtimeWeights{}, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}

	f32Weight := func(name string) (Weight, error) {
		f, err := st.F32(name)
		if err != nil {
			return Weight{}, err
		}
		return Weight{F32: f}, nil
	}
	loadConv := func(weightName, biasName string, kernel, in, out int) (VoxtralRealtimeEncoderConv, error) {
		w, err := st.F32(weightName)
		if err != nil {
			return VoxtralRealtimeEncoderConv{}, err
		}
		if want := kernel * in * out; len(w) != want {
			return VoxtralRealtimeEncoderConv{}, fmt.Errorf("tensor %s has %d elements, want %d (kernel=%d in=%d out=%d)", weightName, len(w), want, kernel, in, out)
		}
		b, err := st.F32(biasName)
		if err != nil {
			return VoxtralRealtimeEncoderConv{}, err
		}
		return VoxtralRealtimeEncoderConv{Weight: w, Bias: b, Kernel: kernel, In: in, Out: out, Stride: 0}, nil
	}

	weights := VoxtralRealtimeWeights{}
	weights.MelWindow = voxtralPeriodicHannWindow(mel.WinLength)
	weights.MelFilterbank = Weight{F32: voxtralSlaneyMelFilterbank(mel.SampleRate, mel.NFFT, mel.NumMels, 0, float64(mel.FMax))}

	const embed = "mm_streams_embeddings.embedding_module."
	conv0, err := loadConv(embed+"whisper_encoder.conv_layers.0.conv.weight", embed+"whisper_encoder.conv_layers.0.conv.bias", 3, mel.NumMels, enc.DModel)
	if err != nil {
		return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}
	conv0.Stride = 1
	conv1, err := loadConv(embed+"whisper_encoder.conv_layers.1.conv.weight", embed+"whisper_encoder.conv_layers.1.conv.bias", 3, enc.DModel, enc.DModel)
	if err != nil {
		return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}
	conv1.Stride = 2
	weights.Encoder.Conv = [2]VoxtralRealtimeEncoderConv{conv0, conv1}

	weights.Encoder.FinalNorm, err = st.F32(embed + "whisper_encoder.transformer.norm.weight")
	if err != nil {
		return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}

	encLayers := make([]VoxtralRealtimeEncoderLayer, 0, enc.NLayers)
	for l := range enc.NLayers {
		p := fmt.Sprintf("%swhisper_encoder.transformer.layers.%d.", embed, l)
		var layer VoxtralRealtimeEncoderLayer
		if layer.AttnNorm, err = st.F32(p + "attention_norm.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.Q, err = f32Weight(p + "attention.wq.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.K, err = f32Weight(p + "attention.wk.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.V, err = f32Weight(p + "attention.wv.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.Out, err = f32Weight(p + "attention.wo.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		layer.QB, _ = st.F32(p + "attention.wq.bias")
		layer.VB, _ = st.F32(p + "attention.wv.bias")
		layer.OutB, _ = st.F32(p + "attention.wo.bias")
		if layer.FFNNorm, err = st.F32(p + "ffn_norm.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.FFNGate, err = f32Weight(p + "feed_forward.w1.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.FFNUp, err = f32Weight(p + "feed_forward.w3.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.FFNDown, err = f32Weight(p + "feed_forward.w2.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		layer.FFNDownB, _ = st.F32(p + "feed_forward.w2.bias")
		encLayers = append(encLayers, layer)
		if l == 0 || l+1 == enc.NLayers || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded voxtral encoder layer %d/%d\n", l+1, enc.NLayers)
		}
	}
	weights.Encoder.Layers = encLayers

	if weights.Projector.Linear1, err = f32Weight(embed + "audio_language_projection.0.weight"); err != nil {
		return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}
	if weights.Projector.Linear2, err = f32Weight(embed + "audio_language_projection.2.weight"); err != nil {
		return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}

	if weights.Decoder.TokenEmbd, err = f32Weight(embed + "tok_embeddings.weight"); err != nil {
		return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}
	if weights.Decoder.OutputNorm, err = st.F32("norm.weight"); err != nil {
		return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
	}
	weights.Decoder.TimeEmbedInvFreq = voxtralSinusoidalInvFreq(timeCfg.EmbedDim, timeCfg.EmbedTheta)

	decLayers := make([]VoxtralRealtimeDecoderLayer, 0, dec.NLayers)
	for l := range dec.NLayers {
		p := fmt.Sprintf("layers.%d.", l)
		var layer VoxtralRealtimeDecoderLayer
		if layer.AttnNorm, err = st.F32(p + "attention_norm.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.Q, err = f32Weight(p + "attention.wq.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.K, err = f32Weight(p + "attention.wk.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.V, err = f32Weight(p + "attention.wv.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.O, err = f32Weight(p + "attention.wo.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.FFNNorm, err = st.F32(p + "ffn_norm.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.AdaLinear1, err = f32Weight(p + "ada_rms_norm_t_cond.0.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.AdaLinear2, err = f32Weight(p + "ada_rms_norm_t_cond.2.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.FFNGate, err = f32Weight(p + "feed_forward.w1.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.FFNUp, err = f32Weight(p + "feed_forward.w3.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		if layer.FFNDown, err = f32Weight(p + "feed_forward.w2.weight"); err != nil {
			return config, weights, nil, fmt.Errorf("loading voxtral official model: %w", err)
		}
		decLayers = append(decLayers, layer)
		if l == 0 || l+1 == dec.NLayers || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded voxtral decoder layer %d/%d\n", l+1, dec.NLayers)
		}
	}
	weights.Decoder.Layers = decLayers

	_ = useMetal // Metal-resident copies land alongside the GGUF loader's own useMetal support, not yet wired here.
	return config, weights, tok, nil
}
