package gopherllm

import (
	"fmt"
	"io"
	"math"
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
	// RopeInterleaved selects which row order the encoder/decoder Q/K
	// weights are stored in, and therefore which RoPE rotation convention
	// (applyPreparedRope's interleaved parameter) matches them: false
	// (split-half, e.g. sub[i]/sub[i+half] rotated as a pair) for a GGUF
	// whose conversion tool pre-permuted Q/K rows the way this loader's
	// original "stt.*"-convention reference file does, true (interleaved,
	// sub[2i]/sub[2i+1]) for one that did not -- matching how this
	// codebase's ropeInterleaved(arch) already treats the "mistral3"/
	// "ministral" architecture family the decoder is built on, which
	// mainline llama.cpp conversion leaves interleaved by default. Set
	// from the tensor-naming convention detected at load time (see
	// voxtralTensorNames), since the two known GGUF conventions for this
	// checkpoint come from different conversion tooling and were verified
	// (by comparing decode behavior against a known-transcribing
	// reference file) to disagree on this.
	RopeInterleaved bool
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

// gguFirstU32/gguFirstF32/gguFirstBool try each key in order and return the
// first one present with a compatible kind, falling back to def if none
// match -- for reading the same value under whichever of the two known
// metadata namespaces (see voxtralTensorNames) a given GGUF actually uses.
func gguFirstU32(gguf *GGUFFile, def uint32, keys ...string) uint32 {
	for _, k := range keys {
		if v, ok := gguf.Metadata[k]; ok {
			if n, ok := v.AsU32(); ok {
				return n
			}
		}
	}
	return def
}

func gguFirstF32(gguf *GGUFFile, def float32, keys ...string) float32 {
	for _, k := range keys {
		if v, ok := gguf.Metadata[k]; ok {
			if n, ok := v.AsF32(); ok {
				return n
			}
		}
	}
	return def
}

func gguFirstBool(gguf *GGUFFile, def bool, keys ...string) bool {
	for _, k := range keys {
		if v, ok := gguf.Metadata[k]; ok {
			if b, ok := v.AsBool(); ok {
				return b
			}
		}
	}
	return def
}

// voxtralEncBlockNames/voxtralDecBlockNames are one transformer block's
// tensor names, resolved for whichever naming convention the file uses.
type voxtralEncBlockNames struct {
	AttnNorm, Q, QB, K, V, VB, Out, OutB, FFNNorm, Gate, Up, Down, DownB string
}

type voxtralDecBlockNames struct {
	AttnNorm, Q, K, V, O, FFNNorm, Ada1, Ada2, Gate, Up, Down string
}

// voxtralTensorNames resolves every tensor and metadata key this loader
// reads, since two different public GGUF conversions of the identical
// mistralai/Voxtral-Mini-4B-Realtime-2602 checkpoint use different naming
// throughout: the original ("stt.frontend.*"/"stt.voxtral_realtime.*"
// metadata, "enc.blocks.N."/"dec.blocks.N." tensors) this loader was written
// against, and a second, more compact convention seen in at least one other
// upload ("voxtral.*" metadata, "enc.blk.N."/"dec.blk.N." tensors, FFN
// gate/up/down folded into ffn_w1/w3/w2, no "dec."/"enc." prefix on the
// top-level embedding/norm tensors). Verified by cross-checking tensor
// shapes and metadata values between a working file in the first convention
// and one in the second -- same numbers, different names, confirming they
// describe the identical model rather than a genuinely different one.
//
// The compact convention also omits two tensors the first convention
// stores explicitly: the STFT's Hann window (a fixed, reproducible function
// of WinLength, not learned data) and the time-conditioning sinusoid's
// inverse-frequency table (a fixed function of EmbedDim/EmbedTheta) --
// LoadVoxtralRealtimeModel computes both algorithmically when their tensor
// is absent, falling back from "load the checkpoint's exact values" to
// "recompute the same standard formula", per this file's own note that the
// checkpoint tensor exists mainly for byte-exact fidelity, not because the
// values are checkpoint-specific.
type voxtralTensorNames struct {
	compact bool
}

func detectVoxtralTensorNames(tensors map[string]TensorInfo) voxtralTensorNames {
	_, compact := tensors["dec.blk.0.attn_q.weight"]
	return voxtralTensorNames{compact: compact}
}

// window returns "" when this convention has no window tensor at all, so
// the caller must compute a periodic Hann window instead of loading one.
func (n voxtralTensorNames) window() string {
	if n.compact {
		return ""
	}
	return "frontend.window"
}

// filterbank additionally reports whether the tensor's on-disk layout is
// the transpose of what this loader needs (see the long comment at its call
// site in LoadVoxtralRealtimeModel for why that's a real, verified
// difference and not just a naming one).
func (n voxtralTensorNames) filterbank() (name string, transposed bool) {
	if n.compact {
		return "audio.mel_filters", true
	}
	return "frontend.mel_filterbank", false
}

func (n voxtralTensorNames) conv(i int) (weight, bias string) {
	if n.compact {
		return fmt.Sprintf("enc.conv%d.weight", i), fmt.Sprintf("enc.conv%d.bias", i)
	}
	return fmt.Sprintf("enc.conv.%d.weight", i), fmt.Sprintf("enc.conv.%d.bias", i)
}

func (n voxtralTensorNames) encFinalNorm() string {
	if n.compact {
		return "enc.norm.weight"
	}
	return "enc.final_norm.weight"
}

func (n voxtralTensorNames) encBlock(l int) voxtralEncBlockNames {
	if n.compact {
		p := fmt.Sprintf("enc.blk.%d.", l)
		return voxtralEncBlockNames{
			AttnNorm: p + "attn_norm.weight",
			Q:        p + "attn_q.weight", QB: p + "attn_q.bias",
			K: p + "attn_k.weight",
			V: p + "attn_v.weight", VB: p + "attn_v.bias",
			Out: p + "attn_o.weight", OutB: p + "attn_o.bias",
			FFNNorm: p + "ffn_norm.weight",
			Gate:    p + "ffn_w1.weight", Up: p + "ffn_w3.weight",
			Down: p + "ffn_w2.weight", DownB: p + "ffn_w2.bias",
		}
	}
	p := fmt.Sprintf("enc.blocks.%d.", l)
	return voxtralEncBlockNames{
		AttnNorm: p + "norm_attn.weight",
		Q:        p + "attn.q.weight", QB: p + "attn.q.bias",
		K: p + "attn.k.weight",
		V: p + "attn.v.weight", VB: p + "attn.v.bias",
		Out: p + "attn.out.weight", OutB: p + "attn.out.bias",
		FFNNorm: p + "norm_ffn.weight",
		Gate:    p + "ffn.gate.weight", Up: p + "ffn.up.weight",
		Down: p + "ffn.down.weight", DownB: p + "ffn.down.bias",
	}
}

func (n voxtralTensorNames) proj() (linear1, linear2 string) {
	if n.compact {
		return "adapter.0.weight", "adapter.2.weight"
	}
	return "proj.linear_1.weight", "proj.linear_2.weight"
}

func (n voxtralTensorNames) tokenEmbd() string {
	if n.compact {
		return "tok_embeddings.weight"
	}
	return "dec.token_embd.weight"
}

func (n voxtralTensorNames) outputNorm() string {
	if n.compact {
		return "norm.weight"
	}
	return "dec.output_norm.weight"
}

// timeEmbedInvFreq returns "" when this convention has no such tensor, so
// the caller must compute it from Time.EmbedDim/EmbedTheta instead.
func (n voxtralTensorNames) timeEmbedInvFreq() string {
	if n.compact {
		return ""
	}
	return "dec.time_embed.inv_freq"
}

func (n voxtralTensorNames) decBlock(l int) voxtralDecBlockNames {
	if n.compact {
		p := fmt.Sprintf("dec.blk.%d.", l)
		return voxtralDecBlockNames{
			AttnNorm: p + "attn_norm.weight",
			Q:        p + "attn_q.weight", K: p + "attn_k.weight", V: p + "attn_v.weight", O: p + "attn_o.weight",
			FFNNorm: p + "ffn_norm.weight",
			Ada1:    p + "ada0.weight", Ada2: p + "ada2.weight",
			Gate: p + "ffn_w1.weight", Up: p + "ffn_w3.weight", Down: p + "ffn_w2.weight",
		}
	}
	p := fmt.Sprintf("dec.blocks.%d.", l)
	return voxtralDecBlockNames{
		AttnNorm: p + "norm_attn.weight",
		Q:        p + "attn.q.weight", K: p + "attn.k.weight", V: p + "attn.v.weight", O: p + "attn.o.weight",
		FFNNorm: p + "norm_ffn.weight",
		Ada1:    p + "ada.linear_1.weight", Ada2: p + "ada.linear_2.weight",
		Gate: p + "ffn.gate.weight", Up: p + "ffn.up.weight", Down: p + "ffn.down.weight",
	}
}

// voxtralPeriodicHannWindow computes torch.hann_window(n, periodic=True):
// w[i] = 0.5 - 0.5*cos(2*pi*i/n). Used only as a fallback when a GGUF omits
// the checkpoint's own frontend.window tensor -- see voxtralTensorNames'
// doc comment for why that's a safe substitution here.
func voxtralPeriodicHannWindow(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// voxtralSinusoidalInvFreq computes the standard inverse-frequency table
// inv_freq[i] = 1/theta^(2i/dim) for i in [0, dim/2). Used only as a
// fallback when a GGUF omits the checkpoint's own dec.time_embed.inv_freq
// tensor -- see voxtralTensorNames' doc comment.
func voxtralSinusoidalInvFreq(dim int, theta float32) []float32 {
	half := dim / 2
	f := make([]float32, half)
	for i := range f {
		f[i] = float32(1 / math.Pow(float64(theta), float64(2*i)/float64(dim)))
	}
	return f
}

// transposeF32 returns a copy of a row-major [rows][cols] slice as
// [cols][rows].
func transposeF32(f []float32, rows, cols int) []float32 {
	out := make([]float32, len(f))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = f[r*cols+c]
		}
	}
	return out
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

	// Every metadata read below tries the "stt.*" key this loader was
	// originally written against first, then the "voxtral.*" key the
	// second known GGUF convention uses instead -- see voxtralTensorNames'
	// doc comment. win_length doubles as n_fft's fallback because the
	// "voxtral.*" convention has no separate n_fft key at all, and this
	// implementation already requires win_length==n_fft everywhere else
	// (computeVoxtralRealtimeMelSpectrogramContext enforces it explicitly).
	winLength := int(gguFirstU32(gguf, 0, "stt.frontend.win_length", "voxtral.audio.window_size"))
	mel := VoxtralRealtimeMelConfig{
		SampleRate:      int(gguFirstU32(gguf, 16000, "stt.frontend.sample_rate", "voxtral.audio.sample_rate")),
		NumMels:         int(gguFirstU32(gguf, 0, "stt.frontend.num_mels", "voxtral.audio.num_mel_bins")),
		NFFT:            int(gguFirstU32(gguf, uint32(winLength), "stt.frontend.n_fft")),
		HopLength:       int(gguFirstU32(gguf, 0, "stt.frontend.hop_length", "voxtral.audio.hop_length")),
		WinLength:       winLength,
		FMin:            gguFirstF32(gguf, 0, "stt.frontend.f_min"),
		FMax:            gguFirstF32(gguf, 8000, "stt.frontend.f_max"),
		Center:          gguFirstBool(gguf, true, "stt.frontend.center"),
		Dither:          gguFirstF32(gguf, 0, "stt.frontend.dither"),
		PreEmphasis:     gguFirstF32(gguf, 0, "stt.frontend.pre_emphasis"),
		GlobalLogMelMax: gguFirstF32(gguf, 0, "stt.frontend.global_log_mel_max", "voxtral.audio.global_log_mel_max"),
	}
	// PadMode/Normalize/MelNorm are informational only -- already baked
	// into the loaded filterbank/frontend behavior, never read by the mel
	// spectrogram computation -- so they're left blank when a convention
	// doesn't carry them rather than given fallback keys.
	mel.PadMode, _ = gguf.GetString("stt.frontend.pad_mode")
	mel.Normalize, _ = gguf.GetString("stt.frontend.normalize")
	mel.MelNorm, _ = gguf.GetString("stt.frontend.mel_norm")
	if mel.NumMels <= 0 || mel.NFFT <= 0 || mel.HopLength <= 0 || mel.WinLength <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: invalid mel frontend config %+v", mel)
	}

	enc := VoxtralRealtimeEncoderConfig{
		DModel:        int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.d_model", "voxtral.encoder.dim")),
		FFNDim:        int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.ffn_dim", "voxtral.encoder.hidden_dim")),
		HeadDim:       int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.head_dim", "voxtral.encoder.head_dim")),
		NHeads:        int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.n_heads", "voxtral.encoder.n_heads")),
		NKVHeads:      int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.n_kv_heads", "voxtral.encoder.n_kv_heads")),
		NLayers:       int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.n_layers", "voxtral.encoder.n_layers")),
		NumMelBins:    int(gguFirstU32(gguf, uint32(mel.NumMels), "stt.voxtral_realtime.encoder.num_mel_bins", "voxtral.audio.num_mel_bins")),
		SlidingWindow: int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.encoder.sliding_window", "voxtral.encoder.sliding_window")),
		RopeTheta:     gguFirstF32(gguf, 1e6, "stt.voxtral_realtime.encoder.rope_theta", "voxtral.encoder.rope_theta"),
		Epsilon:       gguFirstF32(gguf, 1e-5, "stt.voxtral_realtime.encoder.rms_norm_eps", "voxtral.encoder.norm_eps"),
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
		// input_dim has no "voxtral.*" key at all; it's always exactly
		// downsample_factor*DModel (validated again where it's consumed,
		// forwardVoxtralEncoderChunk), so that product is both the
		// fallback and, for the original convention, a cross-check.
		InputDim:          int(gguFirstU32(gguf, uint32(enc.DModel*int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.projector.downsample_factor", "voxtral.audio.downsample_factor"))), "stt.voxtral_realtime.projector.input_dim")),
		DownsampleFactor:  int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.projector.downsample_factor", "voxtral.audio.downsample_factor")),
		AudioLengthPerTok: int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.projector.audio_length_per_tok")),
	}
	if proj.InputDim <= 0 || proj.DownsampleFactor <= 0 {
		return VoxtralRealtimeConfig{}, VoxtralRealtimeWeights{}, fmt.Errorf("loading voxtral realtime model: invalid projector config %+v", proj)
	}

	dec := VoxtralRealtimeDecoderConfig{
		HiddenSize:            int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.hidden_size", "voxtral.decoder.dim")),
		IntermediateSize:      int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.intermediate_size", "voxtral.decoder.hidden_dim")),
		HeadDim:               int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.head_dim", "voxtral.decoder.head_dim")),
		NHeads:                int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.n_heads", "voxtral.decoder.n_heads")),
		NKVHeads:              int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.n_kv_heads", "voxtral.decoder.n_kv_heads")),
		NLayers:               int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.n_layers", "voxtral.decoder.n_layers")),
		VocabSize:             int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.vocab_size", "voxtral.params.vocab_size", "voxtral.vocab_size")),
		SlidingWindow:         int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.sliding_window", "voxtral.decoder.sliding_window")),
		RopeTheta:             gguFirstF32(gguf, 1e6, "stt.voxtral_realtime.decoder.rope_theta", "voxtral.decoder.rope_theta"),
		Epsilon:               gguFirstF32(gguf, 1e-5, "stt.voxtral_realtime.decoder.rms_norm_eps", "voxtral.decoder.norm_eps"),
		MaxPositionEmbeddings: int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.decoder.max_position_embeddings", "voxtral.params.model_max_length")),
		TieWordEmbeddings:     gguFirstBool(gguf, true, "stt.voxtral_realtime.decoder.tie_word_embeddings", "voxtral.params.tied_embeddings"),
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
		// embed_dim/ada_hidden have no "voxtral.*" keys; embed_dim always
		// equals the decoder hidden size (this file's own doc comment on
		// VoxtralRealtimeTimeConfig.EmbedDim, cross-checked by
		// voxtralRealtimeTimeCond's own len(invFreq)==EmbedDim/2 == also
		// dec.HiddenSize/2 check) and ada_hidden is the ada MLP's
		// bottleneck width, carried under a differently-named but
		// identical-purpose key in that convention.
		EmbedDim:              int(gguFirstU32(gguf, uint32(dec.HiddenSize), "stt.voxtral_realtime.time.embed_dim")),
		AdaHidden:             int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.time.ada_hidden", "voxtral.params.ada_rms_norm_t_cond_dim", "voxtral.ada_rms_norm_t_cond_dim")),
		EmbedTheta:            gguFirstF32(gguf, 10000, "stt.voxtral_realtime.time.embed_theta"),
		DefaultNumDelayTokens: int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.time.default_num_delay_tokens", "voxtral.audio.n_delay_tokens")),
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
		StreamingPadTokenID: int(gguFirstU32(gguf, 0, "stt.voxtral_realtime.streaming_pad_token_id", "voxtral.token.streaming_pad")),
	}

	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	names := detectVoxtralTensorNames(tensors)
	config.RopeInterleaved = names.compact
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, false, false, useMetal)
	}
	loadConv := func(weightName, biasName string, kernel, in, out, stride int) (VoxtralRealtimeEncoderConv, error) {
		info, ok := tensors[weightName]
		if !ok {
			return VoxtralRealtimeEncoderConv{}, fmt.Errorf("tensor %s not found", weightName)
		}
		if want := kernel * in * out; int(info.Numel()) != want {
			return VoxtralRealtimeEncoderConv{}, fmt.Errorf("tensor %s has %d elements, want %d (kernel=%d in=%d out=%d)", weightName, info.Numel(), want, kernel, in, out)
		}
		w, err := loadF32Vec(data, gguf.DataOffset, weightName, tensors, inferred)
		if err != nil {
			return VoxtralRealtimeEncoderConv{}, err
		}
		b, err := loadF32Vec(data, gguf.DataOffset, biasName, tensors, inferred)
		if err != nil {
			return VoxtralRealtimeEncoderConv{}, err
		}
		return VoxtralRealtimeEncoderConv{Weight: w, Bias: b, Kernel: kernel, In: in, Out: out, Stride: stride}, nil
	}

	weights := VoxtralRealtimeWeights{}

	// The STFT window is precomputed data in one convention but absent from
	// the other; when absent it's a fixed, reproducible periodic Hann
	// window (see voxtralTensorNames' doc comment), not something the
	// loader can fail on.
	if windowName := names.window(); windowName != "" {
		weights.MelWindow, _ = loadF32Vec(data, gguf.DataOffset, windowName, tensors, inferred)
	} else {
		weights.MelWindow = voxtralPeriodicHannWindow(mel.WinLength)
	}
	if len(weights.MelWindow) != mel.WinLength {
		return config, weights, fmt.Errorf("loading voxtral realtime model: mel window has %d elements, want win_length=%d", len(weights.MelWindow), mel.WinLength)
	}
	filterbankName, filterbankTransposed := names.filterbank()
	if filterbankTransposed {
		// This convention stores the filterbank as [NumMels][NFFT/2+1] in
		// GGUF's native (fastest-dim-first) order, the transpose of the
		// [NFFT/2+1][NumMels] row-major layout Weight.MatvecInto needs
		// (Cols must equal the caller's input length, NFFT/2+1) --
		// verified against a working file in the other convention: same
		// values, dims reported in the opposite order. A plain load()
		// here would silently read the wrong stride instead of failing,
		// so this is a real transpose, not just a relabeling.
		raw, err := loadF32Vec(data, gguf.DataOffset, filterbankName, tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		nFreq := mel.NFFT/2 + 1
		if len(raw) != mel.NumMels*nFreq {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %s has %d elements, want num_mels*(n_fft/2+1)=%d", filterbankName, len(raw), mel.NumMels*nFreq)
		}
		weights.MelFilterbank = Weight{F32: transposeF32(raw, nFreq, mel.NumMels)}
	} else {
		melFilterbank, err := load(filterbankName)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		weights.MelFilterbank = melFilterbank
	}

	conv0Name, conv0Bias := names.conv(0)
	conv0, err := loadConv(conv0Name, conv0Bias, 3, mel.NumMels, enc.DModel, 1)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	conv1Name, conv1Bias := names.conv(1)
	conv1, err := loadConv(conv1Name, conv1Bias, 3, enc.DModel, enc.DModel, 2)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	weights.Encoder.Conv = [2]VoxtralRealtimeEncoderConv{conv0, conv1}

	weights.Encoder.FinalNorm, err = loadF32Vec(data, gguf.DataOffset, names.encFinalNorm(), tensors, inferred)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}

	encLayers := make([]VoxtralRealtimeEncoderLayer, 0, enc.NLayers)
	for l := range enc.NLayers {
		n := names.encBlock(l)
		var layer VoxtralRealtimeEncoderLayer
		layer.AttnNorm, err = loadF32Vec(data, gguf.DataOffset, n.AttnNorm, tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.Q, err = load(n.Q); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.K, err = load(n.K); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.V, err = load(n.V); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.Out, err = load(n.Out); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		qkvOutDim := enc.NHeads * enc.HeadDim
		layer.QB = loadOptionalF32Vec(data, gguf.DataOffset, n.QB, tensors, inferred, qkvOutDim)
		layer.VB = loadOptionalF32Vec(data, gguf.DataOffset, n.VB, tensors, inferred, qkvOutDim)
		layer.OutB = loadOptionalF32Vec(data, gguf.DataOffset, n.OutB, tensors, inferred, enc.DModel)
		layer.FFNNorm, err = loadF32Vec(data, gguf.DataOffset, n.FFNNorm, tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNGate, err = load(n.Gate); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNUp, err = load(n.Up); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNDown, err = load(n.Down); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		layer.FFNDownB = loadOptionalF32Vec(data, gguf.DataOffset, n.DownB, tensors, inferred, enc.DModel)
		encLayers = append(encLayers, layer)
		if l == 0 || l+1 == enc.NLayers || (l+1)%8 == 0 {
			fmt.Fprintf(logw, "  Loaded voxtral encoder layer %d/%d\n", l+1, enc.NLayers)
		}
	}
	weights.Encoder.Layers = encLayers

	projLinear1, projLinear2 := names.proj()
	if weights.Projector.Linear1, err = load(projLinear1); err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	if weights.Projector.Linear2, err = load(projLinear2); err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}

	if weights.Decoder.TokenEmbd, err = load(names.tokenEmbd()); err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	weights.Decoder.OutputNorm, err = loadF32Vec(data, gguf.DataOffset, names.outputNorm(), tensors, inferred)
	if err != nil {
		return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
	}
	// The time-embedding inverse-frequency table is precomputed data in one
	// convention but absent from the other; when absent it's the standard
	// sinusoidal-embedding formula applied to this checkpoint's own
	// EmbedDim/EmbedTheta (see voxtralTensorNames' doc comment), not
	// something the loader can fail on.
	if invFreqName := names.timeEmbedInvFreq(); invFreqName != "" {
		weights.Decoder.TimeEmbedInvFreq, err = loadF32Vec(data, gguf.DataOffset, invFreqName, tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
	} else {
		weights.Decoder.TimeEmbedInvFreq = voxtralSinusoidalInvFreq(timeCfg.EmbedDim, timeCfg.EmbedTheta)
	}
	if want := dec.HiddenSize / 2; len(weights.Decoder.TimeEmbedInvFreq) != want {
		return config, weights, fmt.Errorf("loading voxtral realtime model: time embedding inv_freq has %d elements, want hidden_size/2=%d", len(weights.Decoder.TimeEmbedInvFreq), want)
	}

	decLayers := make([]VoxtralRealtimeDecoderLayer, 0, dec.NLayers)
	for l := range dec.NLayers {
		n := names.decBlock(l)
		var layer VoxtralRealtimeDecoderLayer
		layer.AttnNorm, err = loadF32Vec(data, gguf.DataOffset, n.AttnNorm, tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.Q, err = load(n.Q); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.K, err = load(n.K); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.V, err = load(n.V); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.O, err = load(n.O); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		layer.FFNNorm, err = loadF32Vec(data, gguf.DataOffset, n.FFNNorm, tensors, inferred)
		if err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.AdaLinear1, err = load(n.Ada1); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.AdaLinear2, err = load(n.Ada2); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNGate, err = load(n.Gate); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNUp, err = load(n.Up); err != nil {
			return config, weights, fmt.Errorf("loading voxtral realtime model: %w", err)
		}
		if layer.FFNDown, err = load(n.Down); err != nil {
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
