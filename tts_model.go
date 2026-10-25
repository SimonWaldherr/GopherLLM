package gopherllm

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/SimonWaldherr/GopherLLM/internal/mmapfile"
	"github.com/SimonWaldherr/GopherLLM/internal/numeric"
)

// SpeechAudio is mono, floating-point PCM at SampleRate samples per second.
type SpeechAudio struct {
	Samples    []float32
	SampleRate int
}

// TTSOptions controls MMS speech synthesis. Zero Speed means the model's
// default rate. Seed makes repeated calls reproducible within this engine;
// it does not select the same random sequence as PyTorch.
type TTSOptions struct {
	Seed  int64
	Speed float32
	// Nil preserves the checkpoint's noise level. A pointer to zero explicitly
	// disables that source of variation, for deterministic numerical comparisons.
	NoiseScale         *float32
	DurationNoiseScale *float32
}

type ttsMatrix struct {
	data             []float32
	channels, length int
}

type ttsWeight struct {
	data  []float32
	shape []int
}

type ttsVITSConfig struct {
	ModelType                       string  `json:"model_type"`
	HiddenSize                      int     `json:"hidden_size"`
	FlowSize                        int     `json:"flow_size"`
	FFNDim                          int     `json:"ffn_dim"`
	FFNKernelSize                   int     `json:"ffn_kernel_size"`
	HiddenAct                       string  `json:"hidden_act"`
	NumAttentionHeads               int     `json:"num_attention_heads"`
	NumHiddenLayers                 int     `json:"num_hidden_layers"`
	WindowSize                      int     `json:"window_size"`
	VocabSize                       int     `json:"vocab_size"`
	UseBias                         bool    `json:"use_bias"`
	LayerNormEps                    float32 `json:"layer_norm_eps"`
	NumSpeakers                     int     `json:"num_speakers"`
	SpeakerEmbeddingSize            int     `json:"speaker_embedding_size"`
	SamplingRate                    int     `json:"sampling_rate"`
	SpeakingRate                    float32 `json:"speaking_rate"`
	NoiseScale                      float32 `json:"noise_scale"`
	NoiseScaleDuration              float32 `json:"noise_scale_duration"`
	UseStochasticDurationPrediction bool    `json:"use_stochastic_duration_prediction"`
	DepthSeparableChannels          int     `json:"depth_separable_channels"`
	DepthSeparableNumLayers         int     `json:"depth_separable_num_layers"`
	DurationPredictorKernelSize     int     `json:"duration_predictor_kernel_size"`
	DurationPredictorNumFlows       int     `json:"duration_predictor_num_flows"`
	DurationPredictorFlowBins       int     `json:"duration_predictor_flow_bins"`
	DurationPredictorTailBound      float32 `json:"duration_predictor_tail_bound"`
	PriorEncoderNumFlows            int     `json:"prior_encoder_num_flows"`
	PriorEncoderNumWavenetLayers    int     `json:"prior_encoder_num_wavenet_layers"`
	WavenetKernelSize               int     `json:"wavenet_kernel_size"`
	WavenetDilationRate             int     `json:"wavenet_dilation_rate"`
	UpsampleInitialChannel          int     `json:"upsample_initial_channel"`
	UpsampleRates                   []int   `json:"upsample_rates"`
	UpsampleKernelSizes             []int   `json:"upsample_kernel_sizes"`
	ResblockKernelSizes             []int   `json:"resblock_kernel_sizes"`
	ResblockDilationSizes           [][]int `json:"resblock_dilation_sizes"`
	LeakyReLUSlope                  float32 `json:"leaky_relu_slope"`
}

// TTSModel runs the single-speaker MMS/VITS architecture directly in Go. The
// initial supported tokenizer is the German MMS character tokenizer. Model
// weights have their own license; OpenTTS never downloads or redistributes them.
type TTSModel struct {
	gate    chan struct{}
	weights map[string]ttsWeight
	config  ttsVITSConfig
	vocab   map[rune]int
}

const (
	ttsMaxTextRunes = 512
	ttsMaxFrames    = 2048 // 32.768 seconds at MMS's 256-sample hop, 16 kHz.
)

// OpenTTS loads config.json, tokenizer_config.json, vocab.json and
// model.safetensors from a local facebook/mms-tts-deu model directory.
// Training-only posterior and duration-posterior weights are not retained.
func OpenTTS(ctx context.Context, path string) (*TTSModel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m := &TTSModel{gate: make(chan struct{}, 1)}
	if err := ttsReadJSON(filepath.Join(path, "config.json"), &m.config); err != nil {
		return nil, err
	}
	if err := m.config.validate(); err != nil {
		return nil, err
	}
	var tok struct {
		Language  string `json:"language"`
		AddBlank  bool   `json:"add_blank"`
		Normalize bool   `json:"normalize"`
		Phonemize bool   `json:"phonemize"`
		IsUroman  bool   `json:"is_uroman"`
		PadToken  string `json:"pad_token"`
	}
	if err := ttsReadJSON(filepath.Join(path, "tokenizer_config.json"), &tok); err != nil {
		return nil, err
	}
	if tok.Language != "deu" || !tok.AddBlank || !tok.Normalize || tok.Phonemize || tok.IsUroman || tok.PadToken != " " {
		return nil, fmt.Errorf("tts: unsupported tokenizer; expected German MMS characters with blank insertion and no phonemizer")
	}
	var vocabulary map[string]int
	if err := ttsReadJSON(filepath.Join(path, "vocab.json"), &vocabulary); err != nil {
		return nil, err
	}
	if len(vocabulary) != m.config.VocabSize {
		return nil, fmt.Errorf("tts: vocabulary has %d entries, expected %d", len(vocabulary), m.config.VocabSize)
	}
	m.vocab = make(map[rune]int, len(vocabulary))
	seen := make([]bool, m.config.VocabSize)
	for symbol, id := range vocabulary {
		if utf8.RuneCountInString(symbol) != 1 || id < 0 || id >= len(seen) || seen[id] {
			return nil, fmt.Errorf("tts: invalid character vocabulary entry %q=%d", symbol, id)
		}
		seen[id] = true
		r, _ := utf8.DecodeRuneInString(symbol)
		m.vocab[r] = id
	}
	if id, ok := m.vocab[' ']; !ok || id != 0 {
		return nil, fmt.Errorf("tts: blank token must be space with ID 0")
	}
	modelPath := filepath.Join(path, "model.safetensors")
	st, err := os.Stat(modelPath)
	if err != nil {
		return nil, err
	}
	if !st.Mode().IsRegular() || st.Size() <= 8 || st.Size() > 256<<20 {
		return nil, fmt.Errorf("tts: model file exceeds the supported 256 MiB limit")
	}
	mapped, err := mmapfile.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer mapped.Close()
	data := mapped.Bytes()
	if len(data) < 8 || len(data) > 256<<20 || binary.LittleEndian.Uint64(data[:8]) > 1<<20 {
		return nil, fmt.Errorf("tts: invalid or oversized safetensors file")
	}
	file, err := ParseSafetensors(data)
	if err != nil {
		return nil, err
	}
	m.weights, err = ttsLoadWeights(ctx, file, m.config)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Close releases model weights after any active synthesis completes. It is
// safe to call repeatedly. Calls waiting to synthesize can be cancelled.
func (m *TTSModel) Close() error {
	if m == nil || m.gate == nil {
		return nil
	}
	m.gate <- struct{}{}
	defer func() { <-m.gate }()
	m.weights = nil
	return nil
}

// Synthesize converts one short German utterance to PCM. Inputs are limited to
// 512 Unicode characters and predicted audio to 32.768 seconds; split longer
// text at sentence boundaries. Speed must be between 0.25 and 4 when supplied.
func (m *TTSModel) Synthesize(ctx context.Context, text string, options TTSOptions) (SpeechAudio, error) {
	if m == nil || m.gate == nil {
		return SpeechAudio{}, fmt.Errorf("tts: nil model")
	}
	select {
	case m.gate <- struct{}{}:
	case <-ctx.Done():
		return SpeechAudio{}, ctx.Err()
	}
	defer func() { <-m.gate }()
	if err := ctx.Err(); err != nil {
		return SpeechAudio{}, err
	}
	if m.weights == nil {
		return SpeechAudio{}, fmt.Errorf("tts: model is closed")
	}
	ids, err := m.tokenize(text)
	if err != nil {
		return SpeechAudio{}, err
	}
	speed := options.Speed
	if speed == 0 {
		speed = m.config.SpeakingRate
	}
	if !finite32(speed) || speed < .25 || speed > 4 {
		return SpeechAudio{}, fmt.Errorf("tts: speed must be between 0.25 and 4")
	}
	noise, durationNoise := m.config.NoiseScale, m.config.NoiseScaleDuration
	if options.NoiseScale != nil {
		noise = *options.NoiseScale
	}
	if options.DurationNoiseScale != nil {
		durationNoise = *options.DurationNoiseScale
	}
	if !finite32(noise) || !finite32(durationNoise) || noise < 0 || noise > 2 || durationNoise < 0 || durationNoise > 2 {
		return SpeechAudio{}, fmt.Errorf("tts: noise scales must be between 0 and 2")
	}
	hidden, means, logs, err := m.encode(ctx, ids)
	if err != nil {
		return SpeechAudio{}, err
	}
	rng := rand.New(rand.NewSource(options.Seed))
	dnoise := make([]float32, 2*len(ids))
	for i := range dnoise {
		dnoise[i] = float32(rng.NormFloat64()) * durationNoise
	}
	logDuration, err := m.duration(ctx, hidden, dnoise)
	if err != nil {
		return SpeechAudio{}, err
	}
	durations, frames, err := ttsDurations(logDuration, speed)
	if err != nil {
		return SpeechAudio{}, err
	}
	latent := ttsMatrix{data: make([]float32, m.config.FlowSize*frames), channels: m.config.FlowSize, length: frames}
	for c := 0; c < latent.channels; c++ {
		pos := 0
		for t, d := range durations {
			mu, sigma := means.data[c*means.length+t], float32(math.Exp(float64(logs.data[c*logs.length+t])))
			for range d {
				latent.data[c*frames+pos] = mu + float32(rng.NormFloat64())*sigma*noise
				pos++
			}
		}
	}
	latent, err = m.flow(ctx, latent)
	if err != nil {
		return SpeechAudio{}, err
	}
	wave, err := ttsVocode(ctx, m.weights, m.config, latent)
	if err != nil {
		return SpeechAudio{}, err
	}
	if !numeric.AllFiniteF32(wave) {
		return SpeechAudio{}, fmt.Errorf("tts: model produced non-finite audio")
	}
	return SpeechAudio{Samples: wave, SampleRate: m.config.SamplingRate}, nil
}

func (m *TTSModel) tokenize(text string) ([]int, error) {
	if !utf8.ValidString(text) || utf8.RuneCountInString(text) > ttsMaxTextRunes {
		return nil, fmt.Errorf("tts: text must be valid UTF-8 and at most %d characters", ttsMaxTextRunes)
	}
	// MMS normalization lowercases and removes characters outside its alphabet;
	// it does not expand numbers, transliterate or invoke a phonemizer.
	var b strings.Builder
	for _, r := range strings.ToLower(text) {
		if _, ok := m.vocab[r]; ok {
			b.WriteRune(r)
		}
	}
	normalized := strings.TrimSpace(b.String())
	if normalized == "" {
		return nil, fmt.Errorf("tts: text contains no supported characters")
	}
	ids := make([]int, 1, 2*utf8.RuneCountInString(normalized)+1)
	for _, r := range normalized {
		ids = append(ids, m.vocab[r], 0)
	}
	return ids, nil
}

func ttsReadJSON(path string, value any) error {
	if err := readModelJSON(path, 1<<20, value); err != nil {
		return fmt.Errorf("tts: %w", err)
	}
	return nil
}

func (c ttsVITSConfig) validate() error {
	if c.ModelType != "vits" ||
		c.HiddenSize != 192 ||
		c.FlowSize != 192 ||
		c.FFNDim != 768 ||
		c.FFNKernelSize != 3 ||
		c.HiddenAct != "relu" ||
		c.NumAttentionHeads != 2 ||
		c.NumHiddenLayers != 6 ||
		c.WindowSize != 4 ||
		c.VocabSize != 45 ||
		!c.UseBias ||
		c.LayerNormEps != 1e-5 ||
		c.NumSpeakers != 1 ||
		c.SpeakerEmbeddingSize != 0 ||
		c.SamplingRate != 16000 ||
		!c.UseStochasticDurationPrediction ||
		c.DepthSeparableChannels != 2 ||
		c.DepthSeparableNumLayers != 3 ||
		c.DurationPredictorKernelSize != 3 ||
		c.DurationPredictorNumFlows != 4 ||
		c.DurationPredictorFlowBins != 10 ||
		c.DurationPredictorTailBound != 5 ||
		c.PriorEncoderNumFlows != 4 ||
		c.PriorEncoderNumWavenetLayers != 4 ||
		c.WavenetKernelSize != 5 ||
		c.WavenetDilationRate != 1 ||
		c.UpsampleInitialChannel != 512 ||
		!slices.Equal(c.UpsampleRates, []int{8, 8, 2, 2}) ||
		!slices.Equal(c.UpsampleKernelSizes, []int{16, 16, 4, 4}) ||
		!slices.Equal(c.ResblockKernelSizes, []int{3, 7, 11}) ||
		len(c.ResblockDilationSizes) != 3 ||
		c.LeakyReLUSlope != .1 {
		return fmt.Errorf("tts: unsupported VITS configuration; expected the single-speaker MMS German architecture")
	}
	for _, d := range c.ResblockDilationSizes {
		if !slices.Equal(d, []int{1, 3, 5}) {
			return fmt.Errorf("tts: unsupported vocoder dilations")
		}
	}
	if !finite32(c.SpeakingRate) ||
		c.SpeakingRate < .25 ||
		c.SpeakingRate > 4 ||
		!finite32(c.NoiseScale) ||
		c.NoiseScale < 0 ||
		c.NoiseScale > 2 ||
		!finite32(c.NoiseScaleDuration) ||
		c.NoiseScaleDuration < 0 ||
		c.NoiseScaleDuration > 2 {
		return fmt.Errorf("tts: invalid model synthesis defaults")
	}
	return nil
}

func ttsDurations(logs []float32, speed float32) ([]int, int, error) {
	d := make([]int, len(logs))
	total := 0
	for i, l := range logs {
		v := math.Ceil(float64(float32(math.Exp(float64(l))) / speed))
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > ttsMaxFrames-float64(total) {
			return nil, 0, fmt.Errorf("tts: predicted speech exceeds %.3f seconds; use shorter sentences", float64(ttsMaxFrames*256)/16000)
		}
		d[i] = int(v)
		total += d[i]
	}
	if total == 0 {
		return nil, 0, fmt.Errorf("tts: predicted empty speech")
	}
	return d, total, nil
}
