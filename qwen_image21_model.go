package gopherllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math/rand"
	"path/filepath"
	"strings"
)

// QwenImage21Options controls native text-to-image generation. Zero dimensions
// use 256x256 for a manageable local workload; the official pipeline defaults
// to 2048x2048. Dimensions must be multiples of 32, between 32 and 2048.
// Zero Steps uses 40 for the base model or the checkpoint's fixed Turbo schedule.
// Zero GuidanceScale means 1 (no classifier-free guidance). A scale above 1
// requires NegativePrompt and executes a second transformer branch each step.
type QwenImage21Options struct {
	Prompt, NegativePrompt string
	Width, Height, Steps   int
	Seed                   int64
	GuidanceScale          float32
	// Progress is called synchronously after each completed denoising step.
	// It must not re-enter this model. The total reflects the checkpoint schedule.
	Progress func(completed, total int)
}

// QwenImage21Model owns native text-to-image components. Image editing and the
// vision encoder are not supported. Requests serialize their workspaces; no
// Python process, network service or model-supplied code runs during inference.
type QwenImage21Model struct {
	gate        chan struct{}
	encoder     *qi21TextEncoder
	transformer *qi21Transformer
	vae         *qwenImage21VAE
	scheduler   qi21SchedulerConfig
	sigmas      []float32
	stores      []*qi21Weights
}

// OpenQwenImage21 loads a local Diffusers layout or the compact GGUF component
// layout produced by scripts/download_qwen_image21.py. Large matrices stay
// mmap-backed; this function does not download model files.
func OpenQwenImage21(ctx context.Context, path string) (_ *QwenImage21Model, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var index struct {
		ClassName    string    `json:"_class_name"`
		SampleSigmas []float32 `json:"sample_sigmas"`
		Transformer  []string  `json:"transformer"`
		TextEncoder  []string  `json:"text_encoder"`
		VAE          []string  `json:"vae"`
	}
	if err := readModelJSON(filepath.Join(path, "model_index.json"), 64<<10, &index); err != nil {
		return nil, err
	}
	component := func(v []string, want string) bool { return len(v) == 2 && v[1] == want }
	if index.ClassName != "QwenImage21Pipeline" || !component(index.Transformer, "QwenImage21Transformer2DModel") || !component(index.TextEncoder, "Qwen3VLForConditionalGeneration") || !component(index.VAE, "AutoencoderKLQwenImage21") {
		return nil, errors.New("qwen-image-2.1: unsupported pipeline components")
	}
	m := &QwenImage21Model{gate: make(chan struct{}, 1), sigmas: index.SampleSigmas}
	defer func() {
		if err != nil {
			_ = m.Close()
		}
	}()
	var transformerConfig qi21TransformerConfig
	var vaeConfig qwenImage21VAEConfig
	var encoderConfig, tokenizerJSON json.RawMessage
	for _, item := range []struct {
		path  string
		limit int64
		value any
	}{
		{"transformer/config.json", 64 << 10, &transformerConfig},
		{"vae/config.json", 64 << 10, &vaeConfig},
		{"text_encoder/config.json", 128 << 10, &encoderConfig},
		{"processor/tokenizer.json", 16 << 20, &tokenizerJSON},
		{"scheduler/scheduler_config.json", 64 << 10, &m.scheduler},
	} {
		if err = readModelJSON(filepath.Join(path, item.path), item.limit, item.value); err != nil {
			return nil, err
		}
	}
	if _, err = qi21Schedule(m.scheduler, 40, 256, m.sigmas); err != nil {
		return nil, err
	}
	for _, name := range []string{"text_encoder", "transformer", "vae"} {
		var store *qi21Weights
		store, err = qi21OpenWeights(ctx, filepath.Join(path, name))
		if err != nil {
			return nil, fmt.Errorf("qwen-image-2.1 %s: %w", name, err)
		}
		m.stores = append(m.stores, store)
	}
	if m.encoder, err = newQI21TextEncoder(encoderConfig, tokenizerJSON, m.stores[0]); err != nil {
		return nil, err
	}
	if m.transformer, err = qi21NewTransformer(ctx, m.stores[1], transformerConfig); err != nil {
		return nil, err
	}
	if m.encoder.config.Dim != transformerConfig.ContextDim || transformerConfig.OutChannels != vaeConfig.ZDim {
		return nil, errors.New("qwen-image-2.1: incompatible component dimensions")
	}
	if m.vae, err = loadQwenImage21VAE(ctx, m.stores[2], vaeConfig); err != nil {
		return nil, err
	}
	// The image decoder owns its floating-point tensors; unlike the
	// two transformers it retains no views into the source mapping.
	m.stores[2].close()
	m.stores = m.stores[:2]
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *QwenImage21Model) Close() error {
	if m == nil || m.gate == nil {
		return nil
	}
	m.gate <- struct{}{}
	defer func() { <-m.gate }()
	m.encoder, m.transformer, m.vae = nil, nil, nil
	for _, store := range m.stores {
		store.close()
	}
	m.stores = nil
	return nil
}

// Generate returns a straight-alpha RGBA image owned by the caller. Seed fixes
// Go's Gaussian sequence; equal seeds do not reproduce PyTorch's RNG sequence.
func (m *QwenImage21Model) Generate(ctx context.Context, opts QwenImage21Options) (*image.NRGBA, error) {
	return m.generate(ctx, opts, nil)
}

// Fixed initial noise permits differential tests independent of RNG algorithms.
func (m *QwenImage21Model) generate(ctx context.Context, opts QwenImage21Options, initialNoise []float32) (*image.NRGBA, error) {
	if m == nil || m.gate == nil {
		return nil, errors.New("qwen-image-2.1: model is closed")
	}
	if strings.TrimSpace(opts.Prompt) == "" || len(opts.Prompt) > 64<<10 || len(opts.NegativePrompt) > 64<<10 {
		return nil, errors.New("qwen-image-2.1: supply a nonempty prompt of at most 64 KiB")
	}
	if opts.Width == 0 {
		opts.Width = 256
	}
	if opts.Height == 0 {
		opts.Height = 256
	}
	if opts.Width < 32 || opts.Height < 32 || opts.Width > 2048 || opts.Height > 2048 || opts.Width%32 != 0 || opts.Height%32 != 0 {
		return nil, errors.New("qwen-image-2.1: dimensions must be multiples of 32 between 32 and 2048")
	}
	if opts.GuidanceScale == 0 {
		opts.GuidanceScale = 1
	}
	if !finite32(opts.GuidanceScale) || opts.GuidanceScale < 1 || opts.GuidanceScale > 20 || (opts.GuidanceScale > 1 && opts.NegativePrompt == "") || (opts.GuidanceScale == 1 && opts.NegativePrompt != "") {
		return nil, errors.New("qwen-image-2.1: guidance must be 1..20; negative prompt requires guidance above 1")
	}
	select {
	case m.gate <- struct{}{}:
		defer func() { <-m.gate }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.encoder == nil {
		return nil, errors.New("qwen-image-2.1: model is closed")
	}
	if opts.Steps == 0 {
		opts.Steps = 40
		if len(m.sigmas) > 0 {
			opts.Steps = len(m.sigmas)
		}
	}
	if len(m.sigmas) > 0 && opts.Steps != len(m.sigmas) {
		return nil, fmt.Errorf("qwen-image-2.1: checkpoint requires its fixed %d-step schedule", len(m.sigmas))
	}
	h, w := opts.Height/16, opts.Width/16
	schedule, err := qi21Schedule(m.scheduler, opts.Steps, h*w, m.sigmas)
	if err != nil {
		return nil, err
	}
	text, _, err := m.encoder.Encode(ctx, opts.Prompt)
	if err != nil {
		return nil, err
	}
	state, err := m.transformer.newState(text, h, w)
	if err != nil {
		return nil, err
	}
	var negative *qi21TransformerState
	if opts.GuidanceScale > 1 {
		text, _, err = m.encoder.Encode(ctx, opts.NegativePrompt)
		if err != nil {
			return nil, err
		}
		negative, err = m.transformer.newState(text, h, w)
		if err != nil {
			return nil, err
		}
	}
	latents := make([]float32, h*w*m.transformer.cfg.InChannels)
	if initialNoise != nil {
		if len(initialNoise) != len(latents) {
			return nil, errors.New("qwen-image-2.1: invalid initial noise shape")
		}
		for _, v := range initialNoise {
			if !finite32(v) {
				return nil, errors.New("qwen-image-2.1: non-finite initial noise")
			}
		}
		copy(latents, initialNoise)
	} else {
		rng := rand.New(rand.NewSource(opts.Seed))
		for i := range latents {
			latents[i] = float32(rng.NormFloat64())
		}
	}
	for step := 0; step < len(schedule)-1; step++ {
		prediction, e := state.predict(ctx, latents, schedule[step])
		if e != nil {
			return nil, e
		}
		if negative != nil {
			other, e := negative.predict(ctx, latents, schedule[step])
			if e != nil {
				return nil, e
			}
			for i := range prediction {
				prediction[i] = other[i] + opts.GuidanceScale*(prediction[i]-other[i])
			}
		}
		if err = qi21EulerStep(latents, prediction, schedule[step], schedule[step+1]); err != nil {
			return nil, err
		}
		if opts.Progress != nil {
			opts.Progress(step+1, len(schedule)-1)
		}
	}
	return m.vae.decode(ctx, latents, h, w)
}
