package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// Flux3ImageKleinModel implements native text-to-image inference for BFL's
// base and flow-map (distilled) checkpoints. Reference-image editing and
// layout conditioning are not supported. It owns all mapped component files.
type Flux3ImageKleinModel struct {
	gate        chan struct{}
	encoder     *flux3TextEncoder
	transformer *flux3Transformer
	vae         *flux3VAE
	stores      []*qi21Weights
}

func flux3Shard(path, stem string) (string, error) {
	var found string
	for _, ext := range []string{".safetensors", ".gguf"} {
		name := stem + ext
		if _, err := os.Stat(filepath.Join(path, name)); err == nil {
			if found != "" {
				return "", fmt.Errorf("flux3: ambiguous %s weights; keep one format", stem)
			}
			found = name
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
	}
	if found == "" {
		return "", fmt.Errorf("flux3: missing %s.safetensors or %s.gguf (BFL native layout required)", stem, stem)
	}
	return found, nil
}

// OpenFlux3ImageKlein loads model.safetensors, ae.safetensors and text_encoder/
// from BFL's native repository layout, or their Q8_0 GGUF counterparts made by
// scripts/download_flux3.py. Large matrices remain packed and mmap-backed.
// NVFP4/FP8 CUDA variants and Diffusers-renamed tensors are not accepted.
func OpenFlux3ImageKlein(ctx context.Context, path string) (_ *Flux3ImageKleinModel, err error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	ditFile, err := flux3Shard(path, "model")
	if err != nil {
		return nil, err
	}
	vaeFile, err := flux3Shard(path, "ae")
	if err != nil {
		return nil, err
	}
	m := &Flux3ImageKleinModel{gate: make(chan struct{}, 1)}
	defer func() {
		if err != nil {
			_ = m.Close()
		}
	}()
	var config, tokenizer json.RawMessage
	if err = readModelJSON(filepath.Join(path, "text_encoder", "config.json"), 128<<10, &config); err != nil {
		return nil, err
	}
	if err = readModelJSON(filepath.Join(path, "text_encoder", "tokenizer.json"), 16<<20, &tokenizer); err != nil {
		return nil, err
	}
	for _, item := range []struct{ dir, file string }{{filepath.Join(path, "text_encoder"), ""}, {path, ditFile}, {path, vaeFile}} {
		var s *qi21Weights
		if item.file == "" {
			s, err = qi21OpenWeights(ctx, item.dir)
		} else {
			s, err = qi21OpenWeights(ctx, item.dir, item.file)
		}
		if err != nil {
			return nil, err
		}
		m.stores = append(m.stores, s)
	}
	if m.encoder, err = newFlux3TextEncoder(config, tokenizer, m.stores[0]); err != nil {
		return nil, err
	}
	if m.transformer, err = newFlux3Transformer(ctx, m.stores[1]); err != nil {
		return nil, err
	}
	if m.transformer.contextDim != 8*m.encoder.core.config.Dim {
		return nil, errors.New("flux3: incompatible text encoder and transformer")
	}
	if m.vae, err = newFlux3VAE(ctx, m.stores[2]); err != nil {
		return nil, err
	}
	m.stores[2].close()
	m.stores = m.stores[:2]
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *Flux3ImageKleinModel) Close() error {
	if m == nil || m.gate == nil {
		return nil
	}
	m.gate <- struct{}{}
	defer func() { <-m.gate }()
	m.encoder, m.transformer, m.vae = nil, nil, nil
	for _, s := range m.stores {
		s.close()
	}
	m.stores = nil
	return nil
}

// Generate defaults to 256x256 and 16 flow-map steps (base: 50 steps, CFG 4).
// Dimensions must be multiples of 16. Workspaces are capped independently of
// the mapped weights. Seed is reproducible in Go, not PyTorch's RNG sequence.
func (m *Flux3ImageKleinModel) Generate(ctx context.Context, opts ImageGenerationOptions) (*image.NRGBA, error) {
	return m.generate(ctx, opts, nil)
}

func flux3Schedule(steps int) ([]float32, error) {
	if steps < 1 || steps > 1000 {
		return nil, errors.New("flux3: steps must be 1..1000")
	}
	s := make([]float32, steps+1)
	for i := range s {
		t := float32(1) - float32(i)/float32(steps)
		s[i] = 16 * t / (1 + 15*t)
	}
	return s, nil
}

func (m *Flux3ImageKleinModel) generate(ctx context.Context, opts ImageGenerationOptions, noise []float32) (*image.NRGBA, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.gate == nil {
		return nil, errors.New("flux3: model is closed")
	}
	if strings.TrimSpace(opts.Prompt) == "" || len(opts.Prompt) > 64<<10 || len(opts.NegativePrompt) > 64<<10 {
		return nil, errors.New("flux3: supply a nonempty prompt of at most 64 KiB")
	}
	if opts.Width == 0 {
		opts.Width = 256
	}
	if opts.Height == 0 {
		opts.Height = 256
	}
	if opts.Width < 64 || opts.Height < 64 || opts.Width > 2048 || opts.Height > 2048 || opts.Width%16 != 0 || opts.Height%16 != 0 {
		return nil, errors.New("flux3: dimensions must be multiples of 16 between 64 and 2048")
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
		return nil, errors.New("flux3: model is closed")
	}
	flow := m.transformer.flowMap
	if opts.Steps == 0 {
		opts.Steps = 16
		if !flow {
			opts.Steps = 50
		}
	}
	if opts.GuidanceScale == 0 {
		opts.GuidanceScale = 1
		if !flow {
			opts.GuidanceScale = 4
		}
	}
	if !finite32(opts.GuidanceScale) || opts.GuidanceScale < 1 || opts.GuidanceScale > 20 || (flow && opts.GuidanceScale != 1) || (opts.GuidanceScale == 1 && opts.NegativePrompt != "") {
		return nil, errors.New("flux3: guidance must be 1..20; distilled checkpoints require 1 and no negative prompt")
	}
	schedule, err := flux3Schedule(opts.Steps)
	if err != nil {
		return nil, err
	}
	h, w := opts.Height/16, opts.Width/16
	if err := m.vae.validateSize(h, w); err != nil {
		return nil, err
	}
	text, err := m.encoder.encode(ctx, opts.Prompt)
	if err != nil {
		return nil, err
	}
	state, err := m.transformer.newState(text, h, w)
	if err != nil {
		return nil, err
	}
	var negative *flux3State
	if opts.GuidanceScale > 1 {
		text, err := m.encoder.encode(ctx, opts.NegativePrompt)
		if err != nil {
			return nil, err
		}
		negative, err = m.transformer.newState(text, h, w)
		if err != nil {
			return nil, err
		}
	}
	latents := make([]float32, h*w*128)
	if noise != nil {
		if len(noise) != len(latents) {
			return nil, errors.New("flux3: invalid initial noise shape")
		}
		copy(latents, noise)
	} else {
		rng := rand.New(rand.NewSource(opts.Seed))
		for i := range latents {
			latents[i] = float32(rng.NormFloat64())
		}
	}
	for step := 0; step < opts.Steps; step++ {
		t, next := schedule[step], schedule[step+1]
		velocity, err := state.predict(ctx, latents, t, next)
		if err != nil {
			return nil, err
		}
		if negative != nil {
			unconditional, err := negative.predict(ctx, latents, t, next)
			if err != nil {
				return nil, err
			}
			for i := range velocity {
				velocity[i] = unconditional[i] + opts.GuidanceScale*(velocity[i]-unconditional[i])
			}
		}
		for i := range latents {
			latents[i] += (next - t) * velocity[i]
		}
		if opts.Progress != nil {
			opts.Progress(step+1, opts.Steps)
		}
	}
	return m.vae.decode(ctx, latents, h, w)
}
