package gopherllm

import (
	"context"
	"errors"
	"image"
	"os"
	"path/filepath"
)

// ImageGenerationOptions controls native text-to-image generation. Dimensions,
// step defaults and guidance support depend on the checkpoint. Progress runs
// synchronously and must not call back into the same model.
type ImageGenerationOptions struct {
	Prompt, NegativePrompt string
	Width, Height, Steps   int
	Seed                   int64
	GuidanceScale          float32
	Progress               func(completed, total int)
}

// ImageGenerationModel owns its mapped weights and serializes generation.
// Close releases them after any active generation finishes.
type ImageGenerationModel interface {
	Generate(context.Context, ImageGenerationOptions) (*image.NRGBA, error)
	Close() error
}

// OpenImageModel recognizes a local Qwen-Image-2.1 pipeline or a FLUX 3 Image
// Klein checkpoint in BFL's native layout. It never downloads or executes code.
func OpenImageModel(ctx context.Context, path string) (ImageGenerationModel, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if _, err := os.Stat(filepath.Join(path, "model_index.json")); err == nil {
		m, err := OpenQwenImage21(ctx, path)
		if err != nil {
			return nil, err
		}
		return m, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	m, err := OpenFlux3ImageKlein(ctx, path)
	if err != nil {
		return nil, err
	}
	return m, nil
}
