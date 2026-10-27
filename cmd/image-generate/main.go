// Command image-generate runs native Qwen/FLUX text-to-image generation
// or the unconditional CIFAR-10 reference model.
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"os/signal"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/outputfile"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "local Qwen-Image-2.1 / FLUX 3 Klein directory or DDPM-CIFAR-10 model")
	prompt := flag.String("prompt", "", "text prompt; omit for unconditional CIFAR generation")
	negative := flag.String("negative-prompt", "", "negative prompt for classifier-free guidance")
	width := flag.Int("width", 0, "output width (Qwen: multiple of 32, FLUX: 16; default 256)")
	height := flag.Int("height", 0, "output height (Qwen: multiple of 32, FLUX: 16; default 256)")
	guidance := flag.Float64("guidance", 0, "guidance scale (1 disables CFG; 0 uses checkpoint default)")
	output := flag.String("out", "", "new PNG output file; existing files are not overwritten")
	seed := flag.Int64("seed", 1, "random seed")
	steps := flag.Int("steps", 0, "denoising steps; 0 uses checkpoint default (Qwen Turbo: 8, Qwen base: 40, FLUX: 16/50, CIFAR: 50)")
	threads := flag.Int("threads", 0, "CPU worker count (0 uses runtime default)")
	timeout := flag.Duration("timeout", 30*time.Minute, "maximum load and generation time")
	flag.Parse()
	if *modelPath == "" || *output == "" || flag.NArg() != 0 {
		return fmt.Errorf("usage: image-generate -model DIRECTORY -out image.png [-seed 1] [-steps 50]")
	}
	if *steps < 0 || *steps > 1000 || *timeout <= 0 || *threads < 0 {
		return fmt.Errorf("steps must be 0..1000, timeout positive and threads nonnegative")
	}
	if err := outputfile.CheckNew(*output); err != nil {
		return err
	}
	if *threads > 0 {
		gopherllm.SetNumThreads(*threads)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	start := time.Now()
	var img image.Image
	var err error
	actualSteps := *steps
	if *prompt != "" {
		model, loadErr := gopherllm.OpenImageModel(ctx, *modelPath)
		if loadErr != nil {
			return loadErr
		}
		defer model.Close()
		img, err = model.Generate(ctx, gopherllm.ImageGenerationOptions{Prompt: *prompt, NegativePrompt: *negative, Width: *width, Height: *height, GuidanceScale: float32(*guidance), Seed: *seed, Steps: *steps,
			Progress: func(done, total int) { actualSteps = total; fmt.Fprintf(os.Stderr, "denoising %d/%d\n", done, total) },
		})
	} else {
		if *width != 0 || *height != 0 || *negative != "" || (*guidance != 0 && *guidance != 1) {
			return fmt.Errorf("dimensions and guidance require a text prompt")
		}
		model, loadErr := gopherllm.OpenDiffusion(ctx, *modelPath)
		if loadErr != nil {
			return loadErr
		}
		defer model.Close()
		img, err = model.Generate(ctx, gopherllm.DiffusionOptions{Seed: *seed, Steps: *steps})
		if actualSteps == 0 {
			actualSteps = 50
		}
	}
	if err != nil {
		return err
	}
	elapsed := time.Since(start)
	if err := outputfile.Write(*output, func(w io.Writer) error { return png.Encode(w, img) }); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: %dx%d image, %d steps including load in %s\n", *output, img.Bounds().Dx(), img.Bounds().Dy(), actualSteps, elapsed.Round(time.Millisecond))
	return nil
}
