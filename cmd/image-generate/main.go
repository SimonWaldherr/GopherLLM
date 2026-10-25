// Command image-generate samples the native unconditional CIFAR-10 diffusion
// model. This first image-generation path produces 32x32 PNGs without a prompt.
package main

import (
	"context"
	"flag"
	"fmt"
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
	modelPath := flag.String("model", "", "local DDPM-CIFAR-10 model directory or Safetensors file")
	output := flag.String("out", "", "new PNG output file; existing files are not overwritten")
	seed := flag.Int64("seed", 1, "random seed")
	steps := flag.Int("steps", 50, "deterministic DDIM denoising steps (1..1000)")
	threads := flag.Int("threads", 0, "CPU worker count (0 uses runtime default)")
	timeout := flag.Duration("timeout", 5*time.Minute, "maximum load and generation time")
	flag.Parse()
	if *modelPath == "" || *output == "" || flag.NArg() != 0 {
		return fmt.Errorf("usage: image-generate -model DIRECTORY -out image.png [-seed 1] [-steps 50]")
	}
	if *steps < 1 || *steps > 1000 || *timeout <= 0 || *threads < 0 {
		return fmt.Errorf("steps must be 1..1000, timeout positive and threads nonnegative")
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
	model, err := gopherllm.OpenDiffusion(ctx, *modelPath)
	if err != nil {
		return err
	}
	defer model.Close()
	start := time.Now()
	img, err := model.Generate(ctx, gopherllm.DiffusionOptions{Seed: *seed, Steps: *steps})
	if err != nil {
		return err
	}
	elapsed := time.Since(start)
	if err := outputfile.Write(*output, func(w io.Writer) error { return png.Encode(w, img) }); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: %dx%d image, %d steps in %s\n", *output, img.Bounds().Dx(), img.Bounds().Dy(), *steps, elapsed.Round(time.Millisecond))
	return nil
}
