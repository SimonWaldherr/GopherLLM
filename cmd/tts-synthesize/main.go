// Command tts-synthesize runs native MMS/VITS text-to-speech and writes mono
// PCM WAV. It uses local model files and does not start a Python process.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"strings"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "local MMS/VITS model directory (config, tokenizer, vocabulary and Safetensors)")
	text := flag.String("text", "", "text to speak; - reads UTF-8 text from stdin")
	output := flag.String("out", "", "new WAV output file; existing files are not overwritten")
	seed := flag.Int64("seed", 1, "random seed for duration and waveform sampling")
	speed := flag.Float64("speed", 1, "speaking rate (1 is the model default)")
	threads := flag.Int("threads", 0, "CPU worker count (0 uses runtime default)")
	timeout := flag.Duration("timeout", 5*time.Minute, "maximum load and synthesis time")
	flag.Parse()
	if *modelPath == "" || *text == "" || *output == "" || flag.NArg() != 0 {
		return fmt.Errorf("usage: tts-synthesize -model DIRECTORY -text 'Hallo Welt' -out speech.wav")
	}
	if *timeout <= 0 || *threads < 0 {
		return fmt.Errorf("timeout must be positive and threads nonnegative")
	}
	if math.IsNaN(*speed) || *speed < .25 || *speed > 4 {
		return fmt.Errorf("speed must be between 0.25 and 4")
	}
	if _, err := os.Lstat(*output); err == nil {
		return fmt.Errorf("output already exists: %s", *output)
	} else if !os.IsNotExist(err) {
		return err
	}
	if *text == "-" {
		data, err := io.ReadAll(io.LimitReader(os.Stdin, 65537))
		if err != nil {
			return err
		}
		if len(data) > 65536 {
			return fmt.Errorf("stdin text exceeds 64 KiB")
		}
		*text = strings.TrimSpace(string(data))
	}
	if *threads > 0 {
		gopherllm.SetNumThreads(*threads)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	model, err := gopherllm.OpenTTS(ctx, *modelPath)
	if err != nil {
		return err
	}
	defer model.Close()
	start := time.Now()
	audio, err := model.Synthesize(ctx, *text, gopherllm.TTSOptions{Seed: *seed, Speed: float32(*speed)})
	if err != nil {
		return err
	}
	elapsed := time.Since(start)
	f, err := os.OpenFile(*output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	complete := false
	defer func() {
		_ = f.Close()
		if !complete {
			_ = os.Remove(*output)
		}
	}()
	if err := audio.WriteWAV(f); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	complete = true
	fmt.Fprintf(os.Stderr, "%s: %.2f s audio at %d Hz, synthesized in %s\n", *output, float64(len(audio.Samples))/float64(audio.SampleRate), audio.SampleRate, elapsed.Round(time.Millisecond))
	return nil
}
