// Command voxtral-transcribe is a minimal, standalone smoke-test CLI for
// GopherLLM's Voxtral Realtime support (see voxtral_realtime.go,
// voxtral_realtime_audio.go, voxtral_realtime_decoder.go in the module
// root). The CLI and web server use the same transcription implementation.
//
// Usage:
//
//	go run ./cmd/voxtral-transcribe --model Voxtral-Mini-4B-Realtime-2602-Q6_K.gguf --audio clip.wav
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func main() {
	modelPath := flag.String("model", "", "path to a Voxtral Realtime GGUF (general.architecture=voxtral_realtime), or a directory containing the official release's consolidated.safetensors + params.json + tekken.json")
	audioPath := flag.String("audio", "", "path to an audio file (any format if ffmpeg is on PATH; otherwise a 16-bit or float32 PCM WAV)")
	maxExtraSteps := flag.Int("max-extra-steps", 0, "additional right-padding audio tokens for the model to finish emitting text")
	verbose := flag.Bool("v", false, "print per-step decode progress to stderr")
	flag.Parse()

	if *modelPath == "" || *audioPath == "" {
		fmt.Fprintln(os.Stderr, "usage: voxtral-transcribe --model voxtral.gguf --audio clip.wav [-v]")
		os.Exit(2)
	}

	if err := run(*modelPath, *audioPath, *maxExtraSteps, *verbose); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(modelPath, audioPath string, maxExtraSteps int, verbose bool) error {
	samples, err := loadAudio16kMono(audioPath)
	if err != nil {
		return fmt.Errorf("loading audio: %w", err)
	}
	var logw io.Writer = io.Discard
	if verbose {
		logw = os.Stderr
	}
	var text string
	if info, statErr := os.Stat(modelPath); statErr == nil && info.IsDir() {
		text, err = gopherllm.TranscribeVoxtralRealtimeFromSafetensors(context.Background(), modelPath, samples, maxExtraSteps, logw)
	} else {
		text, err = gopherllm.TranscribeVoxtralRealtime(context.Background(), modelPath, samples, maxExtraSteps, logw)
	}
	if err != nil {
		return err
	}
	fmt.Println(text)
	return nil
}
