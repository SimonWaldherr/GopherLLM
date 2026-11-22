// Command parakeet-transcribe transcribes audio with Parakeet-TDT GGUF models
// and unpacked Phonon models.
//
// Usage:
//
//	go run ./cmd/parakeet-transcribe --model parakeet-tdt-0.6b-v3.q8_0.gguf --audio clip.wav
package main

import (
	"flag"
	"fmt"
	"os"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/audioinput"
)

func main() {
	modelPath := flag.String("model", "", "path to a Parakeet-TDT GGUF or a Phonon model.fermion next to config.json")
	audioPath := flag.String("audio", "", "path to an audio file (any format if ffmpeg is on PATH; otherwise a 16-bit or float32 PCM WAV)")
	verbose := flag.Bool("v", false, "print encode/decode progress to stderr")
	flag.Parse()

	if *modelPath == "" || *audioPath == "" {
		fmt.Fprintln(os.Stderr, "usage: parakeet-transcribe --model MODEL --audio clip.wav [-v]")
		os.Exit(2)
	}

	samples, err := audioinput.Load16kMono(*audioPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: loading audio:", err)
		os.Exit(1)
	}
	var logf func(string, ...any)
	if *verbose {
		logf = func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }
	}
	text, err := gopherllm.TranscribeParakeet(*modelPath, samples, logf)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Println(text)
}
