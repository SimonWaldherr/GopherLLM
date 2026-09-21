// Command parakeet-transcribe is a minimal, standalone smoke-test CLI for
// GopherLLM's Parakeet-TDT support (see parakeet.go and the other
// parakeet_*.go files in the module root).
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
)

func main() {
	modelPath := flag.String("model", "", "path to a Parakeet-TDT GGUF (general.architecture=asr, asr.head_type=tdt)")
	audioPath := flag.String("audio", "", "path to an audio file (any format if ffmpeg is on PATH; otherwise a 16-bit or float32 PCM WAV)")
	verbose := flag.Bool("v", false, "print encode/decode progress to stderr")
	flag.Parse()

	if *modelPath == "" || *audioPath == "" {
		fmt.Fprintln(os.Stderr, "usage: parakeet-transcribe --model parakeet.gguf --audio clip.wav [-v]")
		os.Exit(2)
	}

	samples, err := loadAudio16kMono(*audioPath)
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
