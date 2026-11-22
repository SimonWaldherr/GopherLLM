// Command whisper-transcribe runs GopherLLM's native Whisper recognizer.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"github.com/SimonWaldherr/GopherLLM/internal/audioinput"
	"github.com/SimonWaldherr/GopherLLM/speech"
)

func main() {
	model := flag.String("model", "", "legacy GGML Whisper F16/F32 checkpoint")
	audio := flag.String("audio", "", "16 kHz mono PCM16 WAV, at most 30 seconds")
	language := flag.String("language", "auto", "Whisper language code or auto")
	flag.Parse()
	if *model == "" || *audio == "" {
		flag.Usage()
		os.Exit(2)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	if err := run(ctx, *model, *audio, *language); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, model, audio, language string) error {
	data, err := os.ReadFile(audio)
	if err != nil {
		return err
	}
	samples, err := audioinput.Decode16kMonoWAV(data, 30)
	if err != nil {
		return err
	}
	recognizer, err := speech.Open(ctx, model, speech.Options{Backend: speech.Whisper, Language: language}, os.Stderr)
	if err != nil {
		return err
	}
	defer recognizer.Close()
	text, err := recognizer.Transcribe(ctx, samples)
	if err != nil {
		return err
	}
	fmt.Println(text)
	return nil
}
