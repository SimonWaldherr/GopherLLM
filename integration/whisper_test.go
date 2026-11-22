package integration_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/SimonWaldherr/GopherLLM/internal/audioinput"
	"github.com/SimonWaldherr/GopherLLM/speech"
)

func TestNativeWhisperBase(t *testing.T) {
	model, audio := os.Getenv("GOPHERLLM_WHISPER_MODEL"), os.Getenv("GOPHERLLM_WHISPER_WAV")
	if model == "" || audio == "" {
		t.Skip("set GOPHERLLM_WHISPER_MODEL to multilingual Base F16 and GOPHERLLM_WHISPER_WAV to the German FLEURS 0015 clip")
	}
	ctx := context.Background()
	raw, err := os.ReadFile(audio)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := audioinput.Decode16kMonoWAV(raw, 30)
	if err != nil {
		t.Fatal(err)
	}
	for _, language := range []string{"auto", "de"} {
		t.Run(language, func(t *testing.T) {
			r, err := speech.Open(ctx, model, speech.Options{Backend: speech.Whisper, Language: language}, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			for range 2 {
				start := time.Now()
				text, err := r.Transcribe(ctx, samples)
				const expected = "diese paare können sich für einen adoptionsplan für ihr baby entscheiden"
				if err != nil || strings.TrimRight(strings.ToLower(text), ".") != expected {
					t.Fatalf("transcription %q, error %v", text, err)
				}
				t.Logf("%v: %s", time.Since(start), text)
			}
			if text, err := r.Transcribe(ctx, make([]float32, 16000)); err != nil || text != "" {
				t.Fatalf("silence produced %q, error %v", text, err)
			}
		})
	}
}
