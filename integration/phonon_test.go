package integration_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
	"unicode"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/audioinput"
)

// TestPhonon25Transcription checks the actual multilingual checkpoint with
// known English and German utterances. The model and recordings stay outside Git.
func TestPhonon25Transcription(t *testing.T) {
	path := os.Getenv("GOPHERLLM_PHONON_MODEL")
	english := os.Getenv("GOPHERLLM_PHONON_ENGLISH_WAV")
	german := os.Getenv("GOPHERLLM_PHONON_GERMAN_WAV")
	if path == "" || (english == "" && german == "") {
		t.Skip("set GOPHERLLM_PHONON_MODEL and at least one Phonon WAV fixture")
	}
	start := time.Now()
	model, err := gopherllm.OpenParakeet(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = model.Close() })
	t.Logf("loaded Phonon in %s", time.Since(start))
	for _, tc := range []struct{ name, audio, want string }{
		{"english", english, "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait"},
		{"german", german, "diese paare können sich für einen adoptionsplan für ihr baby entscheiden"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.audio == "" {
				t.Skip("no WAV fixture")
			}
			samples, err := audioinput.Load16kMono(tc.audio)
			if err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			got, err := model.TranscribeOffline(context.Background(), samples, nil)
			if err != nil {
				t.Fatal(err)
			}
			if normalizedSpeech(got) != tc.want {
				t.Fatalf("got %q, want %q after normalization", got, tc.want)
			}
			t.Logf("transcribed %.2f s of audio in %s: %q", float64(len(samples))/16000, time.Since(start), got)
		})
	}
}

func normalizedSpeech(text string) string {
	text = strings.ReplaceAll(strings.ToLower(text), "’", "'")
	return strings.Join(strings.FieldsFunc(text, func(r rune) bool {
		return r != '\'' && !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}), " ")
}
