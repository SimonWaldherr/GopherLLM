package mobile

import (
	"context"
	"fmt"
	"io"

	"github.com/SimonWaldherr/GopherLLM/speech"
)

func OpenSpeech(ctx context.Context, path, optionsJSON string) (model *speech.Recognizer, err error) {
	defer recoverError("opening speech recognizer", &err)
	var opts speech.Options
	if err := decodeStrictJSON(optionsJSON, &opts); err != nil {
		return nil, fmt.Errorf("invalid speech options: %w", err)
	}
	return speech.Open(ctx, path, opts, io.Discard)
}

// TranscribeSpeech optionally filters a transcript in Go before exposing any
// text to an app. A nil activation phrase is ordinary push-to-talk mode.
func TranscribeSpeech(ctx context.Context, model *speech.Recognizer, samples []float32, activation *string) (_ string, err error) {
	defer recoverError("transcribing speech", &err)
	var wake speech.WakePhrase
	if activation != nil {
		if *activation == "" {
			return "", fmt.Errorf("activation phrase must be non-empty")
		}
		wake, err = speech.ParseWakePhrase(*activation)
		if err != nil {
			return "", err
		}
	}
	text, err := model.Transcribe(ctx, samples)
	if err != nil {
		return "", err
	}
	result := struct {
		Text    string `json:"text"`
		Matched *bool  `json:"matched,omitempty"`
	}{Text: text}
	if activation != nil {
		command, matched := wake.Match(text)
		result.Text, result.Matched = command, &matched
	}
	return marshal(result), nil
}
