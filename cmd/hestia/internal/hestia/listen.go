package hestia

import (
	"context"
	"fmt"

	"github.com/SimonWaldherr/GopherLLM/speech"
)

type ListenResult struct {
	Matched bool  `json:"matched"`
	Turn    *Turn `json:"turn,omitempty"`
}

var errListenDisabled = jsonErr("hestia: listen mode is disabled")

// Listen transcribes an isolated clip in memory. Only a recognized command
// after the leading activation phrase enters the turn pipeline.
func (a *App) Listen(ctx context.Context, conversationID string, pcm []byte) (ListenResult, error) {
	if a.ListenWord == "" {
		return ListenResult{}, errListenDisabled
	}
	if a.Speech == nil {
		return ListenResult{}, ErrSpeechNotConfigured
	}
	phrase, err := speech.ParseWakePhrase(a.ListenWord)
	if err != nil {
		return ListenResult{}, fmt.Errorf("listen_word: %w", err)
	}
	transcript, err := a.Speech.TranscribePCM16(ctx, pcm)
	if err != nil {
		return ListenResult{}, err
	}
	if err := ctx.Err(); err != nil {
		return ListenResult{}, err
	}
	command, matched := phrase.Match(transcript)
	if command == "" {
		return ListenResult{Matched: matched}, nil
	}
	turn, err := a.ProcessText(ctx, conversationID, command)
	return ListenResult{Matched: matched, Turn: turn}, err
}
