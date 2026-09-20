package mobile

import (
	"encoding/json"
	"fmt"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// StreamSink is implemented by Swift/Objective-C. Callbacks happen on a Go
// worker thread; UI clients must hop to the main actor/queue before mutation.
type StreamSink interface {
	OnDelta(string)
	OnComplete(string)
	OnError(string)
}

func (e *Engine) GenerateStream(prompt, optionsJSON string, sink StreamSink) (err error) {
	if sink == nil {
		return fmt.Errorf("stream sink is nil")
	}
	o, err := parseGenerationOptions(optionsJSON)
	if err != nil {
		sink.OnError(err.Error())
		return err
	}
	e.opMu.Lock()
	defer e.opMu.Unlock()
	m, ctx, done, err := e.startGeneration()
	if err != nil {
		sink.OnError(err.Error())
		return err
	}
	defer done()
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("generation failed: %v", r)
			sink.OnError(err.Error())
		}
	}()
	result, err := m.Stream(ctx, []gopherllm.ChatMessage{gopherllm.UserMessage(prompt)}, func(delta string) error { sink.OnDelta(delta); return nil }, o.coreOptions()...)
	if err != nil {
		err = fmt.Errorf("generation failed: %w", err)
		sink.OnError(err.Error())
		return err
	}
	b, _ := json.Marshal(struct {
		Text            string `json:"text"`
		FinishReason    string `json:"finish_reason"`
		GeneratedTokens int    `json:"generated_tokens"`
	}{result.Text, result.FinishReason, result.Stats.GeneratedTokens})
	sink.OnComplete(string(b))
	return nil
}
