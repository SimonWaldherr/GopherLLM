package mobile

import (
	"errors"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// StreamSink receives a streamed generation: OnDelta once per new piece of
// text, in order, then exactly one of OnComplete (with the result JSON Chat
// returns) or OnError. Calls arrive on the generating thread before the
// streaming method returns; UI code must hop to its main thread itself.
//
// Every outcome, including invalid input, is reported through the sink as
// well as the streaming method's return value, so callers may rely on either.
// A callback may call the Engine's Cancel, IsLoaded, ModelName and InfoJSON;
// any other Engine method called from inside a callback can deadlock.
type StreamSink interface {
	OnDelta(string)
	OnComplete(string)
	OnError(string)
}

var errNilSink = errors.New("stream sink is nil")

// GenerateStream is Generate with incremental delivery to sink.
func (e *Engine) GenerateStream(prompt, optionsJSON string, sink StreamSink) error {
	if sink == nil {
		return errNilSink
	}
	return e.stream(singlePrompt(prompt), optionsJSON, sink)
}

// ChatStream is Chat with incremental delivery to sink.
func (e *Engine) ChatStream(messagesJSON, optionsJSON string, sink StreamSink) error {
	if sink == nil {
		return errNilSink
	}
	messages, err := parseMessages(messagesJSON)
	if err != nil {
		sink.OnError(err.Error())
		return err
	}
	return e.stream(messages, optionsJSON, sink)
}

func (e *Engine) stream(messages []gopherllm.ChatMessage, optionsJSON string, sink StreamSink) error {
	result, err := e.run(messages, optionsJSON, func(delta string) error {
		sink.OnDelta(delta)
		return nil
	})
	if err != nil {
		sink.OnError(err.Error())
		return err
	}
	sink.OnComplete(resultJSON(result))
	return nil
}
