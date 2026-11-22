package hestia

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

const (
	assistantQuestionBytes = 2048
	assistantReplyTokens   = 96
	assistantReplyTimeout  = 30 * time.Second
	assistantSystemPrompt  = "Du bist Hestia, ein lokaler Sprachassistent. Antworte knapp, klar und in der Sprache der Frage. Sage, wenn du etwas nicht weißt. Du hast hier keinen Zugriff auf Geräte, private Dokumente, das Internet oder aktuelle Messwerte. Behaupte nicht, eine Aktion ausgeführt zu haben."
)

// AssistantChat answers a single local conversation question.
type AssistantChat interface {
	Answer(context.Context, string) (string, error)
}

type assistantChatModel interface {
	Chat(context.Context, []gopherllm.ChatMessage, ...gopherllm.GenOption) (gopherllm.GenerationResult, error)
	Close() error
}

// LocalAssistantChat loads a local GGUF once. The engine serializes inference
// and owns its working memory. Hestia sends one new question per request.
type LocalAssistantChat struct {
	model assistantChatModel
}

func OpenAssistantChat(path string) (*LocalAssistantChat, error) {
	model, err := gopherllm.Open(context.Background(), path, gopherllm.WithMetal(gopherllm.MetalAvailable()))
	if err != nil {
		return nil, fmt.Errorf("opening local chat model: %w", err)
	}
	switch model.Runner().Architecture() {
	case "bert", "nomic-bert":
		_ = model.Close()
		return nil, errors.New("local chat requires a generative GGUF model")
	}
	return &LocalAssistantChat{model: model}, nil
}

func (c *LocalAssistantChat) Close() error {
	return c.model.Close()
}

// Answer bounds question size, inference time and generated tokens. It adds
// neither conversation history nor retrieved data and offers no tools.
func (c *LocalAssistantChat) Answer(ctx context.Context, question string) (string, error) {
	question = strings.TrimSpace(question)
	if question == "" || len(question) > assistantQuestionBytes {
		return "", errors.New("local chat question is empty or too long")
	}
	ctx, cancel := context.WithTimeout(ctx, assistantReplyTimeout)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	result, err := c.model.Chat(ctx, []gopherllm.ChatMessage{gopherllm.UserMessage(question)},
		gopherllm.WithSystemPrompt(assistantSystemPrompt),
		gopherllm.WithMaxTokens(assistantReplyTokens),
		gopherllm.WithTemperature(0),
		gopherllm.WithToolChoice("none"),
	)
	if err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	reply := strings.TrimSpace(result.Text)
	if reply == "" || len(result.ToolCalls) > 0 {
		return "", errors.New("local chat produced no plain-text answer")
	}
	return reply, nil
}
