package mobile

import (
	"fmt"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type wireMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// parseMessages decodes a JSON array of {"role","content"} chat messages.
func parseMessages(raw string) ([]gopherllm.ChatMessage, error) {
	var wire []wireMessage
	if err := decodeStrictJSON(raw, &wire); err != nil {
		return nil, fmt.Errorf("invalid messages: %w", err)
	}
	if len(wire) == 0 {
		return nil, fmt.Errorf("invalid messages: at least one message is required")
	}
	messages := make([]gopherllm.ChatMessage, len(wire))
	for i, m := range wire {
		var role gopherllm.ChatRole
		switch m.Role {
		case "system":
			role = gopherllm.ChatRoleSystem
		case "user":
			role = gopherllm.ChatRoleUser
		case "assistant":
			role = gopherllm.ChatRoleAssistant
		default:
			return nil, fmt.Errorf("invalid messages: message %d has role %q, want system, user or assistant", i, m.Role)
		}
		messages[i] = gopherllm.ChatMessage{Role: role, Content: m.Content}
	}
	return messages, nil
}
