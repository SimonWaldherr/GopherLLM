package gopherllm

import (
	"fmt"
	"strings"
)

// ContextWindowMode controls how a request behaves when its rendered chat
// history would crowd out the model's context window. The zero value keeps the
// API's historical full-history behavior.
type ContextWindowMode string

const (
	// ContextWindowFull sends every supplied message and preserves the existing
	// context-limit error when the prompt is too large.
	ContextWindowFull ContextWindowMode = "full"
	// ContextWindowRecent retains the newest complete user turn(s), dropping
	// older complete turns only when necessary to reserve room for generation.
	ContextWindowRecent ContextWindowMode = "recent"
	// ContextWindowAutoCompress first applies a loss-aware lexical compression
	// to ordinary chat messages, then retains newest complete turns only when
	// the compressed history still exceeds the context budget.
	ContextWindowAutoCompress ContextWindowMode = "autoCompress"
)

// contextWindowSafetyTokens leaves room for template bookkeeping and the
// assistant turn marker beyond the caller's requested completion budget.
const contextWindowSafetyTokens = 8

// ContextWindowInfo describes the exact, template-aware prompt selected for a
// request. The UI uses it to explain that older messages remain saved locally
// even when they were not sent to the model for this reply.
type ContextWindowInfo struct {
	Mode               ContextWindowMode `json:"mode"`
	ContextLength      int               `json:"context_length"`
	PromptBudget       int               `json:"prompt_budget"`
	PromptTokens       int               `json:"prompt_tokens"`
	InputMessages      int               `json:"input_messages"`
	RetainedMessages   int               `json:"retained_messages"`
	DroppedMessages    int               `json:"dropped_messages"`
	CompressedMessages int               `json:"compressed_messages"`
}

func (m ContextWindowMode) valid() bool {
	return m == "" || m == ContextWindowFull || m == ContextWindowRecent || m == ContextWindowAutoCompress
}

func normalizedContextWindowMode(m ContextWindowMode) ContextWindowMode {
	if m == "" {
		return ContextWindowFull
	}
	return m
}

// PrepareChatContext selects the messages that fit the requested context mode
// using the model's actual tokenizer and chat template. The bounded modes
// never split a user turn: an assistant tool-call and all of its tool results
// stay with the user message that caused them, so the model never receives an
// orphaned tool result or a dangling tool call.
//
// The full original history is never mutated. Callers such as the web UI can
// therefore keep it in local storage while passing the returned suffix to the
// model. A latest turn that alone cannot fit is rejected instead of silently
// slicing user text.
func (r *Runner) PrepareChatContext(messages []ChatMessage, options GenerationOptions) ([]ChatMessage, ContextWindowInfo, error) {
	mode := normalizedContextWindowMode(options.ContextWindowMode)
	info := ContextWindowInfo{
		Mode:          mode,
		ContextLength: r.config.MaxSeqLen,
		InputMessages: len(messages),
	}
	if !mode.valid() {
		return nil, info, fmt.Errorf("context_window_mode must be full, recent, or autoCompress")
	}
	if r.config.MaxSeqLen <= 0 {
		return nil, info, fmt.Errorf("model has an invalid context length (%d)", r.config.MaxSeqLen)
	}

	if mode == ContextWindowFull {
		info.PromptBudget = r.config.MaxSeqLen
		info.PromptTokens = len(r.renderMessages(messages, options.SystemPrompt, options.ActiveTools()))
		info.RetainedMessages = len(messages)
		return messages, info, nil
	}

	working := messages
	if mode == ContextWindowAutoCompress {
		compressed, changed := autoCompressChatMessages(messages)
		// Compression is only useful when the loaded model's actual tokenizer
		// agrees that it saves prompt tokens. This avoids trading clear prose for
		// abbreviations that happen to tokenize less efficiently.
		if changed > 0 && len(r.renderMessages(compressed, options.SystemPrompt, options.ActiveTools())) < len(r.renderMessages(messages, options.SystemPrompt, options.ActiveTools())) {
			working = compressed
			info.CompressedMessages = changed
		}
	}

	reserve := max(1, options.MaxTokens) + contextWindowSafetyTokens
	info.PromptBudget = r.config.MaxSeqLen - reserve
	if info.PromptBudget < 1 {
		return nil, info, fmt.Errorf("context length %d cannot reserve %d completion tokens plus safety margin", r.config.MaxSeqLen, reserve)
	}

	pinned, turns, newestUserTurn := chatContextTurns(working)
	if newestUserTurn < 0 {
		return nil, info, fmt.Errorf("bounded context mode requires at least one user message")
	}

	var kept []ChatMessage
	for first := newestUserTurn; first >= 0; first-- {
		candidate := joinChatContextTurns(pinned, turns[first:])
		tokens := r.renderMessages(candidate, options.SystemPrompt, options.ActiveTools())
		if len(tokens) > info.PromptBudget {
			if first == newestUserTurn {
				info.PromptTokens = len(tokens)
				return nil, info, fmt.Errorf("latest user turn needs %d prompt tokens, but recent context can use at most %d after reserving output space", len(tokens), info.PromptBudget)
			}
			break
		}
		kept = candidate
		info.PromptTokens = len(tokens)
	}
	if kept == nil {
		return nil, info, fmt.Errorf("recent context could not retain the latest user turn")
	}
	info.RetainedMessages = len(kept)
	info.DroppedMessages = len(messages) - len(kept)
	return kept, info, nil
}

// autoCompressionTerms converts common long-form expressions into established
// technical terms or abbreviations. This deliberately operates only on normal
// conversational text; tool payloads and fenced code are opaque data whose
// syntax must never be rewritten.
var autoCompressionTerms = strings.NewReplacer(
	"as soon as possible", "ASAP",
	"for example", "e.g.",
	"that is to say", "i.e.",
	"in other words", "i.e.",
	"application programming interface", "API",
	"large language model", "LLM",
	"retrieval augmented generation", "RAG",
	"context window", "ctx window",
	"zum Beispiel", "z. B.",
	"das heißt", "d. h.",
	"beziehungsweise", "bzw.",
	"unter anderem", "u. a.",
	"im Hinblick auf", "bzgl.",
	"künstliche Intelligenz", "KI",
	"maschinelles Lernen", "ML",
	"großes Sprachmodell", "LLM",
	"Anwendungsprogrammierschnittstelle", "API",
)

// autoCompressChatMessages returns a same-shaped history so existing turn and
// tool-call grouping remains valid. It is intentionally conservative: a
// transformed message must be shorter, and the caller additionally verifies
// a real token reduction before using the returned history.
func autoCompressChatMessages(messages []ChatMessage) ([]ChatMessage, int) {
	compressed := append([]ChatMessage(nil), messages...)
	changed := 0
	for i, message := range messages {
		if message.Role == ChatRoleTool || len(message.ToolCalls) != 0 || strings.Contains(message.Content, "```") {
			continue
		}
		content := autoCompressMessage(message.Content)
		if content != "" && len(content) < len(message.Content) {
			compressed[i].Content = content
			changed++
		}
	}
	return compressed, changed
}

func autoCompressMessage(content string) string {
	content = strings.Join(strings.Fields(content), " ")
	if content == "" {
		return ""
	}
	content = autoCompressionTerms.Replace(content)
	for _, prefix := range []string{
		"please ", "Please ", "could you ", "Could you ",
		"I would like you to ", "Bitte ", "bitte ",
		"Könntest du ", "könntest du ", "Ich möchte gerne, dass du ",
	} {
		content = strings.TrimPrefix(content, prefix)
	}
	return strings.TrimSpace(content)
}

// chatContextTurns keeps leading system instructions pinned. Every later user
// message starts one complete turn; all following assistant and tool messages
// belong to that turn until the next user message starts. Non-user preambles
// form an older disposable group rather than becoming orphans.
func chatContextTurns(messages []ChatMessage) (pinned []ChatMessage, turns [][]ChatMessage, newestUserTurn int) {
	for len(pinned) < len(messages) && messages[len(pinned)].Role == ChatRoleSystem {
		pinned = append(pinned, messages[len(pinned)])
	}
	newestUserTurn = -1
	for _, message := range messages[len(pinned):] {
		if message.Role == ChatRoleUser {
			turns = append(turns, []ChatMessage{message})
			newestUserTurn = len(turns) - 1
			continue
		}
		if len(turns) == 0 {
			turns = append(turns, []ChatMessage{message})
			continue
		}
		turns[len(turns)-1] = append(turns[len(turns)-1], message)
	}
	return pinned, turns, newestUserTurn
}

func joinChatContextTurns(pinned []ChatMessage, turns [][]ChatMessage) []ChatMessage {
	n := len(pinned)
	for _, turn := range turns {
		n += len(turn)
	}
	out := make([]ChatMessage, 0, n)
	out = append(out, pinned...)
	for _, turn := range turns {
		out = append(out, turn...)
	}
	return out
}
