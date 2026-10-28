package engine

import (
	"fmt"
	"strings"
	"unicode"
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
	// ContextWindowAutoCompress first condenses older plain assistant prose,
	// preserving user/system instructions and the complete newest turn, then
	// retains newest complete turns when the history still exceeds the budget.
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
	// Recall reports only historical turns added to the selected prompt by
	// an opt-in conversation-memory planner. It is nil for ordinary planning.
	Recall *ContextRecallInfo `json:"recall,omitempty"`
}

// ContextRecallInfo accounts for conversation history retrieved into one
// selected prompt. Sources contains only turns that the model actually sees,
// rather than every turn considered by the retrieval search.
type ContextRecallInfo struct {
	Mode         string                `json:"mode"`
	IndexedTurns int                   `json:"indexed_turns"`
	TokenBudget  int                   `json:"token_budget"`
	AddedTokens  int                   `json:"added_tokens"`
	Sources      []ContextRecallSource `json:"sources"`
}

// ContextRecallSource identifies one original historical turn included as
// reference data. EndMessage is exclusive; offsets refer to the caller's
// original history, before context reduction or reference-data injection.
type ContextRecallSource struct {
	ID           string  `json:"id"`
	Turn         int     `json:"turn"`
	StartMessage int     `json:"start_message"`
	EndMessage   int     `json:"end_message"`
	Hash         string  `json:"hash"`
	Score        float32 `json:"score"`
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
	return r.prepareChatContext(messages, options, nil)
}

// prepareChatContext optionally hands the final owned text tokens to generation
// so it can avoid rendering the selected history a second time. Image prompts
// and FIM retain the generation renderer's complete multimodal/FIM transaction.
func (r *Runner) prepareChatContext(messages []ChatMessage, options GenerationOptions, renderedTokens *[]uint32) ([]ChatMessage, ContextWindowInfo, error) {
	if renderedTokens != nil {
		*renderedTokens = nil
		if options.FIMSuffix != "" {
			renderedTokens = nil
		}
		for _, message := range messages {
			if len(message.Images) > 0 {
				renderedTokens = nil
				break
			}
		}
	}
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
		tokens := r.renderMessages(messages, options.SystemPrompt, options.ActiveTools())
		info.PromptTokens = len(tokens)
		info.RetainedMessages = len(messages)
		if renderedTokens != nil {
			*renderedTokens = tokens
		}
		return messages, info, nil
	}
	if options.FIMSuffix != "" {
		return nil, info, fmt.Errorf("bounded context modes do not support FIM; use full context mode")
	}

	completion := max(1, options.MaxTokens)
	if completion > int(^uint(0)>>1)-contextWindowSafetyTokens {
		return nil, info, fmt.Errorf("context length %d cannot reserve %d completion tokens plus safety margin", r.config.MaxSeqLen, completion)
	}
	reserve := completion + contextWindowSafetyTokens
	info.PromptBudget = r.config.MaxSeqLen - reserve
	if info.PromptBudget < 1 {
		return nil, info, fmt.Errorf("context length %d cannot reserve %d completion tokens plus safety margin", r.config.MaxSeqLen, reserve)
	}

	newestUser := len(messages) - 1
	for newestUser >= 0 && messages[newestUser].Role != ChatRoleUser {
		newestUser--
	}
	if newestUser < 0 {
		return nil, info, fmt.Errorf("bounded context mode requires at least one user message")
	}

	// A complete prompt that actually fits needs no repeated suffix renders.
	// Avoid that extra whole-history render for obviously large recent-mode
	// inputs; the heuristic only decides whether to try this fast path. Every
	// selected candidate is still measured with the actual chat template.
	// Compression always measures the original first, so a fitting original
	// is never rewritten and token savings can be checked exactly.
	tools := options.ActiveTools()
	originalTokens := 0
	if mode == ContextWindowAutoCompress || tryFullChatContextProbe(messages, info.PromptBudget) {
		originalRendered := r.renderMessages(messages, options.SystemPrompt, tools)
		originalTokens = len(originalRendered)
		if originalTokens <= info.PromptBudget {
			info.PromptTokens, info.RetainedMessages = originalTokens, len(messages)
			if renderedTokens != nil {
				*renderedTokens = originalRendered
			}
			return messages, info, nil
		}
	}
	working := messages
	hasCompressed := false
	if mode == ContextWindowAutoCompress {
		compressed, changed := autoCompressChatMessages(messages)
		if changed > 0 {
			working, hasCompressed = compressed, true
			// The full compressed history may fit without any turn selection,
			// but avoid tokenizing discarded old text just to test that shortcut.
			if tryFullChatContextProbe(compressed, info.PromptBudget) {
				compressedRendered := r.renderMessages(compressed, options.SystemPrompt, tools)
				compressedTokens := len(compressedRendered)
				if compressedTokens < originalTokens && compressedTokens <= info.PromptBudget {
					info.CompressedMessages = changed
					info.PromptTokens, info.RetainedMessages = compressedTokens, len(working)
					if renderedTokens != nil {
						*renderedTokens = compressedRendered
					}
					return working, info, nil
				}
			}
		}
	}

	pinned, turns, newestUserTurn := chatContextTurns(working)
	// Candidate suffixes share one private message slab. Only the pinned
	// prefix moves between candidates; restore the previous prefix's slots
	// before extending so older turns always retain their original contents.
	// Without pinned messages, a suffix can be viewed directly with no copy.
	scratch := working
	if len(pinned) > 0 {
		scratch = append([]ChatMessage(nil), working...)
	}
	originalScratch := scratch
	if hasCompressed {
		originalScratch = messages
		if len(pinned) > 0 {
			originalScratch = append([]ChatMessage(nil), messages...)
		}
	}
	suffixStart, previousStart, keptStart := len(working), len(working), -1
	changedInSuffix, keptCompressed := 0, false
	var keptTokens []uint32
	for first := newestUserTurn; first >= 0; first-- {
		lastStart := suffixStart
		suffixStart -= len(turns[first])
		if hasCompressed {
			for i := suffixStart; i < lastStart; i++ {
				if working[i].Content != messages[i].Content {
					changedInSuffix++
				}
			}
		}
		if len(pinned) > 0 {
			if previousStart < len(working) {
				copy(scratch[previousStart-len(pinned):previousStart], working[previousStart-len(pinned):previousStart])
				if hasCompressed {
					copy(originalScratch[previousStart-len(pinned):previousStart], messages[previousStart-len(pinned):previousStart])
				}
			}
			copy(scratch[suffixStart-len(pinned):suffixStart], pinned)
			if hasCompressed {
				copy(originalScratch[suffixStart-len(pinned):suffixStart], pinned)
			}
		}
		candidate := scratch[suffixStart-len(pinned):]
		tokens := r.renderMessages(candidate, options.SystemPrompt, tools)
		useCompressed := false
		if changedInSuffix > 0 {
			// Savings must hold for the candidate actually being selected. A
			// shorter string can tokenize to more tokens, even if other changed
			// turns made the complete compressed history smaller overall.
			original := originalScratch[suffixStart-len(pinned):]
			originalTokens := r.renderMessages(original, options.SystemPrompt, tools)
			if len(tokens) < len(originalTokens) {
				useCompressed = true
			} else {
				tokens = originalTokens
			}
		}
		if len(tokens) > info.PromptBudget {
			if first == newestUserTurn {
				info.PromptTokens = len(tokens)
				return nil, info, fmt.Errorf("latest user turn needs %d prompt tokens, but recent context can use at most %d after reserving output space", len(tokens), info.PromptBudget)
			}
			break
		}
		keptStart, previousStart = suffixStart, suffixStart
		keptCompressed = useCompressed
		info.CompressedMessages = 0
		if useCompressed {
			info.CompressedMessages = changedInSuffix
		}
		if renderedTokens != nil {
			keptTokens = tokens
		}
		info.PromptTokens = len(tokens)
	}
	if keptStart < 0 {
		return nil, info, fmt.Errorf("recent context could not retain the latest user turn")
	}
	if hasCompressed && !keptCompressed {
		scratch = originalScratch
	}
	if len(pinned) > 0 {
		copy(scratch[keptStart-len(pinned):keptStart], pinned)
	}
	kept := scratch[keptStart-len(pinned):]
	info.RetainedMessages = len(kept)
	info.DroppedMessages = len(messages) - len(kept)
	if renderedTokens != nil {
		*renderedTokens = keptTokens
	}
	return kept, info, nil
}

// Large discarded histories should not be tokenized just to test a possible
// full-fit shortcut. Sixteen bytes per budget token is intentionally a loose
// guess, not a token bound: rejecting this probe preserves the exact existing
// suffix scan, while accepting it still requires an actual tokenizer check.
func tryFullChatContextProbe(messages []ChatMessage, budget int) bool {
	// Some templates let a later system message replace the leading one. In
	// that case a suffix can be much larger than the complete conversation,
	// so raw history bytes cannot decide whether to skip the full-fit check.
	seenConversation := false
	for _, message := range messages {
		if message.Role == ChatRoleSystem {
			if seenConversation {
				return true
			}
		} else {
			seenConversation = true
		}
	}
	if budget >= int(^uint(0)>>1)/16 {
		return true
	}
	remaining := budget * 16
	for _, message := range messages {
		if len(message.Images) > 0 || remaining < 8 || len(message.Content) > remaining-8 {
			return false
		}
		remaining -= 8 + len(message.Content)
		for _, call := range message.ToolCalls {
			if len(call.Function.Name) > remaining || len(call.Function.Arguments) > remaining-len(call.Function.Name) {
				return false
			}
			remaining -= len(call.Function.Name) + len(call.Function.Arguments)
		}
	}
	return true
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
	var compressed []ChatMessage
	changed := 0
	newestUser := len(messages)
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == ChatRoleUser {
			newestUser = i
			break
		}
	}
	for i, message := range messages {
		// User instructions and system policy are never lexical summaries.
		// Protect the newest complete turn and opaque/formatted data as well:
		// whitespace and apparently ordinary words can carry literal meaning.
		if i >= newestUser || message.Role != ChatRoleAssistant || len(message.ToolCalls) != 0 || len(message.Images) != 0 ||
			!chatWhitespaceNormalized(message.Content) || strings.ContainsAny(message.Content, "`~{}[]<>\"'|\\/*#+=:") ||
			strings.HasPrefix(message.Content, "- ") || len(message.Content) > 0 && message.Content[0] >= '0' && message.Content[0] <= '9' {
			continue
		}
		content := autoCompressMessage(message.Content)
		if content != "" && len(content) < len(message.Content) {
			if compressed == nil {
				compressed = append([]ChatMessage(nil), messages...)
			}
			compressed[i].Content = content
			changed++
		}
	}
	if compressed == nil {
		return messages, 0
	}
	return compressed, changed
}

func autoCompressMessage(content string) string {
	if !chatWhitespaceNormalized(content) {
		content = strings.Join(strings.Fields(content), " ")
	}
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

func chatWhitespaceNormalized(content string) bool {
	space := true
	for _, r := range content {
		if r == ' ' {
			if space {
				return false
			}
			space = true
		} else {
			if r >= '\t' && r <= '\r' || r >= 128 && unicode.IsSpace(r) {
				return false
			}
			space = false
		}
	}
	return !space || content == ""
}

// chatContextTurns keeps leading system instructions pinned. Every later user
// message starts one complete turn; all following assistant and tool messages
// belong to that turn until the next user message starts. Non-user preambles
// form an older disposable group rather than becoming orphans.
func chatContextTurns(messages []ChatMessage) (pinned []ChatMessage, turns [][]ChatMessage, newestUserTurn int) {
	n := 0
	for n < len(messages) && messages[n].Role == ChatRoleSystem {
		n++
	}
	if n > 0 {
		pinned = messages[:n:n]
	}
	newestUserTurn = -1
	start := n
	for i := n; i < len(messages); i++ {
		if messages[i].Role == ChatRoleUser {
			if i > start {
				turns = append(turns, messages[start:i:i])
			}
			start = i
			newestUserTurn = len(turns)
		}
	}
	if start < len(messages) {
		turns = append(turns, messages[start:len(messages):len(messages)])
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
