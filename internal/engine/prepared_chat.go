package engine

import (
	"fmt"
	"reflect"
	"sync"
)

// ChatContextPreparer selects an exactly budgeted prompt. A nil preparer in
// PrepareChatGeneration uses the Runner's ordinary context-window planner.
// Custom preparers can add request-local context, such as conversation recall.
// A preparer must not start inference on the same Runner: preflight holds its
// inference lease to prevent Close from freeing model resources while rendering.
type ChatContextPreparer func([]ChatMessage, GenerationOptions) ([]ChatMessage, ContextWindowInfo, error)

// PreparedChat is a single-use, request-local result of chat preflight. Its
// tokens and messages are private: callers cannot substitute token IDs for a
// different prompt. It is bound to the Runner that prepared it.
type PreparedChat struct {
	mu     sync.Mutex
	runner *Runner
	data   *preparedChatData
}

type preparedChatData struct {
	runner                 *Runner
	input, selected        []ChatMessage
	options                GenerationOptions
	info                   ContextWindowInfo
	tokens                 []uint32
	mistralKVPrefixEpoch   uint64
	mistralKVPrefixEnabled bool
}

// PrepareChatGeneration performs preflight once and returns a plan whose Wrap
// method can reuse it for the first identical model call. Every message field
// and generation option must match; any mismatch consumes the plan and falls
// back to normal generation. Later tool rounds therefore always plan again.
//
// Images and FIM use their complete generation renderer and return a nil plan
// after ordinary preflight. A nil plan's Wrap method simply returns fallback.
// No global cache or caller-supplied conversation identifier is involved.
func (r *Runner) PrepareChatGeneration(messages []ChatMessage, options GenerationOptions, prepare ChatContextPreparer) (*PreparedChat, error) {
	if err := options.generationContext().Err(); err != nil {
		return nil, err
	}
	if err := options.Validate(); err != nil {
		return nil, err
	}
	release, err := r.acquireInference(options.generationContext())
	if err != nil {
		return nil, err
	}
	defer release()
	data := &preparedChatData{runner: r}
	// This fence must precede ALL preflight rendering, including a custom
	// preparer's probes. Clear/Disable during preflight must prevent the later
	// generation from repopulating a snapshot under a newly captured epoch.
	if r.kind == loadedStandard {
		data.mistralKVPrefixEpoch, data.mistralKVPrefixEnabled = r.mistralKVPrefixCacheEpoch()
	}
	var selected []ChatMessage
	if prepare == nil {
		selected, data.info, err = r.prepareChatContext(messages, options, &data.tokens)
	} else {
		selected, data.info, err = prepare(messages, options)
	}
	if err != nil {
		return nil, err
	}
	if err := options.generationContext().Err(); err != nil {
		return nil, err
	}
	if options.FIMSuffix != "" {
		return nil, nil
	}
	for _, message := range messages {
		if len(message.Images) != 0 {
			return nil, nil
		}
	}
	for _, message := range selected {
		if len(message.Images) != 0 {
			return nil, nil
		}
	}
	if data.tokens == nil {
		data.tokens = r.renderMessages(selected, options.SystemPrompt, options.ActiveTools())
	}
	if err := options.generationContext().Err(); err != nil {
		return nil, err
	}
	if len(data.tokens) == 0 {
		return nil, fmt.Errorf("prompt rendered to zero tokens")
	}
	if normalizedContextWindowMode(options.ContextWindowMode) != ContextWindowFull && len(data.tokens) > data.info.PromptBudget {
		return nil, fmt.Errorf("prepared prompt needs %d tokens, but context can use at most %d", len(data.tokens), data.info.PromptBudget)
	}
	data.info.PromptTokens = len(data.tokens)
	data.info = clonePreparedChatInfo(data.info)
	data.input, data.selected = clonePreparedChatMessages(messages), clonePreparedChatMessages(selected)
	data.options = clonePreparedChatOptions(options)
	return &PreparedChat{runner: r, data: data}, nil
}

// Wrap returns a generator that consumes this preflight at most once. An exact
// first call runs directly on the preparing Runner; all other calls use
// fallback (which can be a conversation-recall wrapper for later tool rounds).
// A nil fallback uses the preparing Runner's ordinary generator.
func (p *PreparedChat) Wrap(fallback ChatGenerator) ChatGenerator {
	if p == nil {
		return fallback
	}
	if fallback == nil && p.runner != nil {
		fallback = p.runner.GenerateChatStreamUntil
	}
	return func(messages []ChatMessage, options GenerationOptions, onToken func(string) bool) (GenerationResult, error) {
		p.mu.Lock()
		data := p.data
		p.data = nil
		p.mu.Unlock()
		if data != nil && reflect.DeepEqual(messages, data.input) && samePreparedChatOptions(options, data.options) {
			return data.runner.generateChatStreamUntil(data.selected, data.options, onToken, data)
		}
		if fallback != nil {
			return fallback(messages, options, onToken)
		}
		return GenerationResult{}, fmt.Errorf("prepared chat already consumed and no fallback generator supplied")
	}
}

func samePreparedChatOptions(a, b GenerationOptions) bool {
	// Context identity binds the plan to its request. DeepEqual can equate
	// separate contexts and inspect mutable cancellation state recursively.
	contextType := reflect.TypeOf(a.ctx)
	if contextType != reflect.TypeOf(b.ctx) || (contextType != nil && (!contextType.Comparable() || a.ctx != b.ctx)) {
		return false
	}
	a.ctx, b.ctx = nil, nil
	return reflect.DeepEqual(a, b)
}

func clonePreparedChatMessages(messages []ChatMessage) []ChatMessage {
	if messages == nil {
		return nil
	}
	out := append([]ChatMessage{}, messages...)
	for i := range out {
		if out[i].ToolCalls != nil {
			out[i].ToolCalls = append([]ToolCall{}, out[i].ToolCalls...)
		}
	}
	return out
}

func clonePreparedChatOptions(options GenerationOptions) GenerationOptions {
	if options.StopSequences != nil {
		options.StopSequences = append([]string{}, options.StopSequences...)
	}
	if options.Tools != nil {
		options.Tools = append([]ToolDefinition{}, options.Tools...)
		for i := range options.Tools {
			if parameters := options.Tools[i].Function.Parameters; parameters != nil {
				options.Tools[i].Function.Parameters = append([]byte{}, parameters...)
			}
		}
	}
	return options
}

func clonePreparedChatInfo(info ContextWindowInfo) ContextWindowInfo {
	if info.Recall != nil {
		recall := *info.Recall
		recall.Sources = append([]ContextRecallSource(nil), recall.Sources...)
		info.Recall = &recall
	}
	return info
}
