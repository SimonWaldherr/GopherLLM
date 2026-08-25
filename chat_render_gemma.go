package gopherllm

import "strings"

// renderGemmaMessages renders the Gemma turn format:
//
//	<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{reply}<end_of_turn>\n...
//
// Gemma generations before 4 have no system role; the system prompt is folded
// into the first user turn (the convention Google's reference templates use).
// The assistant role is spelled "model".
func (r *Runner) renderGemmaMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	startTurn, ok1 := r.tok.SpecialID("<start_of_turn>")
	endTurn, ok2 := r.tok.SpecialID("<end_of_turn>")
	if !(ok1 && ok2) {
		return nil, false
	}
	system := strings.TrimSpace(systemPrompt)
	loop := make([]ChatMessage, 0, len(messages))
	for _, m := range messages {
		if m.Role == ChatRoleSystem {
			if s := strings.TrimSpace(m.Content); s != "" {
				system = s
			}
			continue
		}
		loop = append(loop, m)
	}
	tokens := []uint32{}
	if r.tok.AddBOS {
		tokens = append(tokens, r.tok.BOSID)
	}
	firstUser := true
	for _, m := range loop {
		role := "user"
		if m.Role == ChatRoleAssistant {
			role = "model"
		}
		content := strings.TrimSpace(m.Content)
		if m.Role == ChatRoleUser && firstUser && system != "" {
			content = system + "\n\n" + content
			firstUser = false
		} else if m.Role == ChatRoleUser {
			firstUser = false
		}
		tokens = append(tokens, startTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n"+content)...)
		tokens = append(tokens, endTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	tokens = append(tokens, startTurn)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("model\n")...)
	return tokens, true
}

// renderGemma4Messages implements the text-only part of Gemma 4's native
// template.  Tool-call serialisation is intentionally delegated to the
// generic formatter above; the important part for ordinary chat is preserving
// Gemma 4's turn and disabled-thinking channel markers:
//
//	<|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
func (r *Runner) renderGemma4Messages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	startTurn, ok1 := r.tok.SpecialID("<|turn>")
	endTurn, ok2 := r.tok.SpecialID("<turn|>")
	channelStart, ok3 := r.tok.SpecialID("<|channel>")
	channelEnd, ok4 := r.tok.SpecialID("<channel|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := make([]uint32, 0, 64)
	if r.tok.AddBOS {
		tokens = append(tokens, r.tok.BOSID)
	}
	appendTurn := func(role, content string) {
		tokens = append(tokens, startTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n")...)
		if content = strings.TrimSpace(content); content != "" {
			tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
		}
		tokens = append(tokens, endTurn)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	hasSystem := false
	for _, message := range messages {
		hasSystem = hasSystem || message.Role == ChatRoleSystem
	}
	if system := strings.TrimSpace(systemPrompt); system != "" && !hasSystem {
		appendTurn("system", system)
	}
	for _, message := range messages {
		role := "user"
		switch message.Role {
		case ChatRoleSystem:
			role = "system"
		case ChatRoleAssistant:
			role = "model"
		}
		appendTurn(role, message.Content)
	}
	tokens = append(tokens, startTurn)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("model\n")...)
	// Some Gemma 4 checkpoints (e.g. 12B/26B) open and immediately close the
	// thought channel here when reasoning is disabled, so generation lands on
	// a fresh channel rather than the old Gemma <start_of_turn> protocol.
	// Others (E2B) never emit this in their own add_generation_prompt branch
	// at all; injecting it there is out-of-distribution and derails
	// generation into gibberish. Checked per-checkpoint against its own
	// chat_template rather than assumed for every gemma4-chat model.
	if r.gemma4ClosesThoughtChannelAtGenerate() {
		tokens = append(tokens, channelStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("thought\n")...)
		tokens = append(tokens, channelEnd)
	}
	return tokens, true
}

// gemma4ClosesThoughtChannelAtGenerate reports whether this checkpoint's own
// chat_template appends '<|channel>thought\n<channel|>' right after
// '<|turn>model\n' in its add_generation_prompt branch. That literal only
// appears there for models whose default-thinking behaviour needs an
// explicit closed channel; the same substring also shows up (via string
// concatenation, not this literal) when serialising a past assistant turn's
// reasoning back into history, so a plain substring check on that other
// occurrence would be a false positive - hence the exact literal match.
func (r *Runner) gemma4ClosesThoughtChannelAtGenerate() bool {
	if r.gguf == nil {
		return false
	}
	v, ok := r.gguf.Metadata["tokenizer.chat_template"]
	if !ok {
		return false
	}
	s, ok := v.AsString()
	if !ok {
		return false
	}
	return strings.Contains(s, `<|channel>thought\n<channel|>`)
}
