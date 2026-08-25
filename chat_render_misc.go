package gopherllm

import "strings"

func (r *Runner) renderGptOssMessages(messages []ChatMessage, systemPrompt string) []uint32 {
	start := specialOr(r.tok, "<|start|>", 200006)
	channel := specialOr(r.tok, "<|channel|>", 200005)
	message := specialOr(r.tok, "<|message|>", 200008)
	end := specialOr(r.tok, "<|end|>", 200007)
	user := specialOrEncoded(r.tok, "user")
	assistant := specialOrEncoded(r.tok, "assistant")
	system := specialOrEncoded(r.tok, "system")
	finalTok := specialOrEncoded(r.tok, "final")
	tokens := []uint32{}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		tokens = append(tokens, start, system, message)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(systemPrompt)...)
		tokens = append(tokens, end)
	}
	for _, m := range messages {
		role := user
		if m.Role == ChatRoleSystem {
			role = system
		} else if m.Role == ChatRoleAssistant {
			role = assistant
		}
		tokens = append(tokens, start, role, message)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
		tokens = append(tokens, end)
	}
	tokens = append(tokens, start, assistant, channel, finalTok, message)
	return tokens
}

// renderAlpacaMessages reproduces Upstage SOLAR's instruct template, the one
// SauerkrautLM-SOLAR-Instruct and the other SOLAR derivatives ship:
//
//	### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n{assistant}
//
// with a trailing "### Assistant:\n" as the generation prompt. Note the
// asymmetry, which is in the original template and is deliberately preserved
// here: system and user turns are followed by a blank line, an assistant turn
// is not — the next "### User:" block supplies its own separation.
//
// The format carries no special tokens, so unlike the ChatML/Gemma renderers
// there is nothing to look up and nothing that can be missing; it only needs
// the BOS the tokenizer already prepends. It therefore always succeeds, but
// keeps the (tokens, ok) shape its siblings use so the dispatch table in
// renderMessages stays uniform.
func (r *Runner) renderAlpacaMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	return r.tok.Encode(alpacaPrompt(messages, systemPrompt)), true
}

// alpacaPrompt is split out from the renderer so the exact template text can
// be asserted directly, without a tokenizer round-trip in the way: the whole
// point of this format is the literal "### Role:" spelling and its spacing.
func alpacaPrompt(messages []ChatMessage, systemPrompt string) string {
	var b strings.Builder
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if s := strings.TrimSpace(systemPrompt); s != "" && !hasSystem {
		b.WriteString("### System:\n")
		b.WriteString(s)
		b.WriteString("\n\n")
	}
	for _, m := range messages {
		content := strings.TrimSpace(m.Content)
		switch m.Role {
		case ChatRoleSystem:
			// The template skips a system turn with empty content rather than
			// emitting a bare header.
			if content == "" {
				continue
			}
			b.WriteString("### System:\n")
			b.WriteString(content)
			b.WriteString("\n\n")
		case ChatRoleAssistant:
			b.WriteString("### Assistant:\n")
			b.WriteString(content)
		default:
			b.WriteString("### User:\n")
			b.WriteString(content)
			b.WriteString("\n\n")
		}
	}
	b.WriteString("### Assistant:\n")
	return b.String()
}

func (r *Runner) renderChatMLMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imEnd, ok2 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, imStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n"+strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, imEnd)
			tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
		}
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt, true)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		appendTurn(role, m.Content, true)
	}
	tokens = append(tokens, imStart)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant\n")...)
	// Some ChatML checkpoints (e.g. Nemotron-H) default enable_thinking to
	// true when a caller does not pass it, and their own add_generation_prompt
	// branch then opens an explicit, unclosed <think> tag rather than leaving
	// the assistant turn bare. Skipping that tag is out-of-distribution for
	// those checkpoints and derails generation; mirror it only for models
	// whose own chat_template actually does this by default.
	if think, ok := r.tok.SpecialID("<think>"); ok && r.chatMLOpensThinkTagByDefault() {
		tokens = append(tokens, think)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	return tokens, true
}

// chatMLOpensThinkTagByDefault reports whether this checkpoint's own
// chat_template emits '<|im_start|>assistant\n<think>\n' in its
// add_generation_prompt branch when enable_thinking is left at the
// template's own default (i.e. the literal appears unconditionally on one
// side of the enable_thinking check, not only when a caller opts in).
func (r *Runner) chatMLOpensThinkTagByDefault() bool {
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
	return strings.Contains(s, `'<|im_start|>assistant\n<think>\n'`) &&
		strings.Contains(s, `enable_thinking = enable_thinking if enable_thinking is defined else True`)
}

// renderSoofiIsarMessages mirrors the text-only portion of the embedded
// Soofi-S-Isar Jinja template. In particular, it provides the model identity
// prompt and deliberately opens the assistant's <think> section; generic
// ChatML would omit both and produces a markedly weaker, non-reasoning turn.
func (r *Runner) renderSoofiIsarMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imEnd, ok2 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2) {
		return nil, false
	}
	const defaultSystem = "You are Soofi (Sovereign Open Source Foundation Models), an open-source AI assistant built for reasoning, developed by a German research consortium.\n\nArchitecture: Hybrid Mixture-of-Experts (MoE) with 23 Mamba-2/MoE layers and 6 Attention layers. 128 experts + 1 shared expert per MoE layer, 6 activated per token. 3.5B active parameters, 30B total.\n\nTraining: Trained from scratch on 25 trillion freely available tokens. Primary languages: English and German. Limited capability in French, Italian, and Spanish. English is the pivot language.\n\nBehaviour:\n- Answer identity questions naturally as Soofi.\n- For non-identity questions, respond normally and helpfully.\n- Match the language the user writes in.\n- Knowledge cutoff: 2025-12"
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, imStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role+"\n"+strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, imEnd)
			tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
		}
	}
	appendTurn("system", strings.TrimSpace(defaultSystem+"\n\n"+systemPrompt), true)
	for _, message := range messages {
		role := "user"
		if message.Role == ChatRoleAssistant {
			role = "assistant"
			// The Isar template keeps a closed empty thought section in history
			// when callers supplied only visible assistant content.
			if !strings.Contains(message.Content, "<think>") && !strings.Contains(message.Content, "</think>") {
				message.Content = "<think></think>" + message.Content
			}
		} else if message.Role == ChatRoleSystem {
			role = "system"
		}
		appendTurn(role, message.Content, true)
	}
	tokens = append(tokens, imStart)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant\n<think>\n")...)
	return tokens, true
}

// renderPhi4Messages mirrors Phi-4's embedded template:
//
//	<|im_start|>role<|im_sep|>content<|im_end|>
//
// It is deliberately not ChatML: ChatML places a newline after the role,
// whereas Phi-4 uses its dedicated <|im_sep|> control token. The final
// assistant turn remains open after that separator for generation.
func (r *Runner) renderPhi4Messages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imSep, ok2 := r.tok.SpecialID("<|im_sep|>")
	imEnd, ok3 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string) {
		tokens = append(tokens, imStart)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, imSep)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
		tokens = append(tokens, imEnd)
	}
	hasSystem := false
	for _, message := range messages {
		hasSystem = hasSystem || message.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt)
	}
	for _, message := range messages {
		role := "user"
		switch message.Role {
		case ChatRoleSystem:
			role = "system"
		case ChatRoleAssistant:
			role = "assistant"
		}
		appendTurn(role, message.Content)
	}
	tokens = append(tokens, imStart)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant")...)
	tokens = append(tokens, imSep)
	return tokens, true
}

func (r *Runner) renderPhiMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	systemTok, ok1 := r.tok.SpecialID("<|system|>")
	userTok, ok2 := r.tok.SpecialID("<|user|>")
	assistantTok, ok3 := r.tok.SpecialID("<|assistant|>")
	endTok, ok4 := r.tok.SpecialID("<|end|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role ChatRole, content string) {
		switch role {
		case ChatRoleSystem:
			tokens = append(tokens, systemTok)
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
		default:
			tokens = append(tokens, userTok)
		}
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n"+strings.TrimSpace(content))...)
		tokens = append(tokens, endTok)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn(ChatRoleSystem, systemPrompt)
	}
	for _, m := range messages {
		appendTurn(m.Role, m.Content)
	}
	tokens = append(tokens, assistantTok)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	return tokens, true
}

func (r *Runner) renderDeepSeekR1QwenMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	userTok, ok1 := r.tok.SpecialID("<｜User｜>")
	assistantTok, ok2 := r.tok.SpecialID("<｜Assistant｜>")
	endTok, ok3 := r.tok.SpecialID("<｜end▁of▁sentence｜>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{r.tok.BOSID}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(systemPrompt))...)
	}
	for _, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
			tokens = append(tokens, endTok)
		default:
			tokens = append(tokens, userTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
		}
	}
	tokens = append(tokens, assistantTok)
	return tokens, true
}

func (r *Runner) renderGraniteMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	startRole, ok1 := r.tok.SpecialID("<|start_of_role|>")
	endRole, ok2 := r.tok.SpecialID("<|end_of_role|>")
	endText, ok3 := r.tok.SpecialID("<|end_of_text|>")
	if !(ok1 && ok2 && ok3) {
		return nil, false
	}
	tokens := []uint32{}
	appendTurn := func(role, content string, close bool) {
		tokens = append(tokens, startRole)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, endRole)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(content))...)
		if close {
			tokens = append(tokens, endText)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(" ")...)
		}
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendTurn("system", systemPrompt, true)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		appendTurn(role, m.Content, true)
	}
	appendTurn("assistant", "", false)
	return tokens, true
}

// renderExaone4Messages mirrors the text-only EXAONE 4 instruct protocol.
// Unlike most turn formats, user content ends with a newline but no
// [|endofturn|]; system, assistant, and tool-derived turns are closed.
func (r *Runner) renderExaone4Messages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	system, ok1 := r.tok.SpecialID("[|system|]")
	user, ok2 := r.tok.SpecialID("[|user|]")
	assistant, ok3 := r.tok.SpecialID("[|assistant|]")
	end, ok4 := r.tok.SpecialID("[|endofturn|]")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := []uint32{}
	appendClosed := func(marker uint32, content string) {
		tokens = append(tokens, marker)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(content))...)
		tokens = append(tokens, end)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		appendClosed(system, systemPrompt)
	}
	for _, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			appendClosed(system, m.Content)
		case ChatRoleAssistant:
			appendClosed(assistant, m.Content)
		default:
			tokens = append(tokens, user)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(strings.TrimSpace(m.Content))...)
			tokens = append(tokens, r.tok.EncodeWithoutBOS("\n")...)
		}
	}
	tokens = append(tokens, assistant)
	return tokens, true
}
