package gopherllm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// renderMessages renders the conversation (and, if any, the tool listing) into
// tokens using the active chat template. Mistral, Llama 3.1, and Kimi get
// their native tool conventions; every other template (and gpt-oss, for
// which tool calling is not yet implemented) uses the generic <tool_call>
// JSON convention, applied uniformly by flattening tool listings and
// tool-call history into ordinary system/user/assistant text before
// delegating to the per-family renderer below.
func (r *Runner) renderMessages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) []uint32 {
	if r.arch == "gpt-oss" {
		return r.renderGptOssMessages(messages, systemPrompt)
	}
	if qwen35Family(r.arch) {
		if tokens, ok := r.renderQwen35Messages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	if r.chatTemplateKind() == "kimi-chat" {
		if tokens, ok := r.renderKimiMessages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	if r.arch == "nemotron_h_moe" {
		generic, genericSystem := injectGenericTools(messages, systemPrompt, tools)
		if tokens, ok := r.renderSoofiIsarMessages(generic, genericSystem); ok {
			return tokens
		}
	}
	if r.chatTemplateKind() == "mistral-inst" {
		if tokens, _, ok, _ := r.renderMistralInstMessages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	if r.chatTemplateKind() == "llama31-chat" {
		if tokens, ok := r.renderLlama31Messages(messages, systemPrompt, tools); ok {
			return tokens
		}
	}
	generic, genericSystem := injectGenericTools(messages, systemPrompt, tools)
	switch r.chatTemplateKind() {
	case "exaone4-chat":
		if tokens, ok := r.renderExaone4Messages(generic, genericSystem); ok {
			return tokens
		}
	case "gemma4-chat":
		if tokens, ok := r.renderGemma4Messages(generic, genericSystem); ok {
			return tokens
		}
	case "gemma-chat":
		if tokens, ok := r.renderGemmaMessages(generic, genericSystem); ok {
			return tokens
		}
	case "header-chat":
		if tokens, ok := r.renderHeaderChatMessages(generic, genericSystem); ok {
			return tokens
		}
	case "llama31-chat":
		// A historical turn which the native Llama 3.1 template cannot replay
		// (for example, multiple calls in one assistant message) still needs a
		// lossless fallback rather than silently dropping tool state.
		if tokens, ok := r.renderHeaderChatMessages(generic, genericSystem); ok {
			return tokens
		}
	case "chatml":
		if tokens, ok := r.renderChatMLMessages(generic, genericSystem); ok {
			return tokens
		}
	case "alpaca-chat":
		if tokens, ok := r.renderAlpacaMessages(generic, genericSystem); ok {
			return tokens
		}
	case "phi4-chat":
		if tokens, ok := r.renderPhi4Messages(generic, genericSystem); ok {
			return tokens
		}
	case "phi-chat":
		if tokens, ok := r.renderPhiMessages(generic, genericSystem); ok {
			return tokens
		}
	case "deepseek-r1-qwen":
		if tokens, ok := r.renderDeepSeekR1QwenMessages(generic, genericSystem); ok {
			return tokens
		}
	case "granite-chat":
		if tokens, ok := r.renderGraniteMessages(generic, genericSystem); ok {
			return tokens
		}
	}
	return r.renderPlainMessages(generic, genericSystem)
}

// injectGenericTools flattens tool listings and tool-call/tool-result history
// into ordinary text so any chat-template renderer that only understands
// system/user/assistant turns can carry tool use anyway. A tool listing is
// merged into an existing explicit system message's content when present,
// otherwise appended to systemPrompt so the caller's default system prompt
// (e.g. "You are a helpful assistant.") is preserved rather than replaced.
// When there is no tool activity at all, messages/systemPrompt are returned
// unchanged (no allocation) so the common no-tools path is a no-op.
func injectGenericTools(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]ChatMessage, string) {
	hasActivity := len(tools) > 0
	if !hasActivity {
		for _, m := range messages {
			if m.Role == ChatRoleTool || (m.Role == ChatRoleAssistant && len(m.ToolCalls) > 0) {
				hasActivity = true
				break
			}
		}
	}
	if !hasActivity {
		return messages, systemPrompt
	}

	hasExplicitSystem := len(messages) > 0 && messages[0].Role == ChatRoleSystem
	out := make([]ChatMessage, len(messages))
	for i, m := range messages {
		switch {
		case i == 0 && hasExplicitSystem && len(tools) > 0:
			m.Content = appendSection(m.Content, genericToolListText(tools))
		case m.Role == ChatRoleAssistant && len(m.ToolCalls) > 0:
			m.Content = renderGenericAssistantToolCalls(m.Content, m.ToolCalls)
		case m.Role == ChatRoleTool:
			m.Role = ChatRoleUser
			m.Content = renderGenericToolResult(m.Name, m.Content)
		}
		out[i] = m
	}
	if !hasExplicitSystem && len(tools) > 0 {
		systemPrompt = appendSection(systemPrompt, genericToolListText(tools))
	}
	return out, systemPrompt
}

func appendSection(base, section string) string {
	base = strings.TrimRight(base, "\n")
	if base == "" {
		return section
	}
	return base + "\n\n" + section
}

// genericToolListText renders an OpenAI-shaped tool list into the Hermes/Qwen
// style calling convention: a <tool_call> JSON block per invocation.
func genericToolListText(tools []ToolDefinition) string {
	b, err := json.Marshal(tools)
	if err != nil {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("You have access to the following tools. To call one, respond with only a block of exactly this form (multiple blocks if you need multiple calls in the same turn):\n")
	sb.WriteString("<tool_call>\n{\"name\": \"<tool name>\", \"arguments\": <arguments object>}\n</tool_call>\n\n")
	sb.WriteString("Available tools:\n")
	sb.Write(b)
	return sb.String()
}

func renderGenericAssistantToolCalls(content string, calls []ToolCall) string {
	var sb strings.Builder
	if trimmed := strings.TrimSpace(content); trimmed != "" {
		sb.WriteString(trimmed)
		sb.WriteString("\n")
	}
	for i, call := range calls {
		if i > 0 {
			sb.WriteString("\n")
		}
		args := call.Function.Arguments
		if strings.TrimSpace(args) == "" {
			args = "{}"
		}
		nameJSON, _ := json.Marshal(call.Function.Name)
		fmt.Fprintf(&sb, "<tool_call>\n{\"name\": %s, \"arguments\": %s}\n</tool_call>", nameJSON, args)
	}
	return sb.String()
}

func renderGenericToolResult(name, content string) string {
	if name != "" {
		return fmt.Sprintf("<tool_response name=%q>\n%s\n</tool_response>", name, content)
	}
	return fmt.Sprintf("<tool_response>\n%s\n</tool_response>", content)
}

func (r *Runner) renderPlainMessages(messages []ChatMessage, systemPrompt string) []uint32 {
	var b strings.Builder
	if strings.TrimSpace(systemPrompt) != "" {
		b.WriteString("System: ")
		b.WriteString(strings.TrimSpace(systemPrompt))
		b.WriteString("\n\n")
	}
	for _, m := range messages {
		switch m.Role {
		case ChatRoleSystem:
			b.WriteString("System: ")
		case ChatRoleAssistant:
			b.WriteString("Assistant: ")
		default:
			b.WriteString("User: ")
		}
		b.WriteString(strings.TrimSpace(m.Content))
		b.WriteString("\n\n")
	}
	b.WriteString("Assistant:")
	return r.tok.Encode(b.String())
}
