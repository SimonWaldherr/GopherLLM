package gopherllm

import (
	"encoding/json"
	"strings"
)

func (r *Runner) renderHeaderChatMessages(messages []ChatMessage, systemPrompt string) ([]uint32, bool) {
	bot, ok1 := r.tok.SpecialID("<|begin_of_text|>")
	startHeader, ok2 := r.tok.SpecialID("<|start_header_id|>")
	endHeader, ok3 := r.tok.SpecialID("<|end_header_id|>")
	eot, ok4 := r.tok.SpecialID("<|eot_id|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}
	tokens := []uint32{bot}
	pushHeader := func(role string) {
		tokens = append(tokens, startHeader)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(role)...)
		tokens = append(tokens, endHeader)
		tokens = append(tokens, r.tok.EncodeWithoutBOS("\n\n")...)
	}
	hasSystem := false
	for _, m := range messages {
		hasSystem = hasSystem || m.Role == ChatRoleSystem
	}
	if strings.TrimSpace(systemPrompt) != "" && !hasSystem {
		pushHeader("system")
		tokens = append(tokens, r.tok.EncodeWithoutBOS(systemPrompt)...)
		tokens = append(tokens, eot)
	}
	for _, m := range messages {
		role := "user"
		if m.Role == ChatRoleSystem {
			role = "system"
		} else if m.Role == ChatRoleAssistant {
			role = "assistant"
		}
		pushHeader(role)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(m.Content)...)
		tokens = append(tokens, eot)
	}
	pushHeader("assistant")
	return tokens, true
}

const (
	llama31KnowledgeCutoff = "December 2023"
	llama31DefaultDate     = "26 Jul 2024"
)

// renderLlama31Messages mirrors Meta's Llama-3.1-Instruct chat template for
// custom function tools. In particular, the template always starts with its
// dated system envelope, puts custom tool definitions into the first user
// turn, serializes an assistant call as {"name", "parameters"}, and feeds a
// tool result back using the ipython header. Built-in tools use Llama's
// separate <|python_tag|> protocol and intentionally remain outside the
// OpenAI-compatible ToolDefinition API.
func (r *Runner) renderLlama31Messages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]uint32, bool) {
	bot, ok1 := r.tok.SpecialID("<|begin_of_text|>")
	startHeader, ok2 := r.tok.SpecialID("<|start_header_id|>")
	endHeader, ok3 := r.tok.SpecialID("<|end_header_id|>")
	eot, ok4 := r.tok.SpecialID("<|eot_id|>")
	if !(ok1 && ok2 && ok3 && ok4) {
		return nil, false
	}

	loopMessages := messages
	system := strings.TrimSpace(systemPrompt)
	// The native template consumes a leading system message into the mandatory
	// initial system envelope. This matches the ordinary Chat API convention
	// while leaving any later system message as a normal, replayable turn.
	if len(loopMessages) > 0 && loopMessages[0].Role == ChatRoleSystem {
		system = strings.TrimSpace(loopMessages[0].Content)
		loopMessages = loopMessages[1:]
	}

	// Meta's template with tools_in_user_message=true requires the first
	// remaining message to be a user message. Returning false here gives the
	// caller's generic fallback a chance to preserve an unusual history.
	toolUser := -1
	if len(tools) > 0 {
		if len(loopMessages) == 0 || loopMessages[0].Role != ChatRoleUser {
			return nil, false
		}
		toolUser = 0
	}

	tokens := make([]uint32, 0, 64)
	appendText := func(text string) {
		tokens = append(tokens, r.tok.EncodeWithoutBOS(text)...)
	}
	pushHeader := func(role string) {
		tokens = append(tokens, startHeader)
		appendText(role)
		tokens = append(tokens, endHeader)
		appendText("\n\n")
	}
	appendTurn := func(role, content string) {
		pushHeader(role)
		appendText(strings.TrimSpace(content))
		tokens = append(tokens, eot)
	}

	tokens = append(tokens, bot)
	pushHeader("system")
	if len(tools) > 0 {
		// This is emitted for custom tools too; it tells the model that the
		// result payload will be provided as an ipython turn.
		appendText("Environment: ipython\n")
	}
	appendText("Cutting Knowledge Date: " + llama31KnowledgeCutoff + "\n")
	appendText("Today Date: " + r.llama31TemplateDate() + "\n\n")
	appendText(system)
	tokens = append(tokens, eot)

	for i, message := range loopMessages {
		if i == toolUser {
			pushHeader("user")
			// Keep the otherwise slightly surprising lack of a newline between
			// the format sentence and "Do not use variables." This is how the
			// local Meta-Llama-3.1 GGUF's Jinja whitespace controls render it.
			appendText("Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n")
			appendText("Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n")
			for _, tool := range tools {
				payload, err := json.MarshalIndent(tool, "", "    ")
				if err != nil {
					return nil, false
				}
				appendText(string(payload))
				appendText("\n\n")
			}
			appendText(strings.TrimSpace(message.Content))
			tokens = append(tokens, eot)
			continue
		}

		switch message.Role {
		case ChatRoleTool:
			// Tool output is intentionally not trimmed. The native template
			// treats it as an ipython value rather than prose, so whitespace can
			// carry semantic meaning for a caller.
			pushHeader("ipython")
			appendText(message.Content)
			tokens = append(tokens, eot)
		case ChatRoleAssistant:
			if len(message.ToolCalls) == 0 {
				appendTurn("assistant", message.Content)
				continue
			}
			// Llama 3.1's custom-function branch explicitly supports one tool
			// call per assistant turn. Do not invent a multi-call encoding: use
			// the generic lossless fallback instead for such historical turns.
			if len(message.ToolCalls) != 1 {
				return nil, false
			}
			call := message.ToolCalls[0]
			args := strings.TrimSpace(call.Function.Arguments)
			if call.Function.Name == "" || len(args) == 0 || args[0] != '{' || !json.Valid([]byte(args)) {
				return nil, false
			}
			// Keep the literal separators from Meta's template rather than
			// using a generic compact wrapper: {"name": "…", "parameters": …}.
			// The arguments themselves are already JSON as required by the
			// OpenAI-compatible ToolCall surface.
			name, err := json.Marshal(call.Function.Name)
			if err != nil {
				return nil, false
			}
			pushHeader("assistant")
			appendText(`{"name": ` + string(name) + `, "parameters": ` + args + `}`)
			tokens = append(tokens, eot)
		case ChatRoleSystem:
			appendTurn("system", message.Content)
		default:
			appendTurn("user", message.Content)
		}
	}
	pushHeader("assistant")
	return tokens, true
}

// llama31TemplateDate reads the Jinja date_string assignment when present so
// closely related Llama 3.1 GGUF exports keep their own declared date. The
// inspected Meta-Llama-3.1-8B-Instruct template falls back to 26 Jul 2024.
func (r *Runner) llama31TemplateDate() string {
	if r == nil || r.gguf == nil {
		return llama31DefaultDate
	}
	v, ok := r.gguf.Metadata["tokenizer.chat_template"]
	if !ok {
		return llama31DefaultDate
	}
	template, ok := v.AsString()
	if !ok {
		return llama31DefaultDate
	}
	const assignment = "date_string"
	for start := 0; start < len(template); {
		i := strings.Index(template[start:], assignment)
		if i < 0 {
			break
		}
		i += start
		rest := strings.TrimSpace(template[i+len(assignment):])
		start = i + len(assignment)
		if !strings.HasPrefix(rest, "=") {
			continue
		}
		rest = strings.TrimSpace(rest[1:])
		if len(rest) < 2 || (rest[0] != '"' && rest[0] != 39) {
			continue
		}
		quote := rest[0]
		if end := strings.IndexByte(rest[1:], quote); end >= 0 {
			if date := rest[1 : end+1]; date != "" {
				return date
			}
		}
	}
	return llama31DefaultDate
}
