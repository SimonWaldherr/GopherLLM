package gopherllm

import (
	"encoding/json"
	"sort"
	"strings"
)

// renderQwen35Messages mirrors the text-only parts of Qwen3.5/3.6/3.8's embedded
// ChatML template. In addition to its thinking prompt it uses Qwen's native
// XML-like tool protocol; generic JSON tool blocks noticeably degrade tool
// calling because the model was trained on <function=...>/<parameter=...>.
// The optional tools argument keeps the simple two-argument form convenient
// for renderer tests and callers that do not expose tools.
func (r *Runner) renderQwen35Messages(messages []ChatMessage, systemPrompt string, toolSets ...[]ToolDefinition) ([]uint32, bool) {
	imStart, ok1 := r.tok.SpecialID("<|im_start|>")
	imEnd, ok2 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2) {
		return nil, false
	}
	var tools []ToolDefinition
	if len(toolSets) > 0 {
		tools = toolSets[0]
	}
	tokens := make([]uint32, 0)
	appendText := func(text string) {
		tokens = append(tokens, r.tok.EncodeWithoutBOS(text)...)
	}
	appendTurn := func(role, content string) {
		tokens = append(tokens, imStart)
		appendText(role + "\n" + strings.TrimSpace(content))
		tokens = append(tokens, imEnd)
		appendText("\n")
	}

	hasSystem := false
	var systemParts []string
	for _, message := range messages {
		if message.Role == ChatRoleSystem {
			hasSystem = true
			if content := strings.TrimSpace(message.Content); content != "" {
				systemParts = append(systemParts, content)
			}
		}
	}
	// The embedded Qwen template permits a system turn only at the beginning.
	// Normalize multiple API-level system messages into that one turn rather
	// than silently dropping a later instruction.
	explicitSystem := strings.Join(systemParts, "\n\n")
	if len(tools) > 0 {
		content := qwen35ToolSystemPrompt(tools)
		if hasSystem {
			content = appendSection(content, explicitSystem)
		} else if strings.TrimSpace(systemPrompt) != "" {
			content = appendSection(content, systemPrompt)
		}
		appendTurn("system", content)
	} else if hasSystem {
		appendTurn("system", explicitSystem)
	} else if strings.TrimSpace(systemPrompt) != "" {
		appendTurn("system", systemPrompt)
	}

	for i, message := range messages {
		if message.Role == ChatRoleSystem {
			// The system turn was rendered above, including the native tool list.
			continue
		}
		switch message.Role {
		case ChatRoleAssistant:
			content := strings.TrimSpace(message.Content)
			if len(message.ToolCalls) > 0 {
				content = renderQwen35AssistantToolCalls(content, message.ToolCalls)
			}
			appendTurn("assistant", content)
		case ChatRoleTool:
			// Qwen groups consecutive tool results into one user turn.
			if i == 0 || messages[i-1].Role != ChatRoleTool {
				tokens = append(tokens, imStart)
				appendText("user")
			}
			appendText("\n<tool_response>\n")
			appendText(strings.TrimSpace(message.Content))
			appendText("\n</tool_response>")
			if i == len(messages)-1 || messages[i+1].Role != ChatRoleTool {
				tokens = append(tokens, imEnd)
				appendText("\n")
			}
		default:
			appendTurn("user", message.Content)
		}
	}
	// add_generation_prompt=True with thinking enabled (the Qwen default).
	tokens = append(tokens, imStart)
	appendText("assistant\n<think>\n")
	return tokens, true
}

func qwen35ToolSystemPrompt(tools []ToolDefinition) string {
	var sb strings.Builder
	sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
	for _, tool := range tools {
		if encoded, err := json.Marshal(tool); err == nil {
			sb.WriteByte('\n')
			sb.Write(encoded)
		}
	}
	sb.WriteString("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n")
	// Keep the native text portion byte-for-byte aligned with Qwen3.5/3.6/3.8's
	// embedded chat template. These models are sensitive to the exact native
	// XML example and its explicit instruction block.
	sb.WriteString("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")
	return sb.String()
}

func renderQwen35AssistantToolCalls(content string, calls []ToolCall) string {
	var sb strings.Builder
	if content != "" {
		sb.WriteString(content)
	}
	for i, call := range calls {
		if i == 0 && sb.Len() > 0 {
			sb.WriteString("\n\n")
		} else if i > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString("<tool_call>\n<function=")
		sb.WriteString(call.Function.Name)
		sb.WriteString(">\n")
		var args map[string]any
		if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil {
			args = map[string]any{}
		}
		keys := make([]string, 0, len(args))
		for key := range args {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			sb.WriteString("<parameter=")
			sb.WriteString(key)
			sb.WriteString(">\n")
			if value, ok := args[key].(string); ok {
				sb.WriteString(value)
			} else if encoded, err := json.Marshal(args[key]); err == nil {
				sb.Write(encoded)
			}
			sb.WriteString("\n</parameter>\n")
		}
		sb.WriteString("</function>\n</tool_call>")
	}
	return sb.String()
}
