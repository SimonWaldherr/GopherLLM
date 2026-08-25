package gopherllm

import (
	"encoding/json"
	"fmt"
	"strings"
)

const kimiDefaultSystemPrompt = "You are Kimi, an AI assistant created by Moonshot AI."

// renderKimiMessages mirrors Moonshot's Kimi-K2-Instruct chat_template.jinja:
//
//	<|im_system|>system<|im_middle|>{system}<|im_end|>
//	<|im_user|>user<|im_middle|>{user}<|im_end|>
//	<|im_assistant|>assistant<|im_middle|>
//
// Kimi's role markers are distinct from ChatML's <|im_start|> convention.
// Tool declarations, tool-call history, and tool results are rendered natively
// rather than through the generic <tool_call> fallback, since K2 was trained
// on its own control-token protocol.
func (r *Runner) renderKimiMessages(messages []ChatMessage, systemPrompt string, tools []ToolDefinition) ([]uint32, bool) {
	systemTok, ok1 := r.tok.SpecialID("<|im_system|>")
	userTok, ok2 := r.tok.SpecialID("<|im_user|>")
	assistantTok, ok3 := r.tok.SpecialID("<|im_assistant|>")
	middleTok, ok4 := r.tok.SpecialID("<|im_middle|>")
	endTok, ok5 := r.tok.SpecialID("<|im_end|>")
	if !(ok1 && ok2 && ok3 && ok4 && ok5) {
		return nil, false
	}

	tokens := make([]uint32, 0, 32)
	appendTurn := func(roleTok uint32, roleName, content string) {
		tokens = append(tokens, roleTok)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(roleName)...)
		tokens = append(tokens, middleTok)
		tokens = append(tokens, r.tok.EncodeWithoutBOS(content)...)
		tokens = append(tokens, endTok)
	}

	if len(tools) > 0 {
		payload, err := marshalKimiToolDefinitions(tools)
		if err != nil {
			return nil, false
		}
		appendTurn(systemTok, "tool_declare", string(payload))
	}

	hasSystem := false
	for _, message := range messages {
		if message.Role == ChatRoleSystem {
			hasSystem = true
			break
		}
	}
	if !hasSystem {
		system := strings.TrimSpace(systemPrompt)
		if system == "" {
			system = kimiDefaultSystemPrompt
		}
		appendTurn(systemTok, "system", system)
	}

	for _, message := range messages {
		roleName := message.Name
		if roleName == "" {
			switch message.Role {
			case ChatRoleSystem:
				roleName = "system"
			case ChatRoleAssistant:
				roleName = "assistant"
			case ChatRoleTool:
				roleName = "tool"
			default:
				roleName = "user"
			}
		}

		switch message.Role {
		case ChatRoleAssistant:
			tokens = append(tokens, assistantTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(roleName)...)
			tokens = append(tokens, middleTok)
			tokens = append(tokens, r.tok.EncodeWithoutBOS(message.Content)...)
			if len(message.ToolCalls) > 0 {
				callSectionStart, ok1 := r.tok.SpecialID("<|tool_calls_section_begin|>")
				callStart, ok2 := r.tok.SpecialID("<|tool_call_begin|>")
				argumentStart, ok3 := r.tok.SpecialID("<|tool_call_argument_begin|>")
				callEnd, ok4 := r.tok.SpecialID("<|tool_call_end|>")
				callSectionEnd, ok5 := r.tok.SpecialID("<|tool_calls_section_end|>")
				if !(ok1 && ok2 && ok3 && ok4 && ok5) {
					return nil, false
				}
				tokens = append(tokens, callSectionStart)
				for index, call := range message.ToolCalls {
					callID := call.ID
					if _, _, ok := parseKimiToolCallID(callID); !ok {
						callID = kimiToolCallID(call.Function.Name, index)
					}
					arguments := call.Function.Arguments
					if strings.TrimSpace(arguments) == "" {
						arguments = "{}"
					}
					tokens = append(tokens, callStart)
					tokens = append(tokens, r.tok.EncodeWithoutBOS(callID)...)
					tokens = append(tokens, argumentStart)
					tokens = append(tokens, r.tok.EncodeWithoutBOS(arguments)...)
					tokens = append(tokens, callEnd)
				}
				tokens = append(tokens, callSectionEnd)
			}
			tokens = append(tokens, endTok)
		case ChatRoleTool:
			appendTurn(systemTok, roleName, "## Return of "+message.ToolCallID+" "+message.Content)
		case ChatRoleSystem:
			appendTurn(systemTok, roleName, message.Content)
		default:
			appendTurn(userTok, roleName, message.Content)
		}
	}

	// add_generation_prompt=True from the official template.
	tokens = append(tokens, assistantTok)
	tokens = append(tokens, r.tok.EncodeWithoutBOS("assistant")...)
	tokens = append(tokens, middleTok)
	return tokens, true
}

// marshalKimiToolDefinitions reproduces Kimi's `deep_sort_dict` plus compact
// Jinja `tojson(separators=(',', ':'))` output. encoding/json sorts map keys,
// including nested decoded parameter schemas, so the prompt is deterministic
// and matches the official custom tokenizer's stable tool declaration form.
func marshalKimiToolDefinitions(tools []ToolDefinition) ([]byte, error) {
	payload := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		function := map[string]any{"name": tool.Function.Name}
		if tool.Function.Description != "" {
			function["description"] = tool.Function.Description
		}
		if len(tool.Function.Parameters) > 0 {
			var parameters any
			if err := json.Unmarshal(tool.Function.Parameters, &parameters); err != nil {
				return nil, fmt.Errorf("invalid parameters for Kimi tool %q: %w", tool.Function.Name, err)
			}
			function["parameters"] = parameters
		}
		payload = append(payload, map[string]any{
			"function": function,
			"type":     tool.Type,
		})
	}
	return json.Marshal(payload)
}
