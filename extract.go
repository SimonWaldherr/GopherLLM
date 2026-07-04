package main

import (
	"encoding/json"
	"strings"
)

// This file turns a model's raw decoded text into the structured pieces an
// agentic caller actually wants: visible answer text, chain-of-thought
// ("reasoning"), and any tool calls the model requested. All functions here
// are pure string transforms with no model/tokenizer dependency, so they are
// exercised directly with string fixtures in extract_test.go.

// classifyOutput splits a completed (or partial, if generation was canceled)
// raw model response into content/reasoning/tool-calls, using the convention
// appropriate for this Runner's architecture and active chat template. rng
// assigns ids to any tool call the model didn't provide one for (or provided
// an invalid one).
func (r *Runner) classifyOutput(raw string, tools []ToolDefinition, rng *Rng) (content, reasoning string, calls []ToolCall) {
	if r.arch == "gpt-oss" {
		content, reasoning, calls = extractGptOssChannels(raw)
	} else {
		content, reasoning = extractThink(raw)
		if len(tools) > 0 {
			if r.chatTemplateKind() == "mistral-inst" {
				content, calls = extractToolCallsMistral(content)
			} else {
				content, calls = extractToolCallsGeneric(content)
			}
		}
	}
	for i := range calls {
		if calls[i].Type == "" {
			calls[i].Type = "function"
		}
		if !validToolCallID(calls[i].ID) {
			calls[i].ID = newToolCallID(rng)
		}
	}
	return content, reasoning, calls
}

// extractThink pulls DeepSeek-R1/QwQ-style <think>...</think> chain-of-thought
// blocks out of text, returning the remaining visible text and the
// concatenated reasoning separately. An unterminated trailing <think> (e.g.
// generation was cut off by max_tokens mid-thought) is treated as reasoning
// through the end of the text, which is safer than leaking a half-formed
// thought into the visible answer.
func extractThink(text string) (content, reasoning string) {
	const openTag, closeTag = "<think>", "</think>"
	var contentBuf strings.Builder
	var reasoningParts []string
	rest := text
	for {
		i := strings.Index(rest, openTag)
		if i < 0 {
			contentBuf.WriteString(rest)
			break
		}
		contentBuf.WriteString(rest[:i])
		rest = rest[i+len(openTag):]
		j := strings.Index(rest, closeTag)
		if j < 0 {
			reasoningParts = append(reasoningParts, rest)
			break
		}
		reasoningParts = append(reasoningParts, rest[:j])
		rest = rest[j+len(closeTag):]
	}
	return strings.TrimSpace(contentBuf.String()), strings.TrimSpace(strings.Join(reasoningParts, "\n\n"))
}

// extractToolCallsMistral parses Mistral's native tool-calling convention, per
// its actual gguf chat_template (verified directly against the Ministral
// model, not just documentation): each call is rendered as
// "[TOOL_CALLS]{name}[ARGS]{argumentsJSON}", one such segment per call, with
// no array wrapper, no id, and no closing marker — the next "[TOOL_CALLS]" (or
// end of text) delimits the arguments.
func extractToolCallsMistral(text string) (content string, calls []ToolCall) {
	const callMarker, argsMarker = "[TOOL_CALLS]", "[ARGS]"
	i := strings.Index(text, callMarker)
	if i < 0 {
		return text, nil
	}
	content = text[:i]
	rest := text[i:]
	for strings.HasPrefix(rest, callMarker) {
		rest = rest[len(callMarker):]
		ai := strings.Index(rest, argsMarker)
		if ai < 0 {
			break // malformed: a call with no [ARGS] segment
		}
		name := strings.TrimSpace(rest[:ai])
		rest = rest[ai+len(argsMarker):]

		argsText, next := rest, ""
		if j := strings.Index(rest, callMarker); j >= 0 {
			argsText, next = rest[:j], rest[j:]
		}
		argsText = strings.TrimSpace(argsText)
		if argsText == "" {
			argsText = "{}"
		}
		if name != "" {
			calls = append(calls, ToolCall{Type: "function", Function: ToolCallFunction{Name: name, Arguments: argsText}})
		}
		rest = next
	}
	return strings.TrimSpace(content), calls
}

// extractToolCallsGeneric parses the Hermes/Qwen-style convention used for
// every non-Mistral chat template: one or more
// "<tool_call>{"name":..,"arguments":..}</tool_call>" blocks, which may be
// interleaved with ordinary prose. Blocks that fail to parse as a named call
// are left in place as visible text rather than silently discarded.
func extractToolCallsGeneric(text string) (content string, calls []ToolCall) {
	const openTag, closeTag = "<tool_call>", "</tool_call>"
	var contentBuf strings.Builder
	rest := text
	for {
		i := strings.Index(rest, openTag)
		if i < 0 {
			contentBuf.WriteString(rest)
			break
		}
		contentBuf.WriteString(rest[:i])
		rest = rest[i+len(openTag):]

		j := strings.Index(rest, closeTag)
		var block string
		terminated := j >= 0
		if terminated {
			block, rest = rest[:j], rest[j+len(closeTag):]
		} else {
			block, rest = rest, ""
		}

		var call struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(block)), &call); err == nil && call.Name != "" {
			calls = append(calls, ToolCall{Type: "function", Function: ToolCallFunction{Name: call.Name, Arguments: normalizeToolArguments(call.Arguments)}})
		} else {
			contentBuf.WriteString(openTag)
			contentBuf.WriteString(block)
			if terminated {
				contentBuf.WriteString(closeTag)
			}
		}
		if !terminated {
			break
		}
	}
	return strings.TrimSpace(contentBuf.String()), calls
}

// normalizeToolArguments turns a parsed "arguments" field (a JSON object in
// the common case, but defensively also accepted as a pre-encoded JSON
// string) into the OpenAI-compatible wire form: a JSON-encoded object string.
func normalizeToolArguments(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return "{}"
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			return s
		}
	}
	return trimmed
}

// extractGptOssChannels splits gpt-oss/harmony-format text into its channel
// segments: "analysis" becomes reasoning, "final" becomes content, and
// "commentary to=functions.NAME" becomes a tool call. Segments are delimited
// by "<|channel|>NAME[ to=RECIPIENT]<|message|>BODY" and closed by "<|end|>"
// or "<|call|>"; an unterminated trailing segment (generation cut off
// mid-message) is still captured. Text with no channel markers at all
// (a model that never entered the harmony format) is returned unchanged as
// content, so this is a safe no-op against non-channel output.
func extractGptOssChannels(text string) (content, reasoning string, calls []ToolCall) {
	const chanTag, msgTag, endTag, callTag = "<|channel|>", "<|message|>", "<|end|>", "<|call|>"
	var contentParts, reasoningParts []string
	sawChannel := false
	rest := text
	for {
		ci := strings.Index(rest, chanTag)
		if ci < 0 {
			break
		}
		sawChannel = true
		rest = rest[ci+len(chanTag):]
		mi := strings.Index(rest, msgTag)
		if mi < 0 {
			break // truncated before any message body; nothing more to parse
		}
		header := strings.TrimSpace(rest[:mi])
		rest = rest[mi+len(msgTag):]

		end, closerLen := len(rest), 0
		if i := strings.Index(rest, endTag); i >= 0 && i < end {
			end, closerLen = i, len(endTag)
		}
		if i := strings.Index(rest, callTag); i >= 0 && i < end {
			end, closerLen = i, len(callTag)
		}
		body := strings.TrimSpace(rest[:end])
		rest = rest[min(end+closerLen, len(rest)):]

		channel, recipient, _ := strings.Cut(header, " to=")
		switch strings.TrimSpace(channel) {
		case "analysis":
			reasoningParts = append(reasoningParts, body)
		case "commentary":
			name := strings.TrimPrefix(strings.TrimSpace(recipient), "functions.")
			if name != "" {
				calls = append(calls, ToolCall{Type: "function", Function: ToolCallFunction{Name: name, Arguments: normalizeGptOssArgs(body)}})
			} else if body != "" {
				contentParts = append(contentParts, body)
			}
		default: // "final" and anything unrecognized
			if body != "" {
				contentParts = append(contentParts, body)
			}
		}
		if rest == "" {
			break
		}
	}
	if !sawChannel {
		return strings.TrimSpace(text), "", nil
	}
	return strings.TrimSpace(strings.Join(contentParts, "\n\n")), strings.TrimSpace(strings.Join(reasoningParts, "\n\n")), calls
}

// normalizeGptOssArgs mirrors normalizeToolArguments for gpt-oss commentary
// bodies, which are raw JSON text rather than a decoded json.RawMessage.
func normalizeGptOssArgs(body string) string {
	trimmed := strings.TrimSpace(body)
	if trimmed == "" {
		return "{}"
	}
	return trimmed
}
