package gopherllm

import (
	"encoding/json"
	"strconv"
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
	kind := r.chatTemplateKind()
	if r.arch == "gpt-oss" {
		content, reasoning, calls = extractGptOssChannels(raw)
	} else {
		content, reasoning = extractThink(raw)
		if len(tools) > 0 {
			switch {
			case qwen35Family(r.arch):
				content, calls = extractToolCallsQwen35(content)
			case kind == "mistral-inst":
				content, calls = extractToolCallsMistral(content)
			case kind == "kimi-chat":
				content, calls = extractToolCallsKimi(content)
			default:
				content, calls = extractToolCallsGeneric(content)
			}
		}
	}
	for i := range calls {
		if calls[i].Type == "" {
			calls[i].Type = "function"
		}
		if kind == "kimi-chat" {
			if _, _, ok := parseKimiToolCallID(calls[i].ID); !ok {
				calls[i].ID = kimiToolCallID(calls[i].Function.Name, i)
			}
		} else if !validToolCallID(calls[i].ID) {
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
// thought into the visible answer. A leading </think> with no prior <think>
// (newer DeepSeek-R1 templates force the opening tag into the PROMPT, so the
// model's output begins mid-reasoning) likewise marks everything before it as
// reasoning.
func extractThink(text string) (content, reasoning string) {
	const openTag, closeTag = "<think>", "</think>"
	var contentBuf strings.Builder
	var reasoningParts []string
	rest := text
	if closeIdx := strings.Index(rest, closeTag); closeIdx >= 0 {
		openIdx := strings.Index(rest, openTag)
		if openIdx < 0 || closeIdx < openIdx {
			reasoningParts = append(reasoningParts, rest[:closeIdx])
			rest = rest[closeIdx+len(closeTag):]
		}
	}
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

// ThinkStreamSplitter separates <think> blocks while text is still arriving
// in arbitrary-sized chunks. Tokens need not align with tags, so a short
// suffix is retained until it can no longer be the beginning of a marker.
// It deliberately preserves whitespace: the final GenerationResult remains
// the canonical trimmed representation, while stream consumers receive each
// character exactly once.
type ThinkStreamSplitter struct {
	inReasoning bool
	pending     string
}

func NewThinkStreamSplitter(initialReasoning bool) ThinkStreamSplitter {
	return ThinkStreamSplitter{inReasoning: initialReasoning}
}

// Push accepts another raw model-text chunk. emit receives whether the text
// belongs to the reasoning channel; returning false stops processing.
func (s *ThinkStreamSplitter) Push(text string, emit func(reasoning bool, text string) bool) bool {
	s.pending += text
	for {
		marker := "<think>"
		if s.inReasoning {
			marker = "</think>"
		}
		if i := strings.Index(s.pending, marker); i >= 0 {
			if i > 0 && !emit(s.inReasoning, s.pending[:i]) {
				return false
			}
			s.pending = s.pending[i+len(marker):]
			s.inReasoning = !s.inReasoning
			continue
		}

		// Keep only the longest suffix that could become a complete marker
		// when the next token arrives (for example, "<thi").
		keep := markerPrefixSuffixLen(s.pending, marker)
		if n := len(s.pending) - keep; n > 0 {
			if !emit(s.inReasoning, s.pending[:n]) {
				return false
			}
			s.pending = s.pending[n:]
		}
		return true
	}
}

// Flush emits a final partial marker literally. That mirrors extractThink:
// incomplete tag text is ordinary content unless it follows an actual
// opening <think> marker.
func (s *ThinkStreamSplitter) Flush(emit func(reasoning bool, text string) bool) bool {
	if s.pending == "" {
		return true
	}
	text := s.pending
	s.pending = ""
	return emit(s.inReasoning, text)
}

func markerPrefixSuffixLen(text, marker string) int {
	maxLen := min(len(text), len(marker)-1)
	for n := maxLen; n > 0; n-- {
		if strings.HasSuffix(text, marker[:n]) {
			return n
		}
	}
	return 0
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

// extractToolCallsKimi parses Kimi K2's native tool protocol, documented by
// Moonshot as:
//
//	<|tool_calls_section_begin|>
//	<|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json}<|tool_call_end|>
//	<|tool_calls_section_end|>
//
// A malformed section remains visible text, matching the generic parser's
// conservative behavior: incomplete model output must not silently discard a
// user-visible answer or fabricate a call.
func extractToolCallsKimi(text string) (content string, calls []ToolCall) {
	const sectionStart = "<|tool_calls_section_begin|>"
	const sectionEnd = "<|tool_calls_section_end|>"
	if !strings.Contains(text, sectionStart) {
		return text, nil
	}

	var visible strings.Builder
	rest := text
	for {
		start := strings.Index(rest, sectionStart)
		if start < 0 {
			visible.WriteString(rest)
			break
		}
		visible.WriteString(rest[:start])
		afterStart := rest[start+len(sectionStart):]
		end := strings.Index(afterStart, sectionEnd)
		if end < 0 {
			// Preserve an incomplete section verbatim.
			visible.WriteString(rest[start:])
			break
		}
		section := afterStart[:end]
		parsed, ok := parseKimiToolCallSection(section)
		if !ok {
			visible.WriteString(rest[start : start+len(sectionStart)+end+len(sectionEnd)])
		} else {
			calls = append(calls, parsed...)
		}
		rest = afterStart[end+len(sectionEnd):]
	}
	return strings.TrimSpace(visible.String()), calls
}

func parseKimiToolCallSection(section string) ([]ToolCall, bool) {
	const callStart = "<|tool_call_begin|>"
	const argumentStart = "<|tool_call_argument_begin|>"
	const callEnd = "<|tool_call_end|>"

	rest := section
	calls := make([]ToolCall, 0, 1)
	for {
		rest = strings.TrimLeft(rest, " \t\r\n")
		if rest == "" {
			return calls, len(calls) > 0
		}
		if !strings.HasPrefix(rest, callStart) {
			return nil, false
		}
		rest = rest[len(callStart):]
		argumentIndex := strings.Index(rest, argumentStart)
		if argumentIndex < 0 {
			return nil, false
		}
		id := strings.TrimSpace(rest[:argumentIndex])
		name, _, ok := parseKimiToolCallID(id)
		if !ok {
			return nil, false
		}
		rest = rest[argumentIndex+len(argumentStart):]
		endIndex := strings.Index(rest, callEnd)
		if endIndex < 0 {
			return nil, false
		}
		arguments := strings.TrimSpace(rest[:endIndex])
		if arguments == "" {
			arguments = "{}"
		}
		calls = append(calls, ToolCall{
			ID:   id,
			Type: "function",
			Function: ToolCallFunction{
				Name:      name,
				Arguments: arguments,
			},
		})
		rest = rest[endIndex+len(callEnd):]
	}
}

// parseKimiToolCallID validates the identifier grammar generated by K2. The
// name is deliberately kept opaque beyond non-whitespace validation so valid
// OpenAI function names with punctuation such as '-' round-trip unchanged.
func parseKimiToolCallID(id string) (name string, index int, ok bool) {
	rest, ok := strings.CutPrefix(id, "functions.")
	if !ok {
		return "", 0, false
	}
	name, indexText, ok := strings.Cut(rest, ":")
	if !ok || name == "" || strings.TrimSpace(name) != name || strings.TrimSpace(indexText) != indexText || indexText == "" {
		return "", 0, false
	}
	index, err := strconv.Atoi(indexText)
	if err != nil || index < 0 {
		return "", 0, false
	}
	return name, index, true
}

func kimiToolCallID(name string, index int) string {
	return "functions." + name + ":" + strconv.Itoa(index)
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

// extractToolCallsQwen35 parses Qwen3.5/3.6's native XML-like convention:
//
//	<tool_call>
//	<function=name>
//	<parameter=key>
//	value
//	</parameter>
//	</function>
//	</tool_call>
//
// A malformed block deliberately remains visible text. That avoids turning an
// incomplete streamed answer into an invented tool invocation.
func extractToolCallsQwen35(text string) (content string, calls []ToolCall) {
	const openTag, closeTag = "<tool_call>", "</tool_call>"
	var visible strings.Builder
	rest := text
	for {
		start := strings.Index(rest, openTag)
		if start < 0 {
			visible.WriteString(rest)
			break
		}
		visible.WriteString(rest[:start])
		afterStart := rest[start+len(openTag):]
		end := strings.Index(afterStart, closeTag)
		if end < 0 {
			visible.WriteString(rest[start:])
			break
		}
		block := afterStart[:end]
		if call, ok := parseQwen35ToolCall(block); ok {
			calls = append(calls, call)
		} else {
			visible.WriteString(rest[start : start+len(openTag)+end+len(closeTag)])
		}
		rest = afterStart[end+len(closeTag):]
	}
	return strings.TrimSpace(visible.String()), calls
}

func parseQwen35ToolCall(block string) (ToolCall, bool) {
	rest := strings.TrimSpace(block)
	if !strings.HasPrefix(rest, "<function=") {
		return ToolCall{}, false
	}
	headerEnd := strings.Index(rest, ">")
	if headerEnd < len("<function=") {
		return ToolCall{}, false
	}
	name := strings.TrimSpace(rest[len("<function="):headerEnd])
	if name == "" || strings.ContainsAny(name, " \t\r\n<>") {
		return ToolCall{}, false
	}
	rest = strings.TrimLeft(rest[headerEnd+1:], "\r\n")
	args := map[string]json.RawMessage{}
	for {
		rest = strings.TrimLeft(rest, " \t\r\n")
		if strings.HasPrefix(rest, "</function>") {
			if strings.TrimSpace(rest[len("</function>"):]) != "" {
				return ToolCall{}, false
			}
			encoded, err := json.Marshal(args)
			if err != nil {
				return ToolCall{}, false
			}
			return ToolCall{Type: "function", Function: ToolCallFunction{Name: name, Arguments: string(encoded)}}, true
		}
		if !strings.HasPrefix(rest, "<parameter=") {
			return ToolCall{}, false
		}
		parameterEnd := strings.Index(rest, ">")
		if parameterEnd < len("<parameter=") {
			return ToolCall{}, false
		}
		parameter := strings.TrimSpace(rest[len("<parameter="):parameterEnd])
		if parameter == "" || strings.ContainsAny(parameter, " \t\r\n<>") {
			return ToolCall{}, false
		}
		rest = strings.TrimLeft(rest[parameterEnd+1:], "\r\n")
		valueEnd := strings.Index(rest, "</parameter>")
		if valueEnd < 0 {
			return ToolCall{}, false
		}
		if _, duplicate := args[parameter]; duplicate {
			return ToolCall{}, false
		}
		value := strings.TrimSpace(rest[:valueEnd])
		if json.Valid([]byte(value)) {
			args[parameter] = json.RawMessage(value)
		} else {
			encoded, err := json.Marshal(value)
			if err != nil {
				return ToolCall{}, false
			}
			args[parameter] = encoded
		}
		rest = rest[valueEnd+len("</parameter>"):]
	}
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
