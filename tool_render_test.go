package gopherllm

import (
	"strings"
	"testing"
)

func decodeAll(tok *Tokenizer, tokens []uint32) string {
	var sb strings.Builder
	for _, id := range tokens {
		sb.WriteString(tok.DecodeToken(id))
	}
	return sb.String()
}

func sampleTool() ToolDefinition {
	return ToolDefinition{Type: "function", Function: ToolFunctionDef{
		Name:        "get_weather",
		Description: "Get the current weather for a city",
		Parameters:  []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}}
}

func newMistralToolTokenizer() *Tokenizer {
	return newChatTokenizer("[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", "[TOOL_CALLS]", "[ARGS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]")
}

func newLlama31ToolTokenizer() *Tokenizer {
	return newChatTokenizer(
		"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
		"<|eot_id|>", "<|eom_id|>", "<|python_tag|>",
	)
}

func newLlama31ToolRunner(tok *Tokenizer) *Runner {
	return &Runner{tok: tok, arch: "llama", gguf: &GGUFFile{Metadata: map[string]MetaValue{
		// Deliberately contains the Llama-3.1-specific markers rather than
		// merely the shared header protocol, so this verifies native-template
		// selection instead of the older Llama 3 header renderer.
		"tokenizer.chat_template": {Kind: "str", Value: `{% set date_string = "26 Jul 2024" %}{% set tools_in_user_message = true %}<|python_tag|><|eom_id|><|start_header_id|>{{ role }}<|end_header_id|><|eot_id|>`},
	}}}
}

func TestLlama31NativeToolTemplateSystemDateAndIPythonRound(t *testing.T) {
	tok := newLlama31ToolTokenizer()
	r := newLlama31ToolRunner(tok)
	if kind := r.chatTemplateKind(); kind != "llama31-chat" {
		t.Fatalf("template kind = %q, want llama31-chat", kind)
	}
	messages := []ChatMessage{
		{Role: ChatRoleSystem, Content: "Answer concisely."},
		UserMessage("weather in Berlin?"),
		{Role: ChatRoleAssistant, ToolCalls: []ToolCall{{
			ID: "abc123XYZ", Type: "function",
			Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Berlin"}`},
		}}},
		ToolResultMessage("abc123XYZ", "get_weather", `{"temperature":18}`),
	}
	text := decodeAll(tok, r.renderMessages(messages, "ignored default system", []ToolDefinition{sampleTool()}))
	for _, want := range []string{
		"<|begin_of_text|>", "Cutting Knowledge Date: December 2023", "Today Date: 26 Jul 2024",
		"Answer concisely.", "Environment: ipython", "Given the following functions",
		`"get_weather"`, `{"name": "get_weather", "parameters": {"city":"Berlin"}}`,
		"ipython", `{"temperature":18}`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("native Llama 3.1 prompt missing %q: %q", want, text)
		}
	}
	if strings.Contains(text, "ignored default system") {
		t.Fatalf("leading system message must replace the default system prompt: %q", text)
	}
	if !strings.Contains(text, "value}.Do not use variables.") {
		t.Fatalf("tool instruction must keep native Jinja whitespace behavior: %q", text)
	}
	if strings.Contains(text, "<tool_call>") || strings.Contains(text, "<tool_response") {
		t.Fatalf("Llama 3.1 should not fall back to generic tool markup: %q", text)
	}
	if got := strings.Count(text, "<|eot_id|>"); got != 4 {
		t.Fatalf("eot count = %d, want system/user/tool-call/ipython turns; prompt=%q", got, text)
	}
	lastHeader := strings.LastIndex(text, "<|start_header_id|>")
	if lastHeader < 0 || !strings.Contains(text[lastHeader:], "assistant<|end_header_id|>") || strings.Contains(text[lastHeader:], "<|eot_id|>") {
		t.Fatalf("prompt must end with an open assistant header: %q", text)
	}
}

func TestLlama31ToolParserAndClassifier(t *testing.T) {
	r := newLlama31ToolRunner(newLlama31ToolTokenizer())
	raw := `{"name":"get_weather","parameters":{"city":"Berlin","days":2}}`
	content, calls := extractToolCallsLlama31(raw)
	if content != "" || len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("parsed content/calls = %q / %#v", content, calls)
	}
	if calls[0].Function.Arguments != `{"city":"Berlin","days":2}` {
		t.Fatalf("arguments = %q", calls[0].Function.Arguments)
	}
	content, reasoning, calls := r.classifyOutput(raw, []ToolDefinition{sampleTool()}, NewRng(1))
	if content != "" || reasoning != "" || len(calls) != 1 || !validToolCallID(calls[0].ID) {
		t.Fatalf("classifyOutput = content %q reasoning %q calls %#v", content, reasoning, calls)
	}

	malformed := `{"name":"get_weather","parameters":"Berlin"}`
	content, calls = extractToolCallsLlama31(malformed)
	if content != malformed || calls != nil {
		t.Fatalf("malformed native call must stay visible: content=%q calls=%#v", content, calls)
	}
}

func TestLlama31TemplateDateUsesJinjaAssignmentAfterDefinedCheck(t *testing.T) {
	r := newLlama31ToolRunner(newLlama31ToolTokenizer())
	r.gguf.Metadata["tokenizer.chat_template"] = MetaValue{Kind: "str", Value: `{% if not date_string is defined %}{% set date_string = "01 Jan 2030" %}{% endif %}<|python_tag|><|eom_id|>{% set tools_in_user_message = true %}`}
	if got := r.llama31TemplateDate(); got != "01 Jan 2030" {
		t.Fatalf("template date = %q, want the Jinja assignment", got)
	}
}

func TestMistralRenderInjectsAvailableTools(t *testing.T) {
	tok := newMistralToolTokenizer()
	r := &Runner{tok: tok, arch: "ministral"}
	tokens, ok := r.renderMistralInstMessages([]ChatMessage{UserMessage("weather?")}, "", []ToolDefinition{sampleTool()})
	if !ok {
		t.Fatal("ok=false")
	}
	start := indexOfToken(tokens, tok.TokenToID["[AVAILABLE_TOOLS]"])
	end := indexOfToken(tokens, tok.TokenToID["[/AVAILABLE_TOOLS]"])
	instIdx := indexOfToken(tokens, tok.TokenToID["[INST]"])
	if start < 0 || end < 0 || instIdx < 0 {
		t.Fatalf("missing markers: %v", tokens)
	}
	if !(start < end && end < instIdx) {
		t.Fatalf("expected [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS] to precede [INST]: start=%d end=%d inst=%d", start, end, instIdx)
	}
	payload := decodeAll(tok, tokens[start+1:end])
	if !strings.Contains(payload, "get_weather") || !strings.Contains(payload, "\"type\":\"function\"") {
		t.Fatalf("tool payload missing expected fields: %q", payload)
	}
}

func TestMistralRenderNoToolsOmitsMarkers(t *testing.T) {
	tok := newMistralToolTokenizer()
	r := &Runner{tok: tok, arch: "ministral"}
	tokens, ok := r.renderMistralInstMessages([]ChatMessage{UserMessage("hi")}, "", nil)
	if !ok {
		t.Fatal("ok=false")
	}
	if indexOfToken(tokens, tok.TokenToID["[AVAILABLE_TOOLS]"]) >= 0 {
		t.Fatalf("did not expect [AVAILABLE_TOOLS] with no tools: %v", tokens)
	}
}

func TestMistralRenderReplaysAssistantToolCalls(t *testing.T) {
	tok := newMistralToolTokenizer()
	r := &Runner{tok: tok, arch: "ministral"}
	messages := []ChatMessage{
		UserMessage("weather in Berlin?"),
		{Role: ChatRoleAssistant, ToolCalls: []ToolCall{{ID: "abc123XYZ", Type: "function", Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city": "Berlin"}`}}}},
		ToolResultMessage("abc123XYZ", "get_weather", "18C, sunny"),
	}
	tokens, ok := r.renderMistralInstMessages(messages, "", []ToolDefinition{sampleTool()})
	if !ok {
		t.Fatal("ok=false")
	}
	callIdx := indexOfToken(tokens, tok.TokenToID["[TOOL_CALLS]"])
	argsIdx := indexOfToken(tokens, tok.TokenToID["[ARGS]"])
	resultsIdx := indexOfToken(tokens, tok.TokenToID["[TOOL_RESULTS]"])
	resultsEndIdx := indexOfToken(tokens, tok.TokenToID["[/TOOL_RESULTS]"])
	if callIdx < 0 || argsIdx < 0 || resultsIdx < 0 || resultsEndIdx < 0 {
		t.Fatalf("missing tool-call/result markers: %v", tokens)
	}
	if !(callIdx < argsIdx && argsIdx < resultsIdx && resultsIdx < resultsEndIdx) {
		t.Fatalf("markers out of order: call=%d args=%d results=%d resultsEnd=%d", callIdx, argsIdx, resultsIdx, resultsEndIdx)
	}
	// Leading space is expected SentencePiece word-boundary behavior (the same
	// convention every other renderer already relies on for e.g. [INST] content).
	name := strings.TrimSpace(decodeAll(tok, tokens[callIdx+1:argsIdx]))
	if name != "get_weather" {
		t.Fatalf("call name = %q", name)
	}
	result := strings.TrimSpace(decodeAll(tok, tokens[resultsIdx+1:resultsEndIdx]))
	if result != "18C, sunny" {
		t.Fatalf("result content = %q", result)
	}
	// Assistant turn (tool-call or not) always ends with EOS, and the eos
	// must come right after the call/args block, before the tool result.
	if tokens[argsIdx+1+len(tok.EncodeWithoutBOS(`{"city": "Berlin"}`))] != tok.EOSID {
		t.Fatalf("expected EOS immediately after the tool-call args: %v", tokens)
	}
}

func TestInjectGenericToolsNoActivityIsNoOp(t *testing.T) {
	messages := []ChatMessage{UserMessage("hi")}
	out, sys := injectGenericTools(messages, "be nice", nil)
	if len(out) != 1 || out[0].Content != "hi" || sys != "be nice" {
		t.Fatalf("out=%+v sys=%q", out, sys)
	}
}

func TestInjectGenericToolsAppendsToDefaultSystemPrompt(t *testing.T) {
	out, sys := injectGenericTools([]ChatMessage{UserMessage("hi")}, "be nice", []ToolDefinition{sampleTool()})
	if len(out) != 1 || out[0].Content != "hi" {
		t.Fatalf("messages should be untouched: %+v", out)
	}
	if !strings.HasPrefix(sys, "be nice") {
		t.Fatalf("system prompt should be preserved as a prefix: %q", sys)
	}
	if !strings.Contains(sys, "<tool_call>") || !strings.Contains(sys, "get_weather") {
		t.Fatalf("system prompt missing tool listing: %q", sys)
	}
}

func TestInjectGenericToolsMergesIntoExplicitSystemMessage(t *testing.T) {
	messages := []ChatMessage{{Role: ChatRoleSystem, Content: "custom system"}, UserMessage("hi")}
	out, sys := injectGenericTools(messages, "ignored default", []ToolDefinition{sampleTool()})
	if sys != "ignored default" {
		t.Fatalf("systemPrompt should be untouched when an explicit system message exists: %q", sys)
	}
	if !strings.HasPrefix(out[0].Content, "custom system") || !strings.Contains(out[0].Content, "get_weather") {
		t.Fatalf("explicit system message should absorb the tool listing: %q", out[0].Content)
	}
}

func TestInjectGenericToolsRewritesAssistantAndToolMessages(t *testing.T) {
	messages := []ChatMessage{
		UserMessage("weather in Berlin?"),
		{Role: ChatRoleAssistant, ToolCalls: []ToolCall{{Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Berlin"}`}}}},
		ToolResultMessage("id1", "get_weather", "18C, sunny"),
	}
	out, _ := injectGenericTools(messages, "", nil)
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if !strings.Contains(out[1].Content, "<tool_call>") || !strings.Contains(out[1].Content, "get_weather") {
		t.Fatalf("assistant tool call not rendered: %q", out[1].Content)
	}
	if out[2].Role != ChatRoleUser {
		t.Fatalf("tool message should be remapped to user for generic templates, got %v", out[2].Role)
	}
	if !strings.Contains(out[2].Content, "<tool_response") || !strings.Contains(out[2].Content, "18C, sunny") {
		t.Fatalf("tool result not rendered: %q", out[2].Content)
	}
}

func TestRenderMessagesChatMLCarriesToolsEndToEnd(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_end|>")
	// The shared fixture's base vocabulary also defines [INST]/[/INST] (for
	// the Mistral-focused tests), so drive template selection via the
	// chat_template metadata (as a real qwen2 GGUF would have) rather than
	// relying on the ambiguous special-token sniffing fallback.
	r := &Runner{tok: tok, arch: "qwen2", gguf: &GGUFFile{Metadata: map[string]MetaValue{
		"tokenizer.chat_template": {Kind: "string", Value: "{{bos_token}}<|im_start|>{{content}}<|im_end|>"},
	}}}
	tokens := r.renderMessages([]ChatMessage{UserMessage("weather?")}, "", []ToolDefinition{sampleTool()})
	text := decodeAll(tok, tokens)
	if !strings.Contains(text, "get_weather") || !strings.Contains(text, "<tool_call>") {
		t.Fatalf("chatml render missing tool listing: %q", text)
	}
	if strings.Contains(text, "[AVAILABLE_TOOLS]") {
		t.Fatalf("chatml render should use the generic convention, not Mistral's: %q", text)
	}
}

func TestQwen35NativeToolsRenderAndParseEndToEnd(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_end|>", "<think>", "=")
	r := &Runner{tok: tok, arch: "qwen35"}
	messages := []ChatMessage{
		UserMessage("weather in Berlin?"),
		{Role: ChatRoleAssistant, ToolCalls: []ToolCall{{
			ID: "call12345", Type: "function",
			Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Berlin","days":2}`},
		}}},
		ToolResultMessage("call12345", "get_weather", "18C, sunny"),
	}
	tokens := r.renderMessages(messages, "be concise", []ToolDefinition{sampleTool()})
	text := decodeAll(tok, tokens)
	for _, want := range []string{
		"<tools>", "get_weather", "<tool_call>", "<function=get_weather>",
		"<parameter=city>", "Berlin", "<parameter=days>", "<tool_response>", "18C, sunny",
		"<parameter=example_parameter_2>", "that can span\nmultiple lines", "<IMPORTANT>",
		"answer the question like normal", "assistant\n<think>\n",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("native Qwen tool prompt missing %q: %q", want, text)
		}
	}
	if strings.Contains(text, `{"name":"get_weather","arguments":`) {
		t.Fatalf("Qwen should replay native XML-like calls rather than generic JSON: %q", text)
	}

	raw := "I will check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
	content, calls := extractToolCallsQwen35(raw)
	if content != "I will check." || len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("parsed content=%q calls=%#v", content, calls)
	}
	if calls[0].Function.Arguments != `{"city":"Berlin","days":2}` {
		t.Fatalf("arguments = %s", calls[0].Function.Arguments)
	}
	content, reasoning, calls := r.classifyOutput("thinking</think>"+raw, []ToolDefinition{sampleTool()}, NewRng(1))
	if reasoning != "thinking" || content != "I will check." || len(calls) != 1 || !validToolCallID(calls[0].ID) {
		t.Fatalf("classifyOutput = content %q reasoning %q calls %#v", content, reasoning, calls)
	}
}

func TestQwen35RendererNormalizesSystemMessagesIntoInitialTurn(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_end|>", "<think>")
	r := &Runner{tok: tok, arch: "qwen35"}
	text := decodeAll(tok, r.renderMessages([]ChatMessage{
		UserMessage("first"),
		{Role: ChatRoleSystem, Content: "later instruction"},
		UserMessage("second"),
	}, "default", nil))
	systemAt := strings.Index(text, "system\nlater instruction")
	firstUserAt := strings.Index(text, "user\nfirst")
	if systemAt < 0 || firstUserAt < 0 || systemAt > firstUserAt {
		t.Fatalf("system instruction should be normalized to the initial Qwen turn: %q", text)
	}
	if strings.Count(text, "later instruction") != 1 {
		t.Fatalf("system instruction was duplicated or dropped: %q", text)
	}
}

func TestQwen35ToolParserPreservesMalformedCall(t *testing.T) {
	raw := "before<tool_call><function=get_weather><parameter=city>Berlin</function></tool_call>after"
	content, calls := extractToolCallsQwen35(raw)
	if content != raw || calls != nil {
		t.Fatalf("malformed Qwen tool call should remain text: content=%q calls=%#v", content, calls)
	}
}
