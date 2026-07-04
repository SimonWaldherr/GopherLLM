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
