package gopherllm

import (
	"reflect"
	"strings"
	"testing"
)

func TestTokenizerModeDetectsKimiK2PreTokenizer(t *testing.T) {
	// llama.cpp Kimi-K2 GGUFs advertise the tiktoken family through `pre`,
	// even when the generic tokenizer model field is not "gpt2".
	meta := map[string]MetaValue{
		"tokenizer.ggml.tokens": {Kind: "arr", Value: []string{"<unk>", "[BOS]", "[EOS]"}},
		"tokenizer.ggml.model":  {Kind: "str", Value: "llama"},
		"tokenizer.ggml.pre":    {Kind: "str", Value: "kimi-k2"},
	}
	tok, err := TokenizerFromMetadata(meta)
	if err != nil {
		t.Fatalf("TokenizerFromMetadata: %v", err)
	}
	if tok.Mode != TokenizerGPT2BPE {
		t.Fatalf("Mode = %v, want TokenizerGPT2BPE for kimi-k2", tok.Mode)
	}
}

func TestPretokenizeKimi(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"iPhone 12345", []string{"i", "Phone", " ", "123", "45"}},
		{"中文English", []string{"中文", "English"}},
		{"DON'T", []string{"DON'T"}},
		{"Hi\n\nthere", []string{"Hi", "\n\n", "there"}},
	}
	for _, tc := range cases {
		if got := pretokenizeKimi(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("pretokenizeKimi(%q) = %#v, want %#v", tc.in, got, tc.want)
		}
	}
}

func newKimiTestTokenizer() *Tokenizer {
	tok := newChatTokenizer(
		"<|im_system|>", "<|im_user|>", "<|im_assistant|>", "<|im_middle|>", "<|im_end|>",
		"<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>",
		"<|tool_call_argument_begin|>", "<|tool_call_end|>", "#",
	)
	return tok
}

func newKimiTestRunner(tok *Tokenizer) *Runner {
	return &Runner{tok: tok, arch: "deepseek2", gguf: &GGUFFile{Metadata: map[string]MetaValue{
		"tokenizer.chat_template": {Kind: "string", Value: "<|im_system|>{{ message }}<|im_user|>{{ message }}<|im_assistant|>{{ message }}<|im_middle|><|im_end|>"},
	}}}
}

func TestKimiChatRendererUsesNativeMarkersAndStop(t *testing.T) {
	tok := newKimiTestTokenizer()
	r := newKimiTestRunner(tok)
	if kind := r.chatTemplateKind(); kind != "kimi-chat" {
		t.Fatalf("chatTemplateKind = %q, want kimi-chat", kind)
	}

	tokens := r.renderMessages([]ChatMessage{UserMessage("hi")}, "custom system", nil)
	system := tok.TokenToID["<|im_system|>"]
	user := tok.TokenToID["<|im_user|>"]
	assistant := tok.TokenToID["<|im_assistant|>"]
	middle := tok.TokenToID["<|im_middle|>"]
	if !hasAll(tokens, system, user, assistant, middle) {
		t.Fatalf("missing Kimi role markers: %v", tokens)
	}
	if tokens[len(tokens)-1] != middle {
		t.Fatalf("last token = %d, want open assistant <|im_middle|> %d", tokens[len(tokens)-1], middle)
	}
	if !(indexOfToken(tokens, system) < indexOfToken(tokens, user) && indexOfToken(tokens, user) < len(tokens)-2) {
		t.Fatalf("unexpected Kimi turn order: %v", tokens)
	}
	if !r.isStopToken(tok.TokenToID["<|im_end|>"]) {
		t.Fatal("<|im_end|> must terminate a Kimi assistant response")
	}
}

func TestKimiChatRendererAndParserUseNativeTools(t *testing.T) {
	tok := newKimiTestTokenizer()
	r := newKimiTestRunner(tok)
	messages := []ChatMessage{
		UserMessage("weather in Berlin?"),
		{Role: ChatRoleAssistant, ToolCalls: []ToolCall{{
			ID:       "functions.get_weather:0",
			Type:     "function",
			Function: ToolCallFunction{Name: "get_weather", Arguments: `{"city":"Berlin"}`},
		}}},
		ToolResultMessage("functions.get_weather:0", "get_weather", "18C, sunny"),
	}
	tokens := r.renderMessages(messages, "", []ToolDefinition{sampleTool()})
	text := decodeAll(tok, tokens)
	for _, marker := range []string{
		"<|tool_calls_section_begin|>", "<|tool_call_begin|>", "<|tool_call_argument_begin|>", "<|tool_call_end|>", "<|tool_calls_section_end|>",
	} {
		if !strings.Contains(text, marker) {
			t.Fatalf("native Kimi marker %q missing from %q", marker, text)
		}
	}
	if !strings.Contains(text, "tool_declare") || !strings.Contains(text, "functions.get_weather:0") || strings.Contains(text, "<tool_call>") {
		t.Fatalf("Kimi tools should use its native declaration and call format: %q", text)
	}

	raw := `Checking.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city":"Berlin"}<|tool_call_end|><|tool_calls_section_end|>`
	content, calls := extractToolCallsKimi(raw)
	if content != "Checking." {
		t.Fatalf("content = %q, want visible prefix", content)
	}
	if len(calls) != 1 || calls[0].ID != "functions.get_weather:0" || calls[0].Function.Name != "get_weather" || calls[0].Function.Arguments != `{"city":"Berlin"}` {
		t.Fatalf("calls = %#v", calls)
	}
	content, _, calls = r.classifyOutput(raw, []ToolDefinition{sampleTool()}, NewRng(1))
	if content != "Checking." || len(calls) != 1 || calls[0].ID != "functions.get_weather:0" {
		t.Fatalf("classifyOutput = content %q calls %#v; Kimi id must survive", content, calls)
	}
}

func TestExtractKimiToolCallsPreservesMalformedSection(t *testing.T) {
	raw := `before<|tool_calls_section_begin|><|tool_call_begin|>functions.bad:0<|tool_calls_section_end|>after`
	content, calls := extractToolCallsKimi(raw)
	if calls != nil || content != raw {
		t.Fatalf("malformed Kimi tool block should remain text: content=%q calls=%#v", content, calls)
	}
}
