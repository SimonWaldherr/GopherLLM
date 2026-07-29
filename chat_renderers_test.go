package gopherllm

import (
	"strings"
	"testing"
)

// newChatTokenizer builds a SentencePiece-style tokenizer with the base
// character vocabulary plus the given control tokens, so the chat renderers can
// be exercised without a real model.
func newChatTokenizer(specials ...string) *Tokenizer {
	tok := newInstTestTokenizer()
	for _, s := range specials {
		if _, ok := tok.TokenToID[s]; !ok {
			addSpecial(tok, s)
		}
	}
	return tok
}

func hasAll(tokens []uint32, ids ...uint32) bool {
	for _, id := range ids {
		if indexOfToken(tokens, id) < 0 {
			return false
		}
	}
	return true
}

func TestRenderChatMLMessages(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_end|>")
	r := &Runner{tok: tok, arch: "qwen2"}
	tokens, ok := r.renderChatMLMessages([]ChatMessage{UserMessage("hi")}, "be nice")
	if !ok {
		t.Fatal("ok=false")
	}
	if !hasAll(tokens, tok.TokenToID["<|im_start|>"], tok.TokenToID["<|im_end|>"]) {
		t.Fatalf("missing im_start/im_end: %v", tokens)
	}
	// Ends open for the assistant turn (last im_start has no following im_end).
	if last := indexOfToken(tokens, tok.TokenToID["<|im_end|>"]); last >= len(tokens)-1 {
		t.Fatal("expected assistant turn left open after final im_end")
	}
	if _, ok := (&Runner{tok: newInstTestTokenizer(), arch: "qwen2"}).renderChatMLMessages(nil, ""); ok {
		t.Fatal("render without im_start/im_end tokens should fail")
	}
}

// TestRenderChatMLMessagesOpensThinkTagOnlyForCheckpointsThatDefaultToIt
// guards a real bug: Nemotron-H's own chat_template defaults
// enable_thinking to true and then emits an unclosed '<think>\n' after
// '<|im_start|>assistant\n'. The generic ChatML renderer left that out
// entirely, which fed the checkpoint a prompt suffix it never saw in
// training and derailed generation into word salad. The fix checks each
// checkpoint's own chat_template before adding the tag, so plain ChatML
// models (no thinking convention) are unaffected.
func TestRenderChatMLMessagesOpensThinkTagOnlyForCheckpointsThatDefaultToIt(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_end|>", "<think>")

	defaultsToThinking := &Runner{
		tok:  tok,
		arch: "nemotron_h",
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "str", Value: "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}" +
				"{%- if enable_thinking %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- else %}{{- '<|im_start|>assistant\\n<think></think>' }}{%- endif %}"},
		}},
	}
	tokens, ok := defaultsToThinking.renderChatMLMessages([]ChatMessage{UserMessage("hi")}, "")
	if !ok {
		t.Fatal("ok=false")
	}
	thinkID := tok.TokenToID["<think>"]
	imStartID := tok.TokenToID["<|im_start|>"]
	lastImStart, lastThink := -1, -1
	for i, id := range tokens {
		if id == imStartID {
			lastImStart = i
		}
		if id == thinkID {
			lastThink = i
		}
	}
	if lastThink < 0 || lastThink < lastImStart {
		t.Fatalf("expected an open <think> tag after the final assistant turn: %v", tokens)
	}

	plainChatML := &Runner{
		tok:  tok,
		arch: "stablelm",
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "str", Value: "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"},
		}},
	}
	tokens, ok = plainChatML.renderChatMLMessages([]ChatMessage{UserMessage("hi")}, "")
	if !ok {
		t.Fatal("ok=false")
	}
	if hasAll(tokens, tok.TokenToID["<think>"]) {
		t.Fatalf("plain ChatML checkpoints must not get a fabricated think tag: %v", tokens)
	}
}

func TestRenderQwen35MessagesOpensThinkingAndStopsAtChatMLEnd(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_end|>", "<think>")
	r := &Runner{tok: tok, arch: "qwen35"}
	tokens, ok := r.renderQwen35Messages([]ChatMessage{UserMessage("hi")}, "be precise")
	if !ok {
		t.Fatal("ok=false")
	}
	if text := decodeAll(tok, tokens); !strings.HasSuffix(text, "assistant\n<think>\n") {
		t.Fatalf("prompt should end in an open Qwen thinking turn: %q", text)
	}
	if !r.isStopToken(tok.TokenToID["<|im_end|>"]) {
		t.Fatal("Qwen3.5/3.6 must stop at <|im_end|>")
	}
}

func TestRenderHeaderChatMessages(t *testing.T) {
	tok := newChatTokenizer("<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
	r := &Runner{tok: tok, arch: "llama3"}
	tokens, ok := r.renderHeaderChatMessages([]ChatMessage{UserMessage("hi")}, "sys")
	if !ok {
		t.Fatal("ok=false")
	}
	if tokens[0] != tok.TokenToID["<|begin_of_text|>"] {
		t.Fatalf("expected BOT first, got %d", tokens[0])
	}
	if !hasAll(tokens, tok.TokenToID["<|start_header_id|>"], tok.TokenToID["<|end_header_id|>"], tok.TokenToID["<|eot_id|>"]) {
		t.Fatalf("missing header tokens: %v", tokens)
	}
}

func TestRenderPhiMessages(t *testing.T) {
	tok := newChatTokenizer("<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
	r := &Runner{tok: tok, arch: "phi3"}
	tokens, ok := r.renderPhiMessages([]ChatMessage{UserMessage("hi")}, "sys")
	if !ok {
		t.Fatal("ok=false")
	}
	if !hasAll(tokens, tok.TokenToID["<|system|>"], tok.TokenToID["<|user|>"], tok.TokenToID["<|end|>"]) {
		t.Fatalf("missing phi tokens: %v", tokens)
	}
	if tokens[len(tokens)-1] != tok.TokenToID["<|assistant|>"] && indexOfToken(tokens, tok.TokenToID["<|assistant|>"]) < 0 {
		t.Fatal("expected assistant token")
	}
}

func TestRenderPhi4MessagesUsesIMSeparator(t *testing.T) {
	tok := newChatTokenizer("<|im_start|>", "<|im_sep|>", "<|im_end|>")
	imStart := tok.TokenToID["<|im_start|>"]
	imSep := tok.TokenToID["<|im_sep|>"]
	imEnd := tok.TokenToID["<|im_end|>"]
	// Phi-4 declares <|im_end|> as EOS, so this also guards natural stopping.
	tok.EOSID = imEnd
	r := &Runner{
		tok:  tok,
		arch: "phi3",
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "str", Value: "<|im_start|>role<|im_sep|>content<|im_end|>"},
		}},
	}
	if kind := r.chatTemplateKind(); kind != "phi4-chat" {
		t.Fatalf("template kind = %q, want phi4-chat", kind)
	}
	tokens := r.renderMessages([]ChatMessage{UserMessage("hi")}, "sys", nil)
	if got := countToken(tokens, imStart); got != 3 {
		t.Fatalf("im_start count = %d, want system/user/open-assistant", got)
	}
	if got := countToken(tokens, imSep); got != 3 {
		t.Fatalf("im_sep count = %d, want one per turn", got)
	}
	if got := countToken(tokens, imEnd); got != 2 {
		t.Fatalf("im_end count = %d, want closed system and user turns", got)
	}
	wantTail := append([]uint32{imStart}, tok.EncodeWithoutBOS("assistant")...)
	wantTail = append(wantTail, imSep)
	if len(tokens) < len(wantTail) {
		t.Fatalf("tokens too short: %v", tokens)
	}
	for i, want := range wantTail {
		if got := tokens[len(tokens)-len(wantTail)+i]; got != want {
			t.Fatalf("assistant generation suffix[%d] = %d, want %d; tokens=%v", i, got, want, tokens)
		}
	}
	if !r.isStopToken(imEnd) {
		t.Fatal("Phi-4 must stop at <|im_end|>")
	}
}

func TestRenderDeepSeekMessages(t *testing.T) {
	tok := newChatTokenizer("<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>")
	r := &Runner{tok: tok, arch: "qwen2"}
	tokens, ok := r.renderDeepSeekR1QwenMessages([]ChatMessage{UserMessage("hi")}, "sys")
	if !ok {
		t.Fatal("ok=false")
	}
	if tokens[0] != tok.BOSID {
		t.Fatalf("expected BOS first, got %d", tokens[0])
	}
	if tokens[len(tokens)-1] != tok.TokenToID["<｜Assistant｜>"] {
		t.Fatal("expected trailing Assistant token")
	}
}

func TestRenderGraniteMessages(t *testing.T) {
	tok := newChatTokenizer("<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>")
	r := &Runner{tok: tok, arch: "granite"}
	tokens, ok := r.renderGraniteMessages([]ChatMessage{UserMessage("hi")}, "sys")
	if !ok {
		t.Fatal("ok=false")
	}
	if !hasAll(tokens, tok.TokenToID["<|start_of_role|>"], tok.TokenToID["<|end_of_role|>"], tok.TokenToID["<|end_of_text|>"]) {
		t.Fatalf("missing granite tokens: %v", tokens)
	}
}

func TestRenderGptOssMessages(t *testing.T) {
	tok := newChatTokenizer("<|start|>", "<|channel|>", "<|message|>", "<|end|>")
	r := &Runner{tok: tok, arch: "gpt-oss"}
	tokens := r.renderGptOssMessages([]ChatMessage{UserMessage("hi")}, "sys")
	if len(tokens) == 0 {
		t.Fatal("empty gpt-oss render")
	}
	if !hasAll(tokens, tok.TokenToID["<|start|>"], tok.TokenToID["<|message|>"], tok.TokenToID["<|channel|>"]) {
		t.Fatalf("missing gpt-oss tokens: %v", tokens)
	}
}

func TestRenderPlainMessagesFallback(t *testing.T) {
	tok := newInstTestTokenizer()
	// Remove the [INST] tokens so no template matches -> plain fallback.
	delete(tok.TokenToID, "[INST]")
	delete(tok.TokenToID, "[/INST]")
	r := &Runner{tok: tok, arch: "llama", gguf: &GGUFFile{Metadata: map[string]MetaValue{}}}
	tokens := r.renderMessages([]ChatMessage{UserMessage("hi")}, "sys", nil)
	if len(tokens) == 0 || tokens[0] != tok.BOSID {
		t.Fatalf("plain fallback should start with BOS: %v", tokens)
	}
}
