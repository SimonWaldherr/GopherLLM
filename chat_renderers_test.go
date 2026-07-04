package main

import "testing"

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
