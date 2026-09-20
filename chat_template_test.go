package gopherllm

import (
	"hash/fnv"
	"strings"
	"testing"
)

// newInstTestTokenizer builds a SentencePiece-style tokenizer whose vocabulary
// is single characters plus the Mistral control tokens, so encoded content is a
// predictable one-token-per-character sequence (no merges fire because no
// two-character concatenation is present in the vocabulary).
func newInstTestTokenizer() *Tokenizer {
	tokens := []string{"<unk>", "<s>", "</s>", "[INST]", "[/INST]", "▁", "\n"}
	for c := 'a'; c <= 'z'; c++ {
		tokens = append(tokens, string(c))
	}
	for c := 'A'; c <= 'Z'; c++ {
		tokens = append(tokens, string(c))
	}
	for c := '0'; c <= '9'; c++ {
		tokens = append(tokens, string(c))
	}
	// JSON/tool-calling punctuation, so tool-call payloads and <tool_call>
	// convention markers survive encoding intact instead of silently dropping
	// unknown characters.
	for _, c := range "{}[]\":,_.- <>/=" {
		tokens = append(tokens, string(c))
	}
	toID := make(map[string]uint32, len(tokens))
	for i, tok := range tokens {
		toID[tok] = uint32(i)
	}
	return &Tokenizer{
		Vocab:     tokens,
		Scores:    make([]float32, len(tokens)),
		TokenToID: toID,
		Mode:      TokenizerSentencePiece,
		AddBOS:    true,
		BOSID:     1,
		EOSID:     2,
	}
}

func countToken(tokens []uint32, id uint32) int {
	n := 0
	for _, t := range tokens {
		if t == id {
			n++
		}
	}
	return n
}

func TestMistralInstRenderStructure(t *testing.T) {
	tok := newInstTestTokenizer()
	r := &Runner{tok: tok, arch: "ministral"}
	inst := tok.TokenToID["[INST]"]
	instEnd := tok.TokenToID["[/INST]"]

	tokens, _, ok, _ := r.renderMistralInstMessages([]ChatMessage{UserMessage("hi")}, "", nil)
	if !ok {
		t.Fatal("renderMistralInstMessages returned ok=false")
	}
	if len(tokens) < 3 {
		t.Fatalf("tokens too short: %v", tokens)
	}
	if tokens[0] != tok.BOSID {
		t.Fatalf("tokens[0] = %d, want BOS %d", tokens[0], tok.BOSID)
	}
	if tokens[1] != inst {
		t.Fatalf("tokens[1] = %d, want [INST] %d", tokens[1], inst)
	}
	if tokens[len(tokens)-1] != instEnd {
		t.Fatalf("last token = %d, want [/INST] %d", tokens[len(tokens)-1], instEnd)
	}
	if countToken(tokens, tok.EOSID) != 0 {
		t.Fatalf("single user turn should not contain EOS: %v", tokens)
	}
}

func TestMistralInstRenderFoldsSystemIntoLastUserTurn(t *testing.T) {
	tok := newInstTestTokenizer()
	r := &Runner{tok: tok, arch: "ministral"}

	withoutSys, _, _, _ := r.renderMistralInstMessages([]ChatMessage{UserMessage("hi")}, "", nil)
	withSys, _, _, _ := r.renderMistralInstMessages([]ChatMessage{UserMessage("hi")}, "be nice", nil)
	if len(withSys) <= len(withoutSys) {
		t.Fatalf("system prompt should lengthen the turn: with=%d without=%d", len(withSys), len(withoutSys))
	}
	// Still exactly one [INST]/[/INST] pair: system is folded in, not a new turn.
	if got := countToken(withSys, tok.TokenToID["[INST]"]); got != 1 {
		t.Fatalf("[INST] count = %d, want 1", got)
	}
}

func TestMistralInstRenderMultiTurnClosesAssistantWithEOS(t *testing.T) {
	tok := newInstTestTokenizer()
	r := &Runner{tok: tok, arch: "ministral"}
	tokens, _, _, _ := r.renderMistralInstMessages([]ChatMessage{
		UserMessage("hi"),
		AssistantMessage("yo"),
		UserMessage("bye"),
	}, "", nil)
	if got := countToken(tokens, tok.TokenToID["[INST]"]); got != 2 {
		t.Fatalf("[INST] count = %d, want 2", got)
	}
	if got := countToken(tokens, tok.EOSID); got != 1 {
		t.Fatalf("EOS count = %d, want 1 (after the assistant turn)", got)
	}
	if tokens[len(tokens)-1] != tok.TokenToID["[/INST]"] {
		t.Fatalf("conversation should end ready for assistant reply (last=[/INST])")
	}
}

func addSpecial(tok *Tokenizer, name string) uint32 {
	id := uint32(len(tok.Vocab))
	tok.Vocab = append(tok.Vocab, name)
	tok.Scores = append(tok.Scores, 0)
	tok.TokenToID[name] = id
	return id
}

func indexOfToken(tokens []uint32, id uint32) int {
	for i, t := range tokens {
		if t == id {
			return i
		}
	}
	return -1
}

func TestMistralInstRenderUsesSystemPromptTokens(t *testing.T) {
	tok := newInstTestTokenizer()
	sysStart := addSpecial(tok, "[SYSTEM_PROMPT]")
	sysEnd := addSpecial(tok, "[/SYSTEM_PROMPT]")
	r := &Runner{tok: tok, arch: "mistral3"}
	inst := tok.TokenToID["[INST]"]

	tokens, _, ok, _ := r.renderMistralInstMessages([]ChatMessage{UserMessage("hi")}, "be nice", nil)
	if !ok {
		t.Fatal("renderMistralInstMessages returned ok=false")
	}
	if tokens[0] != tok.BOSID || tokens[1] != sysStart {
		t.Fatalf("expected BOS then [SYSTEM_PROMPT], got %v", tokens[:2])
	}
	posEnd := indexOfToken(tokens, sysEnd)
	posInst := indexOfToken(tokens, inst)
	if posEnd < 0 || posInst < 0 || posEnd >= posInst {
		t.Fatalf("[/SYSTEM_PROMPT] must precede [INST]: end=%d inst=%d", posEnd, posInst)
	}
	if got := countToken(tokens, inst); got != 1 {
		t.Fatalf("[INST] count = %d, want 1 (system not folded into user turn)", got)
	}
}

// Real Mistral Small 3.x/Large/Pixtral-Large models accept more than one
// image in a single turn. Each image is spliced in as its own independent
// [IMG]...[IMG_END] block (encodeChatImage already treats every image as a
// self-contained unit, memoized by content hash), so this only needs the
// render loop itself to stop capping at one.
func TestMistralInstRenderAcceptsMultipleImagesPerMessage(t *testing.T) {
	tok := newInstTestTokenizer()
	imgTok := addSpecial(tok, "[IMG]")
	breakTok := addSpecial(tok, "[IMG_BREAK]")
	endTok := addSpecial(tok, "[IMG_END]")
	r := &Runner{tok: tok, arch: "ministral", vision: &PixtralVisionWeights{ImgBreak: []float32{0.5}}}

	imgA := ImageContent{Bytes: []byte("image-a")}
	imgB := ImageContent{Bytes: []byte("image-b")}
	r.visionCachePut(fnvHash(imgA.Bytes), visionImageCacheEntry{embeds: fakeEmbeds(2), mergedRows: 1, mergedCols: 2})
	r.visionCachePut(fnvHash(imgB.Bytes), visionImageCacheEntry{embeds: fakeEmbeds(3), mergedRows: 1, mergedCols: 3})

	tokens, embeds, ok, err := r.renderMistralInstMessages(
		[]ChatMessage{UserMessageWithImages("compare these", imgA, imgB)}, "", nil,
	)
	if err != nil || !ok {
		t.Fatalf("multi-image render should succeed, got ok=%v err=%v", ok, err)
	}
	if got := countToken(tokens, imgTok); got != 5 {
		t.Fatalf("[IMG] count = %d, want 5 (2 for image A + 3 for image B)", got)
	}
	if got := countToken(tokens, endTok); got != 2 {
		t.Fatalf("[IMG_END] count = %d, want 2 (one per image)", got)
	}
	// Image A's single row has no internal [IMG_BREAK] (mergedRows==1); the
	// two images are simply back to back with no separator of their own.
	if got := countToken(tokens, breakTok); got != 0 {
		t.Fatalf("[IMG_BREAK] count = %d, want 0 for two single-row images", got)
	}
	if len(embeds) != 5 {
		t.Fatalf("imageEmbeds has %d entries, want 5 (one per [IMG] position)", len(embeds))
	}
}

func fnvHash(b []byte) uint64 {
	h := fnv.New64a()
	h.Write(b)
	return h.Sum64()
}

func fakeEmbeds(n int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		out[i] = []float32{float32(i)}
	}
	return out
}

// The original Mistral-7B-v0.1/v0.2 tokenizer never registered [INST]/[/INST]
// as special tokens at all — its chat template spells them out as plain
// bracket text. renderMistralInstMessages must still apply in that case
// (falling back to literal-text encoding, like mistralMarker already does
// for the optional tool-calling markers) rather than silently handing the
// request to the generic plain-text renderer.
func TestMistralInstRenderFallsBackToLiteralTextForMissingControlTokens(t *testing.T) {
	tok := newInstTestTokenizer()
	delete(tok.TokenToID, "[INST]")
	delete(tok.TokenToID, "[/INST]")
	r := &Runner{tok: tok, arch: "ministral"}
	tokens, _, ok, err := r.renderMistralInstMessages([]ChatMessage{UserMessage("hi")}, "", nil)
	if !ok || err != nil {
		t.Fatalf("render should still apply via literal-text fallback, got ok=%v err=%v", ok, err)
	}
	var decoded strings.Builder
	for _, id := range tokens {
		decoded.WriteString(r.tok.DecodeToken(id))
	}
	if got := decoded.String(); !strings.Contains(got, "[INST]") || !strings.Contains(got, "[/INST]") {
		t.Fatalf("decoded output %q missing literal [INST]/[/INST] brackets", got)
	}
}

func TestChatTemplateKindDetectsMistralInst(t *testing.T) {
	tok := newInstTestTokenizer()

	// Detected from an embedded jinja chat template.
	rTmpl := &Runner{
		tok: tok,
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "string", Value: "{{bos_token}}[INST]{{content}}[/INST]"},
		}},
	}
	if kind := rTmpl.chatTemplateKind(); kind != "mistral-inst" {
		t.Fatalf("chatTemplateKind (template) = %q, want mistral-inst", kind)
	}

	// Detected from control tokens when no chat template is present.
	rTok := &Runner{tok: tok, gguf: &GGUFFile{Metadata: map[string]MetaValue{}}}
	if kind := rTok.chatTemplateKind(); kind != "mistral-inst" {
		t.Fatalf("chatTemplateKind (tokens) = %q, want mistral-inst", kind)
	}
}

// SauerkrautLM-SOLAR-Instruct — and every other SOLAR derivative — ships
// Upstage's "### Role:" template, which uses no special tokens at all. Before
// this was detected it fell through to renderPlainMessages, whose
// "User: "/"Assistant: " markers are similar enough to look right in a diff
// and wrong enough to be out of distribution for the model.
func TestChatTemplateKindDetectsAlpaca(t *testing.T) {
	// The verbatim chat_template from upstage/SOLAR-10.7B-Instruct-v1.0.
	const solarTemplate = "{% for message in messages %}{% if message['role'] == 'system' %}" +
		"{% if message['content']%}{{'### System:\n' + message['content']+'\n\n'}}{% endif %}" +
		"{% elif message['role'] == 'user' %}{{'### User:\n' + message['content']+'\n\n'}}" +
		"{% elif message['role'] == 'assistant' %}{{'### Assistant:\n'  + message['content']}}{% endif %}" +
		"{% if loop.last and add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}{% endfor %}"

	r := &Runner{
		tok: newInstTestTokenizer(),
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "string", Value: solarTemplate},
		}},
	}
	if kind := r.chatTemplateKind(); kind != "alpaca-chat" {
		t.Fatalf("chatTemplateKind = %q, want alpaca-chat", kind)
	}
}

// ChatML also contains the word "Assistant" in some templates; make sure the
// more specific kinds still win over the new case.
func TestChatTemplateKindAlpacaDoesNotShadowChatML(t *testing.T) {
	r := &Runner{
		tok: newInstTestTokenizer(),
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "string", Value: "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>{% endfor %}"},
		}},
	}
	if kind := r.chatTemplateKind(); kind != "chatml" {
		t.Fatalf("chatTemplateKind = %q, want chatml", kind)
	}
}

func TestAlpacaPromptMatchesSOLARTemplate(t *testing.T) {
	got := alpacaPrompt([]ChatMessage{
		UserMessage("Wie geht es dir?"),
		AssistantMessage("Gut, danke."),
		UserMessage("Und jetzt?"),
	}, "Du sprichst Deutsch.")

	// System and user turns are followed by a blank line; an assistant turn is
	// not — that asymmetry is in Upstage's own template and models trained on
	// it are sensitive to the spacing.
	want := "### System:\nDu sprichst Deutsch.\n\n" +
		"### User:\nWie geht es dir?\n\n" +
		"### Assistant:\nGut, danke." +
		"### User:\nUnd jetzt?\n\n" +
		"### Assistant:\n"
	if got != want {
		t.Fatalf("alpacaPrompt =\n%q\nwant\n%q", got, want)
	}
}

// An explicit system message must not be duplicated by the systemPrompt
// argument, and an empty system turn emits no bare header.
func TestAlpacaPromptSystemHandling(t *testing.T) {
	got := alpacaPrompt([]ChatMessage{
		{Role: ChatRoleSystem, Content: "Sei praezise."},
		UserMessage("hi"),
	}, "Diese soll ignoriert werden.")
	want := "### System:\nSei praezise.\n\n### User:\nhi\n\n### Assistant:\n"
	if got != want {
		t.Fatalf("explicit system: alpacaPrompt = %q, want %q", got, want)
	}

	got = alpacaPrompt([]ChatMessage{
		{Role: ChatRoleSystem, Content: "   "},
		UserMessage("hi"),
	}, "")
	want = "### User:\nhi\n\n### Assistant:\n"
	if got != want {
		t.Fatalf("empty system: alpacaPrompt = %q, want %q", got, want)
	}
}
