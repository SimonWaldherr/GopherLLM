package gopherllm

import "testing"

func TestRenderCodestralFIMOrdersPrefixSuffixMiddle(t *testing.T) {
	tok := newChatTokenizer("[PREFIX]", "[SUFFIX]", "[MIDDLE]")
	r := &Runner{tok: tok, arch: "mistral"}

	tokens, err := r.renderCodestralFIM("def add(a, b):\n    ", "\n    return result\n")
	if err != nil {
		t.Fatalf("renderCodestralFIM: %v", err)
	}
	prefixTok, suffixTok, middleTok := tok.TokenToID["[PREFIX]"], tok.TokenToID["[SUFFIX]"], tok.TokenToID["[MIDDLE]"]
	if tokens[0] != tok.BOSID {
		t.Fatalf("expected leading BOS, got %v", tokens[0])
	}
	if tokens[1] != prefixTok {
		t.Fatalf("expected [PREFIX] right after BOS, got %v", tokens[1])
	}
	posSuffix := indexOfToken(tokens, suffixTok)
	posMiddle := indexOfToken(tokens, middleTok)
	if posSuffix < 2 || posMiddle < 0 || posMiddle != len(tokens)-1 {
		t.Fatalf("expected [PREFIX] prefix... [SUFFIX] suffix... [MIDDLE] (as the final token), got positions suffix=%d middle=%d len=%d", posSuffix, posMiddle, len(tokens))
	}
	if posMiddle <= posSuffix {
		t.Fatalf("[MIDDLE] must come after [SUFFIX], got suffix=%d middle=%d", posSuffix, posMiddle)
	}
}

func TestRenderCodestralFIMErrorsWithoutInfillTokens(t *testing.T) {
	tok := newInstTestTokenizer() // no [PREFIX]/[SUFFIX]/[MIDDLE]
	r := &Runner{tok: tok, arch: "mistral"}
	if _, err := r.renderCodestralFIM("a", "b"); err == nil {
		t.Fatal("expected an error for a vocabulary without FIM control tokens")
	}
}

// FIMSuffix is the dispatch switch renderMessagesForGeneration uses to divert
// to the FIM renderer entirely, bypassing chat templating. It must reject
// shapes that are not "exactly one plain user message" rather than silently
// rendering something a caller did not ask for.
func TestRenderMessagesForGenerationDispatchesFIM(t *testing.T) {
	tok := newChatTokenizer("[PREFIX]", "[SUFFIX]", "[MIDDLE]", "[INST]", "[/INST]")
	r := &Runner{tok: tok, arch: "mistral"}

	tokens, embeds, err := r.renderMessagesForGeneration([]ChatMessage{UserMessage("prefix text")}, "", nil, "suffix text")
	if err != nil {
		t.Fatalf("renderMessagesForGeneration (FIM): %v", err)
	}
	if embeds != nil {
		t.Fatalf("FIM has no image embeddings, got %v", embeds)
	}
	if countToken(tokens, tok.TokenToID["[PREFIX]"]) != 1 || countToken(tokens, tok.TokenToID["[SUFFIX]"]) != 1 || countToken(tokens, tok.TokenToID["[MIDDLE]"]) != 1 {
		t.Fatalf("expected exactly one of each FIM control token, got %v", tokens)
	}
	if countToken(tokens, tok.TokenToID["[INST]"]) != 0 {
		t.Fatalf("FIM must not apply the chat template, got %v", tokens)
	}

	if _, _, err := r.renderMessagesForGeneration([]ChatMessage{UserMessage("a"), UserMessage("b")}, "", nil, "suffix"); err == nil {
		t.Fatal("expected an error for more than one message under FIMSuffix")
	}
	if _, _, err := r.renderMessagesForGeneration([]ChatMessage{AssistantMessage("a")}, "", nil, "suffix"); err == nil {
		t.Fatal("expected an error for a non-user message under FIMSuffix")
	}
}
