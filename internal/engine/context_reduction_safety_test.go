package engine

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestPrepareChatContextAutoCompressPreservesFittingOriginal(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := autoCompressContextWindowOptions()
	messages := []ChatMessage{
		{Role: ChatRoleSystem, Content: "Always spell out application programming interface."},
		UserMessage("Translate exactly: large language model"),
		AssistantMessage("Please explain the application programming interface, for example its context window."),
		UserMessage("Preserve the exact spelling and whitespace."),
	}
	r.config.MaxSeqLen = contextWindowLimitFor(r, messages, opts)
	got, info, err := r.PrepareChatContext(messages, opts)
	if err != nil || !reflect.DeepEqual(got, messages) || info.CompressedMessages != 0 || info.DroppedMessages != 0 {
		t.Fatalf("fitting original changed: messages=%#v info=%+v err=%v", got, info, err)
	}
}

func TestPrepareChatContextCountsOnlyRetainedCompression(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := autoCompressContextWindowOptions()
	messages := []ChatMessage{
		UserMessage(strings.Repeat("a", 256)),
		AssistantMessage("A large language model uses an application programming interface."),
		UserMessage("dddd"),
	}
	r.config.MaxSeqLen = contextWindowLimitFor(r, messages[2:], opts)
	got, info, err := r.PrepareChatContext(messages, opts)
	if err != nil || !reflect.DeepEqual(got, messages[2:]) || info.CompressedMessages != 0 {
		t.Fatalf("discarded compression counted in selected prompt: got=%#v info=%+v err=%v", got, info, err)
	}
}

func TestPrepareChatContextCompressionNeverDropsAnOriginalThatFits(t *testing.T) {
	vocab := []string{"[UNK]", "beziehungsweise", "b", "##z", "##w", ".", "application", "programming", "interface", "API"}
	tokenToID := make(map[string]uint32, len(vocab))
	for i, token := range vocab {
		tokenToID[token] = uint32(i)
	}
	r := &Runner{tok: &Tokenizer{Vocab: vocab, TokenToID: tokenToID, Mode: TokenizerWordPiece, RawWordPieceVocab: true}}
	opts := autoCompressContextWindowOptions()
	messages := []ChatMessage{
		UserMessage(strings.Repeat("old ", 256)),
		AssistantMessage(strings.TrimSpace(strings.Repeat("application programming interface ", 32))),
		UserMessage("previous"),
		AssistantMessage("beziehungsweise"),
		UserMessage("now"),
	}
	compressed, changed := autoCompressChatMessages(messages)
	if changed != 2 || len(r.renderMessages(compressed, "", nil)) >= len(r.renderMessages(messages, "", nil)) {
		t.Fatal("fixture needs overall token savings from two changed assistant messages")
	}
	want := messages[2:]
	if len(r.renderMessages(compressed[2:], "", nil)) <= len(r.renderMessages(want, "", nil)) {
		t.Fatal("fixture needs a compressed candidate larger than its original")
	}
	r.config.MaxSeqLen = contextWindowLimitFor(r, want, opts)
	var tokens []uint32
	got, info, err := r.prepareChatContext(messages, opts, &tokens)
	if err != nil || !reflect.DeepEqual(got, want) || info.CompressedMessages != 0 {
		t.Fatalf("token-increasing abbreviation dropped a fitting original: got=%#v info=%+v err=%v", got, info, err)
	}
	if !reflect.DeepEqual(tokens, r.renderMessages(want, "", nil)) || len(tokens) != info.PromptTokens {
		t.Fatal("handoff did not preserve the selected original candidate")
	}
}

func TestAutoCompressProtectsInstructionsLatestTurnAndOpaqueData(t *testing.T) {
	for _, opaque := range []ChatMessage{
		AssistantMessage("```text\nlarge language model\n```"),
		AssistantMessage("~~~text\nlarge language model\n~~~"),
		AssistantMessage("first line\n    large language model"),
		AssistantMessage("a\tlarge language model"),
		AssistantMessage("  large language model"),
		AssistantMessage("large  language model"),
		AssistantMessage("large\u00a0language model"),
		AssistantMessage("`large language model`"),
		AssistantMessage(`{"name":"large language model"}`),
		AssistantMessage("Translate exactly: large language model"),
		AssistantMessage("The literal string is 'large language model'."),
		AssistantMessage("* large language model"),
		AssistantMessage("- large language model"),
		AssistantMessage("1. large language model"),
		{Role: ChatRoleAssistant, Content: "large language model", Images: []ImageContent{{Bytes: []byte("image payload")}}},
		{Role: ChatRoleAssistant, Content: "large language model", ToolCalls: []ToolCall{{ID: "call", Function: ToolCallFunction{Arguments: `{"literal":"large language model"}`}}}},
		ToolResultMessage("call", "lookup", "large language model"),
	} {
		messages := []ChatMessage{
			{Role: ChatRoleSystem, Content: "Please preserve large language model exactly."},
			UserMessage("Translate exactly: large language model"),
			opaque,
			AssistantMessage("A large language model has a context window."),
			UserMessage("Translate exactly: large language model"),
			AssistantMessage("The latest turn includes this application programming interface."),
		}
		original := append([]ChatMessage(nil), messages...)
		got, changed := autoCompressChatMessages(messages)
		if changed != 1 || got[3].Content != "A LLM has a ctx window." {
			t.Fatalf("ordinary historical prose was not compressed: changed=%d result=%#v", changed, got)
		}
		for _, i := range []int{0, 1, 2, 4, 5} {
			if !reflect.DeepEqual(got[i], original[i]) {
				t.Fatalf("protected message %d was changed: got=%#v want=%#v", i, got[i], original[i])
			}
		}
		if !reflect.DeepEqual(messages, original) {
			t.Fatal("compression mutated caller-owned history")
		}
	}
}

func TestPrepareChatContextRejectsBoundedFIMAndOverflow(t *testing.T) {
	r := newContextWindowTestRunner(t)
	for _, mode := range []ContextWindowMode{ContextWindowRecent, ContextWindowAutoCompress} {
		opts := recentContextWindowOptions()
		opts.ContextWindowMode, opts.FIMSuffix = mode, strings.Repeat("a", 256)
		stale := []uint32{123}
		if _, _, err := r.prepareChatContext([]ChatMessage{UserMessage("abc")}, opts, &stale); err == nil || !strings.Contains(err.Error(), "FIM") || stale != nil {
			t.Fatalf("bounded FIM: error=%v handed tokens=%v", err, stale)
		}
		opts.FIMSuffix, opts.MaxTokens = "", int(^uint(0)>>1)
		stale = []uint32{123}
		if _, _, err := r.prepareChatContext([]ChatMessage{UserMessage("abc")}, opts, &stale); err == nil || !strings.Contains(err.Error(), "reserve") || stale != nil {
			t.Fatalf("completion overflow: error=%v handed tokens=%v", err, stale)
		}
	}
}

func TestPrepareChatContextRestoresPinnedPrefixAcrossCandidates(t *testing.T) {
	r := newContextWindowTestRunner(t)
	opts := recentContextWindowOptions()
	messages := []ChatMessage{
		{Role: ChatRoleSystem, Content: "a"}, {Role: ChatRoleSystem, Content: "b"},
		{Role: ChatRoleSystem, Content: "c"}, {Role: ChatRoleSystem, Content: "d"},
		{Role: ChatRoleSystem, Content: "e"},
	}
	for i := range 8 {
		messages = append(messages, UserMessage(strings.Repeat("f", 8+i)))
	}
	original := append([]ChatMessage(nil), messages...)
	want := append(append([]ChatMessage(nil), messages[:5]...), messages[len(messages)-3:]...)
	r.config.MaxSeqLen = contextWindowLimitFor(r, want, opts)
	var tokens []uint32
	got, info, err := r.prepareChatContext(messages, opts, &tokens)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("pinned candidate: got=%#v info=%+v err=%v", got, info, err)
	}
	if !reflect.DeepEqual(messages, original) {
		t.Fatal("suffix construction mutated caller-owned history")
	}
	if !reflect.DeepEqual(tokens, r.renderMessages(got, opts.SystemPrompt, opts.ActiveTools())) || len(tokens) != info.PromptTokens {
		t.Fatal("handed tokens do not represent the restored final candidate")
	}
}

func TestPrepareChatContextFullFitDoesNotAssumeMonotonicTemplate(t *testing.T) {
	for _, systemBytes := range []int{128, 100000} {
		t.Run(fmt.Sprint(systemBytes), func(t *testing.T) {
			r := newMistralPrefixCacheRunner()
			opts := recentContextWindowOptions()
			messages := []ChatMessage{
				{Role: ChatRoleSystem, Content: strings.Repeat("a", systemBytes)},
				UserMessage("old"),
				{Role: ChatRoleSystem, Content: "c"},
				UserMessage("new"),
			}
			r.config.MaxSeqLen = contextWindowLimitFor(r, messages, opts)
			latest := []ChatMessage{messages[0], messages[3]}
			if len(r.renderMessages(latest, "", nil)) <= r.config.MaxSeqLen-opts.MaxTokens-contextWindowSafetyTokens {
				t.Fatal("test requires a latest-turn rendering larger than the complete history")
			}
			got, info, err := r.PrepareChatContext(messages, opts)
			if err != nil || !reflect.DeepEqual(got, messages) || info.DroppedMessages != 0 {
				t.Fatalf("fitting nonmonotonic history was lost: got=%#v info=%+v err=%v", got, info, err)
			}
		})
	}
}

func TestChatContextTurnViewsKeepPreamblesAndToolGroups(t *testing.T) {
	for _, tc := range []struct {
		roles       []ChatRole
		pinned      int
		turnLengths []int
		latest      int
	}{
		{nil, 0, nil, -1},
		{[]ChatRole{ChatRoleSystem, ChatRoleSystem}, 2, nil, -1},
		{[]ChatRole{ChatRoleAssistant, ChatRoleTool}, 0, []int{2}, -1},
		{[]ChatRole{ChatRoleUser, ChatRoleUser, ChatRoleUser}, 0, []int{1, 1, 1}, 2},
		{[]ChatRole{ChatRoleSystem, ChatRoleSystem, ChatRoleAssistant, ChatRoleTool, ChatRoleUser, ChatRoleAssistant, ChatRoleTool, ChatRoleSystem, ChatRoleUser, ChatRoleTool}, 2, []int{2, 4, 2}, 2},
	} {
		messages := make([]ChatMessage, len(tc.roles))
		for i, role := range tc.roles {
			messages[i] = ChatMessage{Role: role, Content: strings.Repeat("a", i+1)}
		}
		pinned, turns, latest := chatContextTurns(messages)
		if len(pinned) != tc.pinned || len(turns) != len(tc.turnLengths) || latest != tc.latest {
			t.Fatalf("roles=%v: pinned=%d turns=%d latest=%d", tc.roles, len(pinned), len(turns), latest)
		}
		for i, n := range tc.turnLengths {
			if len(turns[i]) != n {
				t.Fatalf("roles=%v turn %d length=%d want=%d", tc.roles, i, len(turns[i]), n)
			}
		}
		if joined := joinChatContextTurns(pinned, turns); len(joined) != 0 && !reflect.DeepEqual(joined, messages) {
			t.Fatalf("turn boundaries lost original messages: got=%#v want=%#v", joined, messages)
		}
	}
}

func TestAutoCompressMessagePreservesWhitespaceNormalization(t *testing.T) {
	for _, text := range []string{
		"", "abc", "a b", " a b ", "a  b", "a\tb", "a\nb", "a\vb", "a\fb", "a\rb",
		"a\u0085b", "a\u00a0b", "a\u2003b", "a\u2028b", "a\u3000b", "a\xffb",
	} {
		want := strings.Join(strings.Fields(text), " ")
		if got := autoCompressMessage(text); got != want {
			t.Fatalf("normalization for %q: got=%q want=%q", text, got, want)
		}
	}
}
