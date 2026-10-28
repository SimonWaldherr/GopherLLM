package engine

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestMistralRenderMarkerLayoutsPreserveToolPayloadAndContinuation(t *testing.T) {
	for _, variant := range []string{"native", "legacy", "literal", "zero-id"} {
		t.Run(variant, func(t *testing.T) {
			tok := newMistralPrefixCacheTokenizer()
			switch variant {
			case "legacy":
				tok = newMistralToolTokenizer()
			case "literal":
				tok = newInstTestTokenizer()
				delete(tok.TokenToID, "[INST]")
				delete(tok.TokenToID, "[/INST]")
			case "zero-id":
				tok.TokenToID["[INST]"] = 0
			}
			r := &Runner{tok: tok, arch: "ministral"}
			const args = ` {"literal": "a  b", "count": 2} `
			calls := []ToolCall{
				{Function: ToolCallFunction{Name: "lookup", Arguments: args}},
				{Function: ToolCallFunction{Name: "empty"}},
			}
			messages := []ChatMessage{
				{Role: ChatRoleSystem, Content: "first"}, UserMessage("hi"),
				{Role: ChatRoleAssistant, Content: " answer ", ToolCalls: calls},
				ToolResultMessage("call", "lookup", "  result  "), UserMessage("bye"),
				AssistantMessage(" seed "), {Role: ChatRoleSystem, Content: "second"},
			}
			tools := []ToolDefinition{sampleTool()}
			var want []uint32
			marker := func(literal string) {
				if id, ok := tok.SpecialID(literal); ok {
					want = append(want, id)
				} else {
					want = append(want, tok.EncodeWithoutBOS(literal)...)
				}
			}
			text := func(s string) { want = append(want, tok.EncodeWithoutBOS(s)...) }
			if tok.AddBOS {
				want = append(want, tok.BOSID)
			}
			_, _, nativeSystem := r.systemPromptTokens()
			if nativeSystem {
				marker("[SYSTEM_PROMPT]")
				text("second")
				marker("[/SYSTEM_PROMPT]")
			}
			toolsJSON, err := json.Marshal(tools)
			if err != nil {
				t.Fatal(err)
			}
			marker("[AVAILABLE_TOOLS]")
			text(string(toolsJSON))
			marker("[/AVAILABLE_TOOLS]")
			marker("[INST]")
			text("hi")
			marker("[/INST]")
			text("answer")
			for i, call := range calls {
				marker("[TOOL_CALLS]")
				text(call.Function.Name)
				marker("[ARGS]")
				if i == 0 {
					text(args)
				} else {
					text("{}")
				}
			}
			want = append(want, tok.EOSID)
			marker("[TOOL_RESULTS]")
			text("  result  ")
			marker("[/TOOL_RESULTS]")
			marker("[INST]")
			if nativeSystem {
				text("bye")
			} else {
				text("second\n\nbye")
			}
			marker("[/INST]")
			text("seed") // A final assistant continuation must remain open.
			got, images, ok, err := r.renderMistralInstMessages(messages, "default", tools)
			if err != nil || !ok || images != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("rendered marker layout differs: got=%v want=%v images=%v ok=%v err=%v", got, want, images, ok, err)
			}
		})
	}
}

func TestMistralPromptPrefixHitOwnsDynamicCapacity(t *testing.T) {
	r := newMistralPrefixCacheRunner()
	r.EnableMistralPromptPrefixCache(0)
	system := strings.Repeat("system instruction ", 256)
	messages := []ChatMessage{UserMessage("first")}
	if _, _, _, err := r.renderMistralInstMessages(messages, system, nil); err != nil {
		t.Fatal(err)
	}
	before := r.MistralPromptPrefixCacheStats()
	messages[0].Content = strings.Repeat("new content ", 64)
	got, _, _, err := r.renderMistralInstMessages(messages, system, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, _, _, err := newMistralPrefixCacheRunner().renderMistralInstMessages(messages, system, nil)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("cached dynamic prompt differs from cold rendering")
	}
	// Neither dynamic appends nor subsequent caller edits may touch the
	// resident immutable prefix, even when they exhaust the reserved capacity.
	for i := range got {
		got[i] = ^uint32(0)
	}
	again, _, _, err := r.renderMistralInstMessages(messages, system, nil)
	if err != nil || !reflect.DeepEqual(again, want) {
		t.Fatal("caller mutation corrupted the cached prefix")
	}
	after := r.MistralPromptPrefixCacheStats()
	if after.Hits != before.Hits+2 || after.Bytes != before.Bytes || after.Entries != before.Entries {
		t.Fatalf("hit changed resident prefix size: before=%+v after=%+v", before, after)
	}
}

func BenchmarkMistralRenderControlMarkers(b *testing.B) {
	for _, shape := range []string{"native-short", "legacy-short", "native-32turns"} {
		b.Run(shape, func(b *testing.B) {
			r := newMistralPrefixCacheRunner()
			if shape == "legacy-short" {
				r.tok = newInstTestTokenizer()
			}
			messages := []ChatMessage{UserMessage("hello")}
			if shape == "native-32turns" {
				for range 32 {
					messages = append(messages, AssistantMessage("answer"), UserMessage("hello"))
				}
			}
			b.ReportAllocs()
			for b.Loop() {
				if _, _, ok, err := r.renderMistralInstMessages(messages, "", nil); err != nil || !ok {
					b.Fatal(err)
				}
			}
		})
	}
}
