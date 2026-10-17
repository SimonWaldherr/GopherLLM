package gopherllm

import (
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
)

// Preserve the previous helper as a controlled benchmark reference. The
// default-disabled KV cache used to render this prefix and then discard it.
func mistralStaticPromptPrefixBeforeFastpath(r *Runner, messages []ChatMessage, system string, tools []ToolDefinition, full []uint32) []uint32 {
	if r == nil || r.kind != loadedStandard || r.chatTemplateKind() != "mistral-inst" || len(full) <= 1 {
		return nil
	}
	var systems []ChatMessage
	for _, m := range messages {
		if m.Role == ChatRoleSystem {
			systems = append(systems, m)
		}
	}
	prefix, _, ok, err := r.renderMistralInstMessages(systems, system, tools)
	if err != nil || !ok || len(prefix) <= 1 || len(prefix) >= len(full) || sharedTokenPrefix(prefix, full) != len(prefix) {
		return nil
	}
	return prefix
}

func TestMistralDisabledKVCacheDoesNotRenderStaticPrefix(t *testing.T) {
	r := newMistralPrefixCacheRunner()
	r.EnableMistralPromptPrefixCache(0)
	messages := []ChatMessage{{Role: ChatRoleSystem, Content: "Use the latest exact instructions."}, UserMessage("Question?")}
	tools := []ToolDefinition{sampleTool()}
	full := r.renderMessages(messages, "Fallback instructions.", tools)
	before := r.MistralPromptPrefixCacheStats()
	if prefix := r.mistralStaticPromptPrefix(messages, "Fallback instructions.", tools, full); prefix != nil {
		t.Fatalf("disabled snapshot cache produced an unused prefix: %v", prefix)
	}
	if after := r.MistralPromptPrefixCacheStats(); after != before {
		t.Fatalf("disabled KV cache performed another render: before=%+v after=%+v", before, after)
	}
	r.EnableMistralKVPrefixCache(0)
	prefix := r.mistralStaticPromptPrefix(messages, "Fallback instructions.", tools, full)
	if len(prefix) <= 1 || sharedTokenPrefix(prefix, full) != len(prefix) {
		t.Fatal("enabling snapshots did not restore the exact static prefix")
	}
	if after := r.MistralPromptPrefixCacheStats(); after.Hits != before.Hits+1 {
		t.Fatalf("enabled snapshot cache did not consult static rendering: %+v", after)
	}
	r.DisableMistralKVPrefixCache()
	before = r.MistralPromptPrefixCacheStats()
	if prefix := r.mistralStaticPromptPrefix(messages, "Fallback instructions.", tools, full); prefix != nil {
		t.Fatal("disabled cache retained an active helper path")
	}
	if after := r.MistralPromptPrefixCacheStats(); after != before {
		t.Fatal("disabling snapshots left duplicate rendering enabled")
	}
}

func TestMistralStaticPrefixFastpathKeepsEnabledTokensExact(t *testing.T) {
	for _, tc := range []struct {
		name     string
		messages []ChatMessage
		tools    []ToolDefinition
		legacy   bool
	}{
		{"default system", []ChatMessage{UserMessage("hello")}, nil, false},
		{"explicit system and tools", []ChatMessage{{Role: ChatRoleSystem, Content: "First system."}, {Role: ChatRoleSystem, Content: "Last system."}, UserMessage("hello")}, []ToolDefinition{sampleTool()}, false},
		{"late system", []ChatMessage{UserMessage("earlier"), AssistantMessage("answer"), {Role: ChatRoleSystem, Content: "Late override."}, UserMessage("latest")}, []ToolDefinition{sampleTool()}, false},
		{"legacy folded system", []ChatMessage{UserMessage("hello")}, nil, true},
		{"legacy tools", []ChatMessage{UserMessage("hello")}, []ToolDefinition{sampleTool()}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newMistralPrefixCacheRunner()
			if tc.legacy {
				r.tok = newMistralToolTokenizer()
			}
			r.EnableMistralKVPrefixCache(0)
			full := r.renderMessages(tc.messages, "Default exact system.", tc.tools)
			want := mistralStaticPromptPrefixBeforeFastpath(r, tc.messages, "Default exact system.", tc.tools, full)
			got := r.mistralStaticPromptPrefix(tc.messages, "Default exact system.", tc.tools, full)
			if !slices.Equal(got, want) {
				t.Fatalf("enabled prefix changed: got=%v want=%v", got, want)
			}
		})
	}
}

func TestMistralDisabledStaticPrefixGenerationParity(t *testing.T) {
	disabled, enabled := newTinyMistralKVPrefixRunner(t), newTinyMistralKVPrefixRunner(t)
	defer disabled.Close()
	defer enabled.Close()
	disabled.EnableMistralPromptPrefixCache(0)
	enabled.EnableMistralPromptPrefixCache(0)
	enabled.EnableMistralKVPrefixCache(0)
	messages := []ChatMessage{{Role: ChatRoleSystem, Content: "a b"}, UserMessage("c")}
	options := tinyMistralKVPrefixOptions()
	got, err := disabled.GenerateChat(messages, options)
	if err != nil {
		t.Fatal(err)
	}
	want, err := enabled.GenerateChat(messages, options)
	if err != nil {
		t.Fatal(err)
	}
	if got.Text != want.Text || got.FinishReason != want.FinishReason || got.Stats.PromptTokens != want.Stats.PromptTokens ||
		!slices.Equal(disabled.prefixCache.tokens, enabled.prefixCache.tokens) || !slices.Equal(disabled.prefixCache.promptLogits, enabled.prefixCache.promptLogits) {
		t.Fatal("skipping an unused static render changed generation or cached token state")
	}
	if stats := disabled.MistralPromptPrefixCacheStats(); stats.Hits != 0 || stats.Misses != 1 {
		t.Fatalf("default generation rendered its static prefix twice: %+v", stats)
	}
	if stats := enabled.MistralPromptPrefixCacheStats(); stats.Hits != 1 || stats.Misses != 1 {
		t.Fatalf("enabled snapshots did not retain their original render behavior: %+v", stats)
	}
}

func TestMistralStaticPrefixFastpathConcurrentCacheControls(t *testing.T) {
	r := newMistralPrefixCacheRunner()
	r.EnableMistralPromptPrefixCache(0)
	messages := []ChatMessage{{Role: ChatRoleSystem, Content: "Stable exact instructions."}, UserMessage("question")}
	tools := []ToolDefinition{sampleTool()}
	full := r.renderMessages(messages, "", tools)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 100 {
			r.EnableMistralKVPrefixCache(0)
			r.ClearMistralKVPrefixCache()
			r.DisableMistralKVPrefixCache()
		}
	}()
	for range 100 {
		prefix := r.mistralStaticPromptPrefix(messages, "", tools, full)
		if prefix != nil && (len(prefix) <= 1 || sharedTokenPrefix(prefix, full) != len(prefix)) {
			t.Error("cache control race returned a different token prefix")
		}
	}
	wg.Wait()
	r.EnableMistralKVPrefixCache(0)
	if prefix := r.mistralStaticPromptPrefix(messages, "", tools, full); len(prefix) <= 1 {
		t.Fatal("cache control race left enabled rendering unavailable")
	}
}

func benchmarkMistralDisabledStaticPrefix(b *testing.B, newRunner func(testing.TB) *Runner) {
	for _, long := range []bool{false, true} {
		for _, renderCache := range []bool{false, true} {
			for _, fast := range []bool{false, true} {
				b.Run(fmt.Sprintf("long=%t/rendercache=%t/fast=%t", long, renderCache, fast), func(b *testing.B) {
					r := newRunner(b)
					if r.chatTemplateKind() != "mistral-inst" {
						b.Skip("requires a Mistral-family tokenizer/template")
					}
					if renderCache {
						r.EnableMistralPromptPrefixCache(0)
					}
					tool := sampleTool()
					system := "Answer concisely and use verified facts."
					if long {
						system = strings.Repeat(system+"\n", 256)
						tool.Function.Description = strings.Repeat("Search the local read-only knowledge base. ", 96)
					}
					messages := []ChatMessage{{Role: ChatRoleSystem, Content: system}, UserMessage("Find the relevant facts.")}
					tools := []ToolDefinition{tool}
					full := r.renderMessages(messages, "", tools)
					before := append([]uint32(nil), full...)
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						if fast {
							if prefix := r.mistralStaticPromptPrefix(messages, "", tools, full); prefix != nil {
								b.Fatal("disabled KV cache returned an unused prefix")
							}
						} else {
							_ = mistralStaticPromptPrefixBeforeFastpath(r, messages, "", tools, full)
						}
					}
					if !reflect.DeepEqual(full, before) {
						b.Fatal("benchmark changed input tokens")
					}
				})
			}
		}
	}
}

func BenchmarkMistralDisabledStaticPrefix(b *testing.B) {
	benchmarkMistralDisabledStaticPrefix(b, func(testing.TB) *Runner { return newMistralPrefixCacheRunner() })
}

func BenchmarkMistralDisabledStaticPrefixConfiguredTokenizer(b *testing.B) {
	path := os.Getenv("GOPHERLLM_BENCH_MODEL")
	if path == "" {
		b.Skip("set GOPHERLLM_BENCH_MODEL to a local Mistral GGUF; only tokenizer metadata is loaded")
	}
	mapping, err := OpenMmap(path)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = mapping.Close() })
	gguf, err := ParseGGUF(mapping.Bytes())
	if err != nil {
		b.Fatal(err)
	}
	tokenizer, err := TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		b.Fatal(err)
	}
	arch, _ := gguf.GetString("general.architecture")
	benchmarkMistralDisabledStaticPrefix(b, func(testing.TB) *Runner {
		return &Runner{gguf: gguf, arch: arch, tok: tokenizer, kind: loadedStandard}
	})
}
