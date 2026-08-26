package gopherllm

import (
	"reflect"
	"strings"
	"sync"
	"testing"
)

func newMistralPrefixCacheTokenizer() *Tokenizer {
	return newChatTokenizer(
		"[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]",
		"[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]",
		"[TOOL_CALLS]", "[ARGS]", "[TOOL_RESULTS]", "[/TOOL_RESULTS]",
	)
}

func newMistralPrefixCacheRunner() *Runner {
	return &Runner{
		tok:    newMistralPrefixCacheTokenizer(),
		arch:   "ministral",
		config: Config{MaxSeqLen: 4096},
	}
}

func TestMistralPromptPrefixCacheHitsStaticSystemAndTools(t *testing.T) {
	r := newMistralPrefixCacheRunner()
	r.EnableMistralPromptPrefixCache(0)
	system := "Always cite the provided weather observations."
	tools := []ToolDefinition{sampleTool()}
	firstMessages := []ChatMessage{UserMessage("weather in Berlin?")}

	first, _, ok, err := r.renderMistralInstMessages(firstMessages, system, tools)
	if err != nil || !ok {
		t.Fatalf("first render = ok=%v err=%v", ok, err)
	}
	stats := r.MistralPromptPrefixCacheStats()
	if !stats.Enabled || stats.Entries != 1 || stats.Hits != 0 || stats.Misses != 1 || stats.Bytes <= 0 {
		t.Fatalf("first cache stats = %+v", stats)
	}

	secondMessages := []ChatMessage{UserMessage("weather in Munich?")}
	got, _, ok, err := r.renderMistralInstMessages(secondMessages, system, tools)
	if err != nil || !ok {
		t.Fatalf("second render = ok=%v err=%v", ok, err)
	}
	stats = r.MistralPromptPrefixCacheStats()
	if stats.Entries != 1 || stats.Hits != 1 || stats.Misses != 1 {
		t.Fatalf("second cache stats = %+v, want one hit", stats)
	}

	// The cache must only change work, never rendering semantics. Compare a
	// different, uncached Runner so the answer cannot accidentally be derived
	// from the already-rendered first turn.
	cold := newMistralPrefixCacheRunner()
	want, _, ok, err := cold.renderMistralInstMessages(secondMessages, system, tools)
	if err != nil || !ok {
		t.Fatalf("cold render = ok=%v err=%v", ok, err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("cached render differs from cold:\n got %v\nwant %v", got, want)
	}
	if len(first) == 0 {
		t.Fatal("first render unexpectedly empty")
	}
}

func TestMistralPromptPrefixCacheLRUBoundsAndClear(t *testing.T) {
	r := newMistralPrefixCacheRunner()
	tools := []ToolDefinition(nil)
	messages := []ChatMessage{UserMessage("hello")}

	// Measure an entry first, then give the cache exactly enough room for two
	// same-sized static prefixes. Every system string below has three bytes and
	// tokenizes to the same number of test-tokenizer rows.
	r.EnableMistralPromptPrefixCache(0)
	if _, _, _, err := r.renderMistralInstMessages(messages, "one", tools); err != nil {
		t.Fatal(err)
	}
	entryBytes := r.MistralPromptPrefixCacheStats().Bytes
	if entryBytes <= 0 {
		t.Fatalf("entry bytes = %d", entryBytes)
	}

	r.EnableMistralPromptPrefixCache(entryBytes * 2)
	for _, system := range []string{"one", "two", "six"} {
		if _, _, _, err := r.renderMistralInstMessages(messages, system, tools); err != nil {
			t.Fatal(err)
		}
	}
	stats := r.MistralPromptPrefixCacheStats()
	if stats.Entries != 2 || stats.Bytes > stats.MaxBytes {
		t.Fatalf("bounded LRU stats = %+v", stats)
	}
	// "one" was evicted, while "two" remains. Touch it, insert a fourth
	// entry, then ensure the touched entry survives the next eviction.
	if _, _, _, err := r.renderMistralInstMessages(messages, "two", tools); err != nil {
		t.Fatal(err)
	}
	beforeTouch := r.MistralPromptPrefixCacheStats()
	if _, _, _, err := r.renderMistralInstMessages(messages, "red", tools); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := r.renderMistralInstMessages(messages, "two", tools); err != nil {
		t.Fatal(err)
	}
	stats = r.MistralPromptPrefixCacheStats()
	if stats.Hits != beforeTouch.Hits+1 || stats.Entries != 2 || stats.Bytes > stats.MaxBytes {
		t.Fatalf("LRU touch/eviction stats = %+v, before touch %+v", stats, beforeTouch)
	}

	r.ClearMistralPromptPrefixCache()
	stats = r.MistralPromptPrefixCacheStats()
	if !stats.Enabled || stats.Entries != 0 || stats.Bytes != 0 || stats.Hits != 0 || stats.Misses != 0 {
		t.Fatalf("clear stats = %+v", stats)
	}

	// A prefix larger than the configured budget is rendered normally but is
	// never retained, keeping the memory bound strict.
	r.EnableMistralPromptPrefixCache(1)
	if _, _, _, err := r.renderMistralInstMessages(messages, "one", tools); err != nil {
		t.Fatal(err)
	}
	stats = r.MistralPromptPrefixCacheStats()
	if stats.Entries != 0 || stats.Bytes != 0 || stats.Misses != 1 {
		t.Fatalf("oversized entry stats = %+v", stats)
	}

	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	stats = r.MistralPromptPrefixCacheStats()
	if stats.Enabled || stats.Entries != 0 || stats.Bytes != 0 {
		t.Fatalf("Close retained render cache: %+v", stats)
	}
}

func TestMistralPromptPrefixCacheDoesNotCacheLegacyFoldedSystem(t *testing.T) {
	// This vocabulary lacks dedicated system markers. The legacy Mistral
	// format appends the system text to the final user turn, so treating it as
	// a reusable static prefix would be incorrect.
	r := &Runner{tok: newMistralToolTokenizer(), arch: "ministral"}
	r.EnableMistralPromptPrefixCache(0)
	messages := []ChatMessage{UserMessage("question")}
	first, _, ok, err := r.renderMistralInstMessages(messages, "system one", nil)
	if err != nil || !ok {
		t.Fatalf("first legacy render = ok=%v err=%v", ok, err)
	}
	second, _, ok, err := r.renderMistralInstMessages(messages, "system two", nil)
	if err != nil || !ok {
		t.Fatalf("second legacy render = ok=%v err=%v", ok, err)
	}
	if reflect.DeepEqual(first, second) {
		t.Fatalf("different folded system prompts rendered identically: %v", first)
	}
	stats := r.MistralPromptPrefixCacheStats()
	if stats.Entries != 0 || stats.Hits != 0 || stats.Misses != 0 {
		t.Fatalf("legacy folded system was cached: %+v", stats)
	}
}

func TestMistralPromptPrefixCacheClearRejectsInFlightMiss(t *testing.T) {
	r := newMistralPrefixCacheRunner()
	r.EnableMistralPromptPrefixCache(0)
	key := mistralPromptPrefixCacheKey{system: "stable system"}
	if _, epoch, hit := r.appendMistralPromptPrefix(nil, key); hit || epoch == 0 {
		t.Fatalf("initial cache lookup = hit=%v epoch=%d, want cold enabled lookup", hit, epoch)
	} else {
		r.ClearMistralPromptPrefixCache()
		// This mirrors a renderer which began tokenizing before Clear and
		// completed after it. Its stale result must not cross the explicit
		// invalidation boundary.
		r.putMistralPromptPrefix(key, []uint32{1, 2, 3}, epoch)
	}
	stats := r.MistralPromptPrefixCacheStats()
	if !stats.Enabled || stats.Entries != 0 || stats.Bytes != 0 {
		t.Fatalf("stale in-flight prefix repopulated cleared cache: %+v", stats)
	}
}

func TestMistralPromptPrefixCacheConcurrentPrepareChatContext(t *testing.T) {
	r := newMistralPrefixCacheRunner()
	r.EnableMistralPromptPrefixCache(0)
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 8
	opts.SystemPrompt = "Follow the tools and answer in one short sentence."
	opts.Tools = []ToolDefinition{sampleTool()}
	opts.ContextWindowMode = ContextWindowRecent
	messages := []ChatMessage{
		UserMessage("weather in Berlin?"),
		AssistantMessage("I will check."),
		UserMessage("now compare Munich"),
	}

	const goroutines, iterations = 12, 20
	errCh := make(chan error, goroutines)
	var wg sync.WaitGroup
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range iterations {
				kept, info, err := r.PrepareChatContext(messages, opts)
				if err != nil {
					errCh <- err
					return
				}
				if len(kept) == 0 || info.PromptTokens == 0 {
					errCh <- errEmptyMistralPrefixCacheContext
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
	stats := r.MistralPromptPrefixCacheStats()
	if stats.Entries != 1 || stats.Hits == 0 || stats.Bytes > stats.MaxBytes {
		t.Fatalf("concurrent cache stats = %+v", stats)
	}
}

var errEmptyMistralPrefixCacheContext = &mistralPrefixCacheTestError{}

type mistralPrefixCacheTestError struct{}

func (*mistralPrefixCacheTestError) Error() string {
	return "PrepareChatContext returned an empty Mistral prompt"
}

func BenchmarkMistralPromptPrefixCache(b *testing.B) {
	newRunner := func() *Runner { return newMistralPrefixCacheRunner() }
	system := strings.Repeat("Use the supplied tool schemas exactly and cite sources. ", 96)
	tool := sampleTool()
	tool.Function.Description = strings.Repeat("Returns a forecast with source attribution. ", 64)
	tool.Function.Parameters = []byte(`{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}`)
	tools := []ToolDefinition{tool}
	messages := []ChatMessage{
		UserMessage("Find the weather for Berlin and summarize it."),
		AssistantMessage("I will look that up."),
		UserMessage("Use Celsius."),
	}

	b.Run("disabled", func(b *testing.B) {
		r := newRunner()
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if _, _, ok, err := r.renderMistralInstMessages(messages, system, tools); err != nil || !ok {
				b.Fatalf("render = ok=%v err=%v", ok, err)
			}
		}
	})
	b.Run("enabled", func(b *testing.B) {
		r := newRunner()
		r.EnableMistralPromptPrefixCache(0)
		if _, _, ok, err := r.renderMistralInstMessages(messages, system, tools); err != nil || !ok {
			b.Fatalf("warm render = ok=%v err=%v", ok, err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if _, _, ok, err := r.renderMistralInstMessages(messages, system, tools); err != nil || !ok {
				b.Fatalf("render = ok=%v err=%v", ok, err)
			}
		}
	})
}
