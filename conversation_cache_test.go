package gopherllm

import (
	"reflect"
	"strings"
	"testing"
)

func TestConversationCacheRestoresDisplacedChat(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	opts := DefaultGenerationOptions()
	opts.SystemPrompt = ""
	opts.MaxTokens = 1
	opts.Seed = 7
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	a := []ChatMessage{UserMessage(strings.Repeat("a b ", 20))}
	b := []ChatMessage{UserMessage(strings.Repeat("d e ", 20))}
	first, err := r.GenerateChat(a, opts)
	if err != nil {
		t.Fatal(err)
	}
	if first.Stats.PromptTokens < 32 {
		t.Fatal("fixture too short")
	}
	if _, err = r.GenerateChat(b, opts); err != nil {
		t.Fatal(err)
	}
	if len(r.conversations.entries) == 0 {
		t.Fatal("displaced chat not retained")
	}
	resumed := append(append([]ChatMessage(nil), a...), AssistantMessage(first.Text), UserMessage("c"))
	warm, err := r.GenerateChat(resumed, opts)
	if err != nil {
		t.Fatal(err)
	}
	if warm.PromptCache == nil || warm.PromptCache.ReusedTokens < 32 {
		t.Fatalf("cache=%+v", warm.PromptCache)
	}
	r.ClearConversationCache()
	if r.prefixCache.cache != nil || len(r.conversations.entries) != 0 {
		t.Fatal("clear left state behind")
	}
	cold, err := r.GenerateChat(resumed, opts)
	if err != nil {
		t.Fatal(err)
	}
	if warm.Text != cold.Text {
		t.Fatalf("warm=%q cold=%q", warm.Text, cold.Text)
	}
	r.SetConversationCacheLimit(0)
	if r.conversationCacheSupported() {
		t.Fatal("disable ignored")
	}
}

func TestConversationCacheSnapshotsAndLimits(t *testing.T) {
	for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
		makeCache := func() *KVCache {
			switch format {
			case kvF16:
				return NewKVCacheF16(2, 32, 32, 80)
			case kvI8:
				return NewKVCacheI8(2, 32, 32, 80)
			default:
				return NewKVCache(2, 32, 32, 80)
			}
		}
		r := &Runner{kind: loadedStandard}
		live := makeCache()
		a := make([]uint32, 40)
		b := make([]uint32, 40)
		for i := range a {
			a[i] = uint32(i)
			b[i] = uint32(i + 100)
		}
		k, v := make([]float32, 32), make([]float32, 32)
		for pos := range 40 {
			for i := range k {
				k[i] = float32(pos+i) / 64
				v[i] = -k[i]
			}
			for l := range 2 {
				live.storeKV(l, pos, k, v)
			}
		}
		r.prefixCache = prefixCacheState{cache: live, tokens: a, promptTokens: 39}
		r.retainDisplacedConversation(b)
		if len(r.conversations.entries) != 1 {
			t.Fatal("snapshot missing")
		}
		saved := cloneKVPrefix(live, 40)
		a[0] = 999 // snapshot token ownership must be independent
		for l := range 2 {
			for pos := range 40 {
				clear(k)
				clear(v)
				live.storeKV(l, pos, k, v)
			}
		}
		target := append([]uint32(nil), r.conversations.entries[0].tokens...)
		target = append(target, 123)
		r.prefixCache = prefixCacheState{} // no displaced source in this check
		if got := r.reuseConversation(live, target, 0); got != 40 {
			t.Fatalf("format=%v reused=%d", format, got)
		}
		restored := cloneKVPrefix(live, 40)
		if !reflect.DeepEqual(saved, restored) {
			t.Fatalf("format=%v snapshot was not immutable/exact", format)
		}
		// Incompatible storage cannot reuse a snapshot.
		wrong := NewKVCache(1, 32, 32, 80)
		if got := r.reuseConversation(wrong, target, 0); got != 0 {
			t.Fatal("incompatible snapshot reused")
		}
		r.SetConversationCacheLimit(1)
		r.prefixCache = prefixCacheState{cache: live, tokens: target}
		r.retainDisplacedConversation(b)
		if len(r.conversations.entries) != 0 {
			t.Fatal("oversized entry retained")
		}
		r.SetConversationCacheLimit(DefaultConversationCacheBytes)
		for n := 0; n < 7; n++ {
			toks := make([]uint32, 40)
			for i := range toks {
				toks[i] = uint32(n*100 + i)
			}
			r.prefixCache = prefixCacheState{cache: live, tokens: toks}
			r.retainDisplacedConversation([]uint32{9999})
		}
		if len(r.conversations.entries) != 4 || r.conversations.bytes > DefaultConversationCacheBytes {
			t.Fatal("LRU bounds violated")
		}
	}
	for _, kind := range []loadedKind{loadedQwen35, loadedNemotronH, loadedMamba2} {
		r := &Runner{kind: kind}
		if r.conversationCacheSupported() {
			t.Fatal("recurrent model accepted")
		}
	}
}

func TestConversationCacheHitSurvivesEviction(t *testing.T) {
	r := &Runner{kind: loadedStandard}
	live := NewKVCache(1, 4, 4, 64)
	a, b := make([]uint32, 40), make([]uint32, 40)
	for i := range a {
		a[i] = uint32(i)
		b[i] = uint32(i + 100)
	}
	for i := range live.K[0] {
		live.K[0][i] = float32(i)
		live.V[0][i] = -float32(i)
	}
	expected := cloneKVPrefix(live, 40)
	cost := kvCacheStorageBytes(expected) + int64(len(a))*4
	r.SetConversationCacheLimit(cost)
	r.prefixCache = prefixCacheState{cache: live, tokens: a}
	r.retainDisplacedConversation(b)
	clear(live.K[0])
	clear(live.V[0])
	r.prefixCache = prefixCacheState{cache: live, tokens: b}
	// Saving B evicts A under this one-snapshot budget. The pending hit must
	// keep A's immutable source alive until its rows have been copied.
	if got := r.reuseConversation(live, append(append([]uint32(nil), a...), 999), 0); got != 40 {
		t.Fatalf("reused=%d", got)
	}
	if !reflect.DeepEqual(cloneKVPrefix(live, 40), expected) {
		t.Fatal("eviction corrupted pending hit")
	}
	if len(r.conversations.entries) != 1 || r.conversations.bytes > cost {
		t.Fatal("one-snapshot budget violated")
	}
}
