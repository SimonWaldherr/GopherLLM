package engine

import "testing"

func BenchmarkConversationCacheRetainCovered(b *testing.B) {
	const positions = 256
	live := NewKVCache(4, 64, 64, positions)
	tokens := make([]uint32, positions)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	r := &Runner{kind: loadedStandard, prefixCache: prefixCacheState{cache: live, tokens: tokens}}
	other := []uint32{99999}
	r.retainDisplacedConversation(other)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		r.retainDisplacedConversation(other)
	}
}

func BenchmarkConversationCacheRestoreSuffix(b *testing.B) {
	const positions, resident = 512, 480
	live := NewKVCache(4, 64, 64, positions+1)
	tokens := make([]uint32, positions+1)
	for i := range tokens {
		tokens[i] = uint32(i)
	}
	snapshot := cloneKVPrefix(live, positions)
	r := &Runner{kind: loadedStandard,
		prefixCache:   prefixCacheState{cache: live, tokens: tokens[:resident]},
		conversations: conversationCache{entries: []conversationSnapshot{{tokens: tokens[:positions], cache: snapshot}}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if n := r.reuseConversation(live, tokens, resident); n != positions {
			b.Fatalf("reused=%d", n)
		}
	}
}
