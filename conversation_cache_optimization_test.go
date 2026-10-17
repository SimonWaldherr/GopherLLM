package gopherllm

import (
	"fmt"
	"reflect"
	"testing"
)

func conversationCacheTestTokens(n int, base uint32) []uint32 {
	tokens := make([]uint32, n)
	for i := range tokens {
		tokens[i] = base + uint32(i)
	}
	return tokens
}

func TestConversationCacheRetainsBudgetedLongPrefix(t *testing.T) {
	live := NewKVCache(2, 32, 32, 100)
	tokens := conversationCacheTestTokens(100, 0)
	const retained = 40
	cost := kvCacheBytes(2, 32, 32, retained, kvF32) + retained*4
	r := &Runner{kind: loadedStandard}
	r.SetConversationCacheLimit(cost)
	r.prefixCache = prefixCacheState{cache: live, tokens: tokens}
	r.retainDisplacedConversation([]uint32{99999})
	if len(r.conversations.entries) != 1 {
		t.Fatal("long conversation lost its budgeted prefix")
	}
	entry := r.conversations.entries[0]
	if len(entry.tokens) != retained || entry.cache.MaxLen != retained || entry.bytes != cost || r.conversations.bytes != cost {
		t.Fatalf("snapshot tokens=%d maxlen=%d bytes=%d total=%d", len(entry.tokens), entry.cache.MaxLen, entry.bytes, r.conversations.bytes)
	}
	// The same prefix of a longer history must not allocate or replace a
	// snapshot that already owns every retained row.
	r.retainDisplacedConversation([]uint32{99999})
	if r.conversations.entries[0].cache != entry.cache || r.conversations.bytes != cost {
		t.Fatal("covered long prefix was copied again")
	}
	r.prefixCache = prefixCacheState{}
	if got := r.reuseConversation(live, tokens, 0); got != retained {
		t.Fatalf("reused=%d, want %d", got, retained)
	}
}

func TestConversationCacheCoveredSnapshotTouchesLRU(t *testing.T) {
	live := NewKVCache(1, 4, 4, 80)
	a, b := conversationCacheTestTokens(60, 0), conversationCacheTestTokens(40, 100)
	r := &Runner{kind: loadedStandard}
	for _, tokens := range [][]uint32{a, b} {
		r.prefixCache = prefixCacheState{cache: live, tokens: tokens}
		r.retainDisplacedConversation([]uint32{99999})
	}
	entryA, oldBytes := r.conversations.entries[0], r.conversations.bytes
	r.prefixCache = prefixCacheState{cache: live, tokens: a[:40]}
	r.retainDisplacedConversation([]uint32{99999})
	if len(r.conversations.entries) != 2 || r.conversations.entries[1].cache != entryA.cache || r.conversations.bytes != oldBytes {
		t.Fatal("covered snapshot was copied or not promoted to MRU")
	}
	// A newer extended branch replaces its shorter snapshot. A divergent
	// branch retains its own KV state even when much of its prefix matches.
	extended := append(append([]uint32(nil), a...), 60, 61, 62)
	r.prefixCache = prefixCacheState{cache: live, tokens: extended}
	r.retainDisplacedConversation([]uint32{99999})
	if len(r.conversations.entries) != 2 || len(r.conversations.entries[1].tokens) != len(extended) {
		t.Fatal("extended snapshot did not subsume its shorter branch")
	}
	branch := append([]uint32(nil), extended...)
	branch[35] = 9999
	r.prefixCache = prefixCacheState{cache: live, tokens: branch}
	r.retainDisplacedConversation([]uint32{99999})
	if len(r.conversations.entries) != 3 {
		t.Fatal("divergent branch incorrectly subsumed")
	}
}

func TestConversationCacheOnlySkipsVerifiedResidentRows(t *testing.T) {
	for _, mode := range []string{"none", "other_workspace", "partial_match"} {
		t.Run(mode, func(t *testing.T) {
			const positions = 40
			source, live := NewKVCache(1, 4, 4, positions+1), NewKVCache(1, 4, 4, positions+1)
			for i := range source.K[0] {
				source.K[0][i], source.V[0][i] = float32(i+1), -float32(i+1)
			}
			tokens := conversationCacheTestTokens(positions+1, 0)
			r := &Runner{kind: loadedStandard, conversations: conversationCache{entries: []conversationSnapshot{{tokens: tokens[:positions], cache: source}}}}
			switch mode {
			case "other_workspace":
				r.prefixCache = prefixCacheState{cache: source, tokens: tokens[:32]}
			case "partial_match":
				residentTokens := append([]uint32(nil), tokens[:32]...)
				residentTokens[8] = 9999
				r.prefixCache = prefixCacheState{cache: live, tokens: residentTokens}
				copyKVPrefix(live, source, 8)
			}
			if got := r.reuseConversation(live, tokens, 32); got != positions {
				t.Fatalf("reused=%d", got)
			}
			if !reflect.DeepEqual(cloneKVPrefix(source, positions), cloneKVPrefix(live, positions)) {
				t.Fatal("unverified resident rows were skipped")
			}
		})
	}
}

func TestConversationKVRangeExactAndBounded(t *testing.T) {
	for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
		t.Run(fmt.Sprint(format), func(t *testing.T) {
			makeCache := func() *KVCache {
				switch format {
				case kvF16:
					return NewKVCacheF16(2, 32, 64, 8)
				case kvI8:
					return NewKVCacheI8(2, 32, 64, 8)
				default:
					return NewKVCache(2, 32, 64, 8)
				}
			}
			src, dst, want := makeCache(), makeCache(), makeCache()
			k, v := make([]float32, 32), make([]float32, 64)
			for pos := range 8 {
				for i := range k {
					k[i] = float32(pos+i+1) * .17
				}
				for i := range v {
					v[i] = -float32(pos+i+1) * .37
				}
				for layer := range 2 {
					src.storeKV(layer, pos, k, v)
					if pos >= 2 && pos < 6 {
						want.storeKV(layer, pos, k, v)
					}
				}
			}
			if got := copyConversationKVRange(dst, src, 2, 6); got != 6 || !reflect.DeepEqual(dst, want) {
				t.Fatalf("range copy touched a prefix/tail row or changed storage: copied=%d", got)
			}
			before := cloneKVPrefix(dst, 8)
			for _, bounds := range [][2]int{{-1, 6}, {4, 3}, {0, 9}, {0, 0}} {
				if got := copyConversationKVRange(dst, src, bounds[0], bounds[1]); got != 0 {
					t.Fatalf("invalid bounds %v copied=%d", bounds, got)
				}
			}
			if !reflect.DeepEqual(dst, before) {
				t.Fatal("invalid range mutated destination")
			}
			if got := copyConversationKVRange(dst, NewKVCache(1, 32, 64, 8), 2, 6); got != 0 {
				t.Fatal("incompatible shape accepted")
			}
		})
	}
}

func TestConversationCacheSafetyGates(t *testing.T) {
	for _, r := range []*Runner{
		{kind: loadedStandard, config: Config{UsesMLA: true}},
		{kind: loadedStandard, outOfCore: true},
		{kind: loadedQwen35}, {kind: loadedNemotronH}, {kind: loadedMamba2},
	} {
		if r.conversationCacheSupported() {
			t.Fatal("non-attention snapshot path accepted")
		}
	}
}
