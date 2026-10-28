package engine

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
)

// Cancel at the first context check after a snapshot actually overwrote the
// live cache, without depending on timers or racing the copy goroutine.
type cancelOnRestoredKV struct {
	context.Context
	cancel context.CancelFunc
	live   *KVCache
	index  int
	bits   uint32
}

func (c *cancelOnRestoredKV) Err() error {
	if math.Float32bits(c.live.K[0][c.index]) == c.bits {
		c.cancel()
	}
	return c.Context.Err()
}

func TestSnapshotRestoreCancellationInvalidatesResidentPrefix(t *testing.T) {
	oldF16, oldI8 := useF16KVCache.Swap(false), useI8KVCache.Swap(false)
	t.Cleanup(func() { useF16KVCache.Store(oldF16); useI8KVCache.Store(oldI8) })
	for _, static := range []bool{false, true} {
		name := "conversation"
		if static {
			name = "mistral_static"
		}
		t.Run(name, func(t *testing.T) {
			var r *Runner
			var err error
			var a, b []ChatMessage
			opts := tinyMistralKVPrefixOptions()
			opts.SystemPrompt = ""
			if static {
				r = newTinyMistralKVPrefixRunner(t)
				r.SetConversationCacheLimit(0)
				r.EnableMistralKVPrefixCache(0)
				a = []ChatMessage{{Role: ChatRoleSystem, Content: "aaaa"}, UserMessage("b")}
				b = []ChatMessage{{Role: ChatRoleSystem, Content: "dddd"}, UserMessage("b")}
			} else {
				r, err = RunnerFromGGUFBytes(buildTinyLlamaGGUF())
				if err != nil {
					t.Fatal(err)
				}
				a = []ChatMessage{UserMessage(strings.Repeat("a b ", 20))}
				b = []ChatMessage{UserMessage(strings.Repeat("d e ", 20))}
			}
			defer r.Close()
			if _, err := r.GenerateChat(a, opts); err != nil {
				t.Fatal(err)
			}
			if _, err := r.GenerateChat(b, opts); err != nil {
				t.Fatal(err)
			}
			var source *KVCache
			positions := 0
			if static {
				full := r.renderMessages(a, opts.SystemPrompt, opts.ActiveTools())
				prefix := r.mistralStaticPromptPrefix(a, opts.SystemPrompt, opts.ActiveTools(), full)
				source = r.mistralKVPrefixCache.entries[mistralKVPrefixTokenKey(prefix)].cache
				positions = len(prefix)
			} else {
				if len(r.conversations.entries) == 0 {
					t.Fatal("snapshot missing")
				}
				source = r.conversations.entries[0].cache
				positions = len(r.conversations.entries[0].tokens) - 1
			}
			if source == nil {
				t.Fatal("snapshot missing")
			}
			live := r.workspaceCache
			index := -1
			for i := 0; i < positions*live.PerPosKDim; i++ {
				if math.Float32bits(live.K[0][i]) != math.Float32bits(source.K[0][i]) {
					index = i
					break
				}
			}
			if index < 0 {
				t.Fatal("fixture has no differing snapshot row")
			}
			base, cancel := context.WithCancel(context.Background())
			defer cancel()
			ctx := &cancelOnRestoredKV{Context: base, cancel: cancel, live: live, index: index, bits: math.Float32bits(source.K[0][index])}
			if _, err := r.GenerateChat(a, opts.WithContext(ctx)); !errors.Is(err, context.Canceled) {
				t.Fatalf("restore cancellation = %v", err)
			}
			if r.prefixCache.cache != nil {
				t.Fatal("canceled restore retained metadata for overwritten resident KV")
			}
			got, err := r.GenerateChat(b, opts)
			if err != nil {
				t.Fatal(err)
			}
			gotLogits := slices.Clone(r.prefixCache.promptLogits)
			r.ClearConversationCache()
			r.DisableMistralKVPrefixCache()
			want, err := r.GenerateChat(b, opts)
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != want.Text || !slices.Equal(gotLogits, r.prefixCache.promptLogits) {
				t.Fatal("generation after canceled restore differs from cold output")
			}
		})
	}
}
