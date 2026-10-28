package engine

import (
	"fmt"
	"testing"
)

func mistralKVPrefixBudgetTestCache(format kvFormat, positions int) *KVCache {
	switch format {
	case kvF16:
		return NewKVCacheF16(4, 64, 64, positions)
	case kvI8:
		return NewKVCacheI8(4, 64, 64, positions)
	default:
		return NewKVCache(4, 64, 64, positions)
	}
}

func BenchmarkMistralKVPrefixSnapshotOversized(b *testing.B) {
	const positions = 256
	for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
		b.Run(fmt.Sprint(format), func(b *testing.B) {
			r := &Runner{}
			r.EnableMistralKVPrefixCache(1)
			epoch, _ := r.mistralKVPrefixCacheEpoch()
			src := mistralKVPrefixBudgetTestCache(format, positions)
			tokens := make([]uint32, positions)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				r.putMistralKVPrefixSnapshot(src, tokens, epoch)
			}
		})
	}
}

func TestMistralKVPrefixSnapshotBudgetPreflight(t *testing.T) {
	const positions = 32
	for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
		t.Run(fmt.Sprint(format), func(t *testing.T) {
			src := mistralKVPrefixBudgetTestCache(format, positions)
			tokens := make([]uint32, positions)
			cost := kvCacheStorageBytes(src) + int64(len(tokens))*4
			r := &Runner{}
			r.EnableMistralKVPrefixCache(cost - 1)
			epoch, _ := r.mistralKVPrefixCacheEpoch()
			if allocs := testing.AllocsPerRun(10, func() {
				r.putMistralKVPrefixSnapshot(src, tokens, epoch)
			}); allocs != 0 {
				t.Fatalf("rejected oversized snapshot allocated %g times", allocs)
			}
			if stats := r.MistralKVPrefixCacheStats(); stats.Entries != 0 || stats.Bytes != 0 {
				t.Fatalf("rejected snapshot mutated cache: %+v", stats)
			}
			// The boundary includes every token-key byte and uses the actual
			// F32/F16/I8 row layout, without a conservative rounding excess.
			r.EnableMistralKVPrefixCache(cost)
			epoch, _ = r.mistralKVPrefixCacheEpoch()
			r.putMistralKVPrefixSnapshot(src, tokens, epoch)
			if stats := r.MistralKVPrefixCacheStats(); stats.Entries != 1 || stats.Bytes != cost {
				t.Fatalf("exact budget rejected: %+v, cost=%d", stats, cost)
			}
		})
	}
}

func TestMistralKVPrefixSnapshotRejectsMissingRows(t *testing.T) {
	r := &Runner{}
	r.EnableMistralKVPrefixCache(0)
	epoch, _ := r.mistralKVPrefixCacheEpoch()
	src := NewKVCache(1, 4, 4, 8)
	tokens := make([]uint32, 9)
	if allocs := testing.AllocsPerRun(10, func() {
		r.putMistralKVPrefixSnapshot(src, tokens, epoch)
	}); allocs != 0 {
		t.Fatalf("incomplete prefix allocated %g times", allocs)
	}
	if stats := r.MistralKVPrefixCacheStats(); stats.Entries != 0 || stats.Bytes != 0 {
		t.Fatalf("incomplete prefix retained truncated KV rows: %+v", stats)
	}
}
