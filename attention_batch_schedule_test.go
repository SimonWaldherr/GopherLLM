package gopherllm

import (
	"fmt"
	"math"
	"testing"
)

// sequentialBatchAttentionTask preserves the original chronological schedule
// for differential correctness checks and paired scheduling benchmarks.
type sequentialBatchAttentionTask struct{ *batchAttentionTask }

func (t sequentialBatchAttentionTask) runRows(start, end int) {
	for token := start; token < end; token++ {
		t.runToken(token)
	}
}

func TestBatchAttentionScheduleMatchesSequential(t *testing.T) {
	previousThreads := configuredThreads.Swap(4)
	defer configuredThreads.Store(previousThreads)
	const heads, kvHeads, headDim, valueDim = 8, 2, 32, 64
	for _, format := range []string{"f32", "f16", "i8"} {
		for _, mode := range []string{"grouped", "separate", "alibi"} {
			for _, shape := range []struct{ tokens, prefix, window int }{
				{1, 0, 0}, {7, 0, 0}, {8, 0, 0}, {17, 13, 0},
				{7, 0, 5}, {8, 13, 5}, {17, 13, 64},
			} {
				name := fmt.Sprintf("%s/%s/P%d/prefix%d/window%d", format, mode, shape.tokens, shape.prefix, shape.window)
				t.Run(name, func(t *testing.T) {
					cache := NewKVCache(1, kvHeads*headDim, kvHeads*valueDim, shape.prefix+shape.tokens)
					if format == "f16" {
						cache = NewKVCacheF16(1, kvHeads*headDim, kvHeads*valueDim, shape.prefix+shape.tokens)
					} else if format == "i8" {
						cache = NewKVCacheI8(1, kvHeads*headDim, kvHeads*valueDim, shape.prefix+shape.tokens)
					}
					keys, values := make([]float32, kvHeads*headDim), make([]float32, kvHeads*valueDim)
					for pos := 0; pos < cache.MaxLen; pos++ {
						for i := range keys {
							keys[i] = float32((pos*17+i*7)%59-29) / 32
						}
						for i := range values {
							values[i] = float32((pos*11+i*13)%61-30) / 32
						}
						cache.storeKV(0, pos, keys, values)
					}
					q := make([][]float32, shape.tokens)
					want, got := make([][]float32, shape.tokens), make([][]float32, shape.tokens)
					for token := range q {
						q[token] = make([]float32, heads*headDim)
						for i := range q[token] {
							q[token][i] = float32((token*19+i*3)%47-23) / 32
						}
						want[token] = make([]float32, heads*valueDim)
						got[token] = make([]float32, heads*valueDim)
						for i := range got[token] {
							got[token][i] = float32(math.NaN())
						}
					}
					task := batchAttentionTask{
						cache: cache, q: q, attnOut: want, startPos: shape.prefix,
						headDim: headDim, valueDim: valueDim, nHeads: heads, nKVHeads: kvHeads, kvMul: heads / kvHeads,
						slidingWindow: shape.window, usesSWA: shape.window > 0,
						scale: 0.1767767, softcap: 0.7, groupedGQA: mode == "grouped",
						alibi: mode == "alibi", alibiMaxBias: 8,
					}
					sequentialBatchAttentionTask{&task}.runRows(0, shape.tokens)
					task.attnOut = got
					parallelChunksTask(shape.tokens, &task)
					for token := range got {
						for i, value := range got[token] {
							if math.Float32bits(value) != math.Float32bits(want[token][i]) {
								t.Fatalf("token %d element %d: scheduled=%v sequential=%v", token, i, value, want[token][i])
							}
						}
					}
				})
			}
		}
	}
}

// This isolates scheduling from the transformer projections. The first chunk
// has strongly uneven causal spans; prefix and SWA cases track workloads in
// which that imbalance shrinks or disappears. Both variants execute the same
// per-token attention implementation through the persistent worker pool.
func BenchmarkBatchMinistralAttentionSchedule(b *testing.B) {
	const heads, kvHeads, headDim = 32, 8, 128
	for _, shape := range []struct{ tokens, prefix, window int }{
		{128, 0, 0}, {256, 0, 0}, {128, 1024, 0}, {128, 1024, 256},
	} {
		name := fmt.Sprintf("P%d/prefix%d/window%d", shape.tokens, shape.prefix, shape.window)
		b.Run(name, func(b *testing.B) {
			cache := NewKVCache(1, kvHeads*headDim, kvHeads*headDim, shape.prefix+shape.tokens)
			for i := range cache.K[0] {
				cache.K[0][i] = float32(i%59-29) / 32
				cache.V[0][i] = float32(i%61-30) / 32
			}
			q, out := make([][]float32, shape.tokens), make([][]float32, shape.tokens)
			for token := range q {
				q[token] = benchFloatSlice(heads * headDim)
				out[token] = make([]float32, heads*headDim)
			}
			task := &batchAttentionTask{
				cache: cache, q: q, attnOut: out, startPos: shape.prefix,
				headDim: headDim, valueDim: headDim, nHeads: heads, nKVHeads: kvHeads, kvMul: heads / kvHeads,
				slidingWindow: shape.window, usesSWA: shape.window > 0, scale: 0.08838835, groupedGQA: true,
			}
			b.Run("sequential", func(b *testing.B) {
				original := sequentialBatchAttentionTask{task}
				b.ReportAllocs()
				for b.Loop() {
					parallelChunksTask(shape.tokens, original)
				}
			})
			b.Run("balanced", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					parallelChunksTask(shape.tokens, task)
				}
			})
		})
	}
}
