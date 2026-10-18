package gopherllm

import (
	"context"
	"fmt"
	"slices"
	"testing"
)

func TestQwen35RecurrentReplayMatchesVerifiedPrefix(t *testing.T) {
	withQ8Activations(true, func() {
		for _, quantized := range []bool{false, true} {
			cfg, w := qwen35BatchFixture(quantized)
			for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
				for _, keep := range []int{1, 3, 7} {
					t.Run(fmt.Sprintf("quant=%v/kv=%v/keep=%d", quantized, format, keep), func(t *testing.T) {
						cache, ref := qwen35BatchCache(cfg, w, format), qwen35BatchCache(cfg, w, format)
						buf, refBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
						for pos, token := range []uint32{1, 4, 3} {
							ForwardQwen35BodyInto(cfg, w, cache, buf, token, pos)
							ForwardQwen35BodyInto(cfg, w, ref, refBuf, token, pos)
						}
						checkpoint := cache.Qwen35.snapshot()
						inputs := []uint32{5, 12, 19, 26, 1, 8, 15, 22}
						buf.speculative.active = true
						if _, err := forwardQwen35BatchStatesInto(context.Background(), cfg, w, cache, buf, inputs, 3); err != nil {
							t.Fatal(err)
						}
						if !buf.speculative.qwenReplay.valid || !cache.Qwen35.restore(checkpoint) {
							t.Fatal("missing replay inputs or checkpoint")
						}
						if _, err := forwardQwen35BatchStatesInto(context.Background(), cfg, w, ref, refBuf, inputs[:keep], 3); err != nil {
							t.Fatal(err)
						}
						if err := buf.speculative.qwenReplay.replay(context.Background(), cfg, w, cache.Qwen35, buf, keep); err != nil {
							t.Fatal(err)
						}
						if !slices.Equal(cache.Qwen35.Conv, ref.Qwen35.Conv) || !slices.Equal(cache.Qwen35.State, ref.Qwen35.State) {
							t.Fatal("projection-free recurrence replay differs from causal prefix")
						}
						// Rejected attention rows are still present. Continuing must
						// use only the restored recurrent/attention prefix.
						var got, want []float32
						for step, token := range []uint32{2, 17, 6} {
							ForwardQwen35Into(cfg, w, cache, buf, token, 3+keep+step, &got)
							ForwardQwen35Into(cfg, w, ref, refBuf, token, 3+keep+step, &want)
							if !slices.Equal(got, want) {
								t.Fatal("rejected suffix affected continuation")
							}
						}
					})
				}
			}
		}
	})
}

func TestQwen35RecurrentReplayBudgetAndCancellation(t *testing.T) {
	cfg, w := qwen35BatchFixture(false)
	r := new(qwen35RecurrentReplay)
	r.begin(cfg, 2, 8, false)
	if r.valid || len(r.layers) != 0 {
		t.Fatal("retained projection inputs outside a speculative block")
	}
	r.begin(cfg, 1<<24, 8, true)
	if r.valid || len(r.layers) != 0 {
		t.Fatal("allocated replay workspace beyond budget")
	}
	cache := qwen35BatchCache(cfg, w, kvF32)
	buf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
	buf.speculative.active = true
	if _, err := forwardQwen35BatchStatesInto(context.Background(), cfg, w, cache, buf, []uint32{1, 2}, 0); err != nil {
		t.Fatal(err)
	}
	before := cache.Qwen35.snapshot()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := buf.speculative.qwenReplay.replay(ctx, cfg, w, cache.Qwen35, buf, 1); err != context.Canceled {
		t.Fatalf("canceled replay returned %v", err)
	}
	if !slices.Equal(before.Conv, cache.Qwen35.Conv) || !slices.Equal(before.State, cache.Qwen35.State) {
		t.Fatal("pre-canceled replay mutated the cache")
	}
}
