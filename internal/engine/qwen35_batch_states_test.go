package engine

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"
)

func qwen35BlockTolerances(quantized bool, format kvFormat) (outputs, hidden float64) {
	outputs, hidden = 5e-5, 5e-5
	if quantized && format != kvF32 && !useQ8Activations.Load() {
		// Match the existing compressed-KV continuation budgets. Scalar
		// quantized-block algebra and batched dequantize-first dots can cross
		// one codec rounding boundary on non-Q8 activation backends.
		outputs, hidden = 1e-4, 1.25e-4
	}
	return
}

func TestQwen35BlockStatesAllPositions(t *testing.T) {
	for _, quantized := range []bool{false, true} {
		cfg, weights := qwen35BatchFixture(quantized)
		for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
			if format == kvI8 && !quantized {
				continue
			}
			for _, n := range []int{1, 2, 4, 8, 31, 32} {
				t.Run(fmt.Sprintf("quant=%t/kv=%d/tokens=%d", quantized, format, n), func(t *testing.T) {
					gotCache, wantCache := qwen35BatchCache(cfg, weights, format), qwen35BatchCache(cfg, weights, format)
					gotBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					wantBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					const offset = 3
					for pos := range offset {
						ForwardQwen35BodyInto(cfg, weights, gotCache, gotBuf, uint32(pos+1), pos)
						ForwardQwen35BodyInto(cfg, weights, wantCache, wantBuf, uint32(pos+1), pos)
					}
					tokens := make([]uint32, n)
					wantHidden, wantLogits := make([][]float32, n), make([][]float32, n)
					for i := range tokens {
						tokens[i] = uint32((i*7 + 5) % cfg.VocabSize)
						ForwardQwen35Into(cfg, weights, wantCache, wantBuf, tokens[i], offset+i, &wantLogits[i])
						wantHidden[i] = slices.Clone(wantBuf.XN)
					}
					if quantized && format != kvF32 && !useQ8Activations.Load() {
						// Scalar quantized dots fold block scales into partial sums;
						// batch dots dequantize first. Their float differences may
						// cross compressed-KV quant/scale boundaries, so a fixed
						// small hidden tolerance does not bound every position. For
						// this combination compare each returned row to the existing
						// last-row graph run on that causal prefix with the same
						// batch arithmetic. Float KV and native Q8 activation paths
						// retain the independent token-at-a-time reference above.
						wantCache = qwen35BatchCache(cfg, weights, format)
						wantBuf = NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
						for pos := range offset {
							ForwardQwen35BodyInto(cfg, weights, wantCache, wantBuf, uint32(pos+1), pos)
						}
						checkpoint := wantCache.Qwen35.snapshot()
						for i := range tokens {
							if !wantCache.Qwen35.restore(checkpoint) {
								t.Fatal("cannot restore causal reference checkpoint")
							}
							if err := forwardQwen35BatchInto(context.Background(), cfg, weights, wantCache, wantBuf, tokens[:i+1], offset, true, &wantLogits[i]); err != nil {
								t.Fatal(err)
							}
							wantHidden[i] = slices.Clone(wantBuf.XN)
						}
					}
					states, err := forwardQwen35BatchStatesInto(context.Background(), cfg, weights, gotCache, gotBuf, tokens, offset)
					if err != nil || len(states) != n {
						t.Fatalf("hidden rows=%d, err=%v", len(states), err)
					}
					outputTol, hiddenTol := qwen35BlockTolerances(quantized, format)
					var logits []float32
					for i, state := range states {
						if len(state) != cfg.Dim || &state[0] != &gotBuf.qwenBatch.XNFlat[i*cfg.Dim] {
							t.Fatal("returned hidden rows do not borrow the contiguous output slab")
						}
						requireQwen35BatchValuesWithin(t, "position hidden", state, wantHidden[i], hiddenTol)
						weights.Output.MatvecInto(state, &logits)
						requireQwen35BatchValuesWithin(t, "position logits", logits, wantLogits[i], outputTol)
						if argmaxFiniteToken(logits) != argmaxFiniteToken(wantLogits[i]) {
							t.Fatalf("position %d changes greedy next token", i)
						}
					}
					requireQwen35BatchValuesWithin(t, "last residual", gotBuf.X, wantBuf.X, outputTol)
					requireQwen35BatchValuesWithin(t, "last normalized hidden", gotBuf.XN, wantBuf.XN, hiddenTol)
					requireQwen35BatchCache(t, gotCache, wantCache, outputTol, false)
				})
			}
		}
	}
}

func TestQwen35BlockCheckpointRestoreAndAcceptedReplay(t *testing.T) {
	for _, quantized := range []bool{false, true} {
		cfg, weights := qwen35BatchFixture(quantized)
		for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
			if format == kvI8 && !quantized {
				continue
			}
			for _, accepted := range []int{0, 1, 3, 7, 8} {
				t.Run(fmt.Sprintf("quant=%t/kv=%d/accepted=%d", quantized, format, accepted), func(t *testing.T) {
					gotCache, wantCache := qwen35BatchCache(cfg, weights, format), qwen35BatchCache(cfg, weights, format)
					gotBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					wantBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					const offset = 3
					for pos := range offset {
						ForwardQwen35BodyInto(cfg, weights, gotCache, gotBuf, uint32(pos+1), pos)
						ForwardQwen35BodyInto(cfg, weights, wantCache, wantBuf, uint32(pos+1), pos)
					}
					var checkpoint *Qwen35Cache
					gotCache.Qwen35.snapshotInto(&checkpoint)
					tokens := []uint32{5, 12, 19, 26, 1, 8, 15, 22}
					if _, err := forwardQwen35BatchStatesInto(context.Background(), cfg, weights, gotCache, gotBuf, tokens, offset); err != nil {
						t.Fatal(err)
					}
					if slices.Equal(gotCache.Qwen35.State, checkpoint.State) || !gotCache.Qwen35.restore(checkpoint) {
						t.Fatal("block did not update recurrence or its checkpoint could not be restored")
					}
					for i, token := range tokens[:accepted] {
						ForwardQwen35BodyInto(cfg, weights, gotCache, gotBuf, token, offset+i)
						ForwardQwen35BodyInto(cfg, weights, wantCache, wantBuf, token, offset+i)
					}
					// Restoring and replaying use the same token graph, so recurrent
					// state must match bit for bit, even if rejected K/V rows remain.
					if !slices.Equal(gotCache.Qwen35.Conv, wantCache.Qwen35.Conv) || !slices.Equal(gotCache.Qwen35.State, wantCache.Qwen35.State) {
						t.Fatal("restored accepted recurrence differs from ordinary decode")
					}
					var got, want []float32
					for step, token := range []uint32{4, 17, 6, 25} {
						pos := offset + accepted + step
						ForwardQwen35Into(cfg, weights, gotCache, gotBuf, token, pos, &got)
						ForwardQwen35Into(cfg, weights, wantCache, wantBuf, token, pos, &want)
						if !slices.Equal(got, want) || !slices.Equal(gotBuf.XN, wantBuf.XN) {
							t.Fatal("rejected K/V suffix affected subsequent accepted decode")
						}
					}
				})
			}
		}
	}
}

type qwen35BlockCancelAfterWrite struct {
	context.Context
	cache      *Qwen35Cache
	checkpoint *Qwen35Cache
	cancel     context.CancelFunc
}

func (c *qwen35BlockCancelAfterWrite) Err() error {
	if !slices.Equal(c.cache.State, c.checkpoint.State) {
		c.cancel()
	}
	return c.Context.Err()
}

func TestQwen35BlockStatesCancellationAndValidation(t *testing.T) {
	cfg, weights := qwen35BatchFixture(false)
	cache := qwen35BatchCache(cfg, weights, kvF32)
	buf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
	var checkpoint *Qwen35Cache
	cache.Qwen35.snapshotInto(&checkpoint)
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &qwen35BlockCancelAfterWrite{Context: base, cache: cache.Qwen35, checkpoint: checkpoint, cancel: cancel}
	states, err := forwardQwen35BatchStatesInto(ctx, cfg, weights, cache, buf, []uint32{1, 2, 3, 4}, 0)
	if states != nil || !errors.Is(err, context.Canceled) || slices.Equal(cache.Qwen35.State, checkpoint.State) {
		t.Fatalf("mid-block cancellation did not return a failed, mutated block: rows=%d err=%v", len(states), err)
	}
	if !cache.Qwen35.restore(checkpoint) {
		t.Fatal("cannot restore recurrence after cancellation")
	}
	if states, err = forwardQwen35BatchStatesInto(ctx, cfg, weights, cache, buf, []uint32{1}, 0); states != nil || !errors.Is(err, context.Canceled) || !slices.Equal(cache.Qwen35.State, checkpoint.State) {
		t.Fatal("pre-canceled block mutated recurrence")
	}
	for _, tc := range []struct {
		start int
		count int
	}{{-1, 1}, {cfg.MaxSeqLen, 1}, {cfg.MaxSeqLen - 1, 2}, {0, 33}} {
		states, err = forwardQwen35BatchStatesInto(context.Background(), cfg, weights, cache, buf, make([]uint32, tc.count), tc.start)
		if states != nil || err == nil || !slices.Equal(cache.Qwen35.State, checkpoint.State) {
			t.Fatalf("invalid range start=%d n=%d mutated recurrence or succeeded", tc.start, tc.count)
		}
	}
	if states, err = forwardQwen35BatchStatesInto(context.Background(), cfg, weights, cache, buf, nil, 0); states != nil || err != nil {
		t.Fatal("empty block did not preserve the existing state")
	}
}

func TestQwen35RecurrentCheckpointReusesStorage(t *testing.T) {
	cfg, weights := qwen35BatchFixture(false)
	cache := qwen35BatchCache(cfg, weights, kvF32)
	var checkpoint *Qwen35Cache
	cache.Qwen35.snapshotInto(&checkpoint)
	conv, state := &checkpoint.Conv[0], &checkpoint.State[0]
	cache.Qwen35.Conv[0], cache.Qwen35.State[0] = 3, 4
	if checkpoint.Conv[0] != 0 || checkpoint.State[0] != 0 {
		t.Fatal("checkpoint aliases the live recurrence")
	}
	if allocations := testing.AllocsPerRun(20, func() { cache.Qwen35.snapshotInto(&checkpoint) }); allocations != 0 {
		t.Fatalf("reusing one recurrent checkpoint allocates %g times per block", allocations)
	}
	if &checkpoint.Conv[0] != conv || &checkpoint.State[0] != state || checkpoint.Conv[0] != 3 || checkpoint.State[0] != 4 {
		t.Fatal("checkpoint did not retain its backing storage or refresh its values")
	}
	bad := checkpoint.snapshot()
	bad.HeadDim++
	if cache.Qwen35.restore(bad) {
		t.Fatal("recurrent checkpoint restore accepted an incompatible shape")
	}
}
