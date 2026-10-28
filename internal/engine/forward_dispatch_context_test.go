package engine

import (
	"context"
	"errors"
	"slices"
	"testing"
)

// Cancel after the outer prefill loop's check, at the chunk entry point.
// Using Background inside the dispatch wrapper would lose this cancellation.
type prefillChunkEntryCancelContext struct {
	context.Context
	cancel context.CancelFunc
	calls  int
}

func (c *prefillChunkEntryCancelContext) Err() error {
	c.calls++
	if c.calls == 2 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestPrefillBatchedAtPropagatesContextIntoChunk(t *testing.T) {
	// No Metal device is needed: cancellation must reach the chunk before
	// either CPU computation or the whole-GPU-prefill hook can begin.
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.canBatchPrefill() {
		t.Fatal("fixture does not enter the standard batch graph")
	}
	kDim, vDim, headDim, kvHeads, valueDim := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 2)
	buf := NewDecodeBuffer(r.config, headDim, kvHeads, valueDim)
	base, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &prefillChunkEntryCancelContext{Context: base, cancel: cancel}
	logits := []float32{91}
	err = r.prefillBatchedAt(ctx, cache, buf, []uint32{0}, 0, &logits)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("chunk lost request cancellation: err=%v, context checks=%d", err, ctx.calls)
	}
	if ctx.calls != 2 {
		t.Fatalf("cancellation reached an unexpected boundary: %d checks", ctx.calls)
	}
	if len(buf.batch.X) != 0 || !slices.Equal(logits, []float32{91}) {
		t.Fatal("canceled chunk touched activation scratch or logits")
	}
	for layer := range cache.K {
		for _, rows := range [][]float32{cache.K[layer], cache.V[layer]} {
			for _, value := range rows {
				if value != 0 {
					t.Fatal("canceled chunk published K/V rows")
				}
			}
		}
	}
}
