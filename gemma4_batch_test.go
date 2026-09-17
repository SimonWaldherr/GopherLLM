package gopherllm

import (
	"context"
	"math"
	"testing"
)

func TestNativeGemma4BatchMatchesTokenForward(t *testing.T) {
	for _, data := range [][]byte{buildTinyNativeGemma4GGUF(false), buildTinyNativeGemma4E2BGGUF()} {
		r, err := RunnerFromGGUFBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()
		// Cross the local attention boundary inside a chunk, then append a chunk
		// at nonzero offset. E2B additionally reads KV written by earlier layers.
		r.config.SlidingWindow = 2
		if !r.canBatchNativeGemma4() {
			t.Fatal("dense native graph declined")
		}
		kd, vd, hd, kv, val := r.cacheDims()
		for _, half := range []bool{false, true} {
			makeCache := func() *KVCache {
				if half {
					return NewKVCacheF16(r.kvCacheLayerCount(), kd, vd, 12)
				}
				return NewKVCache(r.kvCacheLayerCount(), kd, vd, 12)
			}
			a, b := makeCache(), makeCache()
			ba, bb := NewDecodeBuffer(r.config, hd, kv, val), NewDecodeBuffer(r.config, hd, kv, val)
			tokens := []uint32{9, 10, 11, 10, 9, 11, 9, 10, 11}
			var want, got []float32
			for pos, tok := range tokens {
				ForwardGemma4Into(r.config, r.gemma4, a, ba, tok, pos, &want)
			}
			for pos := 0; pos < 3; pos++ {
				ForwardGemma4Into(r.config, r.gemma4, b, bb, tokens[pos], pos, &got)
			}
			forwardNativeGemma4BatchInto(r.config, r.gemma4, b, bb, tokens[3:7], 3, false, nil)
			forwardNativeGemma4BatchInto(r.config, r.gemma4, b, bb, tokens[7:], 7, true, &got)
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 2e-5 {
					t.Fatalf("half=%v logit %d got %v want %v", half, i, got[i], want[i])
				}
			}
			for l := 0; l < a.layerCount(); l++ {
				if half {
					for i := range a.K16[l] {
						if a.K16[l][i] != b.K16[l][i] {
							t.Fatalf("half K layer=%d idx=%d", l, i)
						}
					}
					for i := range a.V16[l] {
						if a.V16[l][i] != b.V16[l][i] {
							t.Fatalf("half V layer=%d idx=%d", l, i)
						}
					}
				} else {
					requireMatvecOutputsClose(t, "KV keys", b.K[l], a.K[l])
					requireMatvecOutputsClose(t, "KV values", b.V[l], a.V[l])
				}
			}
			// Cancellation must stop before writing a new chunk.
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if err := r.prefillBatchedAt(ctx, b, bb, tokens, 0, &got); err != context.Canceled {
				t.Fatalf("cancel result=%v", err)
			}
		}
	}
}

func TestNativeGemma4BatchFallbacks(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyNativeGemma4MoEGGUF(""))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if r.canBatchNativeGemma4() {
		t.Fatal("routed MoE must use token path")
	}
	dense, err := RunnerFromGGUFBytes(buildTinyNativeGemma4GGUF(false))
	if err != nil {
		t.Fatal(err)
	}
	defer dense.Close()
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "1")
	if dense.canBatchNativeGemma4() {
		t.Fatal("disable flag ignored")
	}
}
