package engine

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func TestForwardBatchStatesMatchesSerial(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	c := r.config
	c.SlidingWindow = 5
	kDim, vDim, headDim, kvHeads, valueDim := r.cacheDims()
	newRun := func() (*KVCache, *DecodeBuffer) {
		return NewKVCache(c.NLayers, kDim, vDim, 48), NewDecodeBuffer(c, headDim, kvHeads, valueDim)
	}
	check := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length %d, want %d", label, len(got), len(want))
		}
		for i, v := range want {
			if !finite32(got[i]) || math.Abs(float64(got[i]-v)) > 1e-5*math.Max(1, math.Abs(float64(v))) {
				t.Fatalf("%s[%d]=%g, want %g", label, i, got[i], v)
			}
		}
	}
	for _, n := range []int{1, 2, 4, 8, 17} {
		ck, cb := newRun()
		bk, bb := newRun()
		for _, start := range []int{0, n, 3} {
			tokens := make([]uint32, n)
			want := make([][]float32, n)
			for i := range tokens {
				tokens[i] = uint32(start+i+1) % uint32(c.VocabSize)
				ForwardBodyInto(c, r.standard, ck, cb, tokens[i], start+i)
				want[i] = slices.Clone(cb.XN)
			}
			got, err := forwardBatchStatesIntoContext(context.Background(), c, r.standard, bk, bb, tokens, start)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != n || &got[0][0] != &bb.batch.XNFlat[0] {
				t.Fatal("hidden states do not borrow the complete batch scratch")
			}
			for i := range got {
				check("hidden", got[i], want[i])
				if len(got[i]) != c.Dim || cap(got[i]) != c.Dim {
					t.Fatal("row exposes unrelated scratch")
				}
			}
			check("final hidden", bb.XN, want[n-1])
			for layer := range ck.K {
				check("K", bk.K[layer][:(start+n)*kDim], ck.K[layer][:(start+n)*kDim])
				check("V", bk.V[layer][:(start+n)*vDim], ck.V[layer][:(start+n)*vDim])
			}
			// A rejected speculative suffix is overwritten by a scalar step.
			ForwardBodyInto(c, r.standard, ck, cb, 1, start+1)
			ForwardBodyInto(c, r.standard, bk, bb, 1, start+1)
			check("continued hidden", bb.XN, cb.XN)
		}
	}

	k, b := newRun()
	if got, err := forwardBatchStatesIntoContext(context.Background(), c, r.standard, k, b, nil, 0); err != nil || got != nil {
		t.Fatalf("empty block = %v, %v", got, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if got, err := forwardBatchStatesIntoContext(ctx, c, r.standard, k, b, []uint32{1, 2}, 0); !errors.Is(err, context.Canceled) || got != nil {
		t.Fatalf("canceled block = %v, %v", got, err)
	}
	for layer := range k.K {
		for _, row := range [][]float32{k.K[layer], k.V[layer]} {
			for _, v := range row {
				if v != 0 {
					t.Fatal("canceled block changed KV")
				}
			}
		}
	}
}
