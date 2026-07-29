package gopherllm

import (
	"math"
	"testing"
)

func TestParallelAttendHeadsMatchesDirectLoop(t *testing.T) {
	const (
		nHeads  = 4
		headDim = 4
		ctx     = 3
	)
	config := Config{
		NHeads:           nHeads,
		NKVHeads:         nHeads,
		HeadDim:          headDim,
		ValueDim:         headDim,
		AttnLogitSoftcap: 0,
	}
	layer := LayerWeights{AttnSinks: []float32{0.25, -0.5, 0.75, -1}}
	cache := NewKVCache(1, nHeads*headDim, nHeads*headDim, ctx)
	for pos := 0; pos < ctx; pos++ {
		k := make([]float32, nHeads*headDim)
		v := make([]float32, nHeads*headDim)
		for i := range k {
			k[i] = float32((pos+1)*(i+1)) / 32
			v[i] = float32((pos+2)*(i-3)) / 16
		}
		cache.storeKV(0, pos, k, v)
	}

	q := make([]float32, nHeads*headDim)
	for i := range q {
		q[i] = float32(i-5) / 8
	}
	want := make([]float32, nHeads*headDim)
	const scale = float32(0.5)
	for h := 0; h < nHeads; h++ {
		off := h * headDim
		cache.attendHeadWithSink(0, h, q[off:off+headDim], headDim, headDim,
			0, ctx-1, scale, 0, layer.AttnSinks[h], true, want[off:off+headDim])
	}

	buf := &DecodeBuffer{Q: q, AttnOut: make([]float32, len(want))}
	parallelAttendHeads(config, layer, cache, buf, 0, ctx-1, 0, scale, 1)
	for i, got := range buf.AttnOut {
		if math.Abs(float64(got-want[i])) > 1e-6 {
			t.Fatalf("attention output[%d] = %g, want %g", i, got, want[i])
		}
	}
}
