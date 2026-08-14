package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func TestGQA4SharedRowKernelsMatchSeparateOperations(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, n := range []int{1, 3, 4, 15, 16, 17, 31, 32, 95, 128} {
		q := [4][]float32{}
		out := [4][]float32{}
		wantOut := [4][]float32{}
		x := make([]float32, n)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		for h := range q {
			q[h] = make([]float32, n)
			out[h] = make([]float32, n)
			wantOut[h] = make([]float32, n)
			for i := range q[h] {
				q[h][i] = rng.Float32()*2 - 1
				out[h][i] = rng.Float32()*2 - 1
				wantOut[h][i] = out[h][i]
			}
		}

		got0, got1, got2, got3 := dotF32x4(
			&q[0][0], &q[1][0], &q[2][0], &q[3][0], &x[0], n)
		gotDots := [...]float32{got0, got1, got2, got3}
		for h, got := range gotDots {
			want := DotF32(q[h], x)
			if diff := math.Abs(float64(got - want)); diff > 1e-6 {
				t.Fatalf("n=%d dot %d = %g, want %g (diff %g)", n, h, got, want, diff)
			}
		}

		alpha := [...]float32{0.25, -0.5, 0.75, 1.25}
		for h := range wantOut {
			AxpyF32(wantOut[h], alpha[h], x)
		}
		axpyF32x4(&out[0][0], &out[1][0], &out[2][0], &out[3][0],
			alpha[0], alpha[1], alpha[2], alpha[3], &x[0], n)
		for h := range out {
			for i, got := range out[h] {
				if got != wantOut[h][i] {
					t.Fatalf("n=%d axpy %d[%d] = %g, want %g", n, h, i, got, wantOut[h][i])
				}
			}
		}
	}
}

func TestGQA4SharedF16RowKernelsMatchSeparateOperations(t *testing.T) {
	rng := rand.New(rand.NewSource(22))
	for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 95, 128} {
		q := [4][]float32{}
		out := [4][]float32{}
		wantOut := [4][]float32{}
		x := make([]uint16, n)
		for i := range x {
			x[i] = F32ToF16(rng.Float32()*2 - 1)
		}
		for h := range q {
			q[h] = make([]float32, n)
			out[h] = make([]float32, n)
			wantOut[h] = make([]float32, n)
			for i := range q[h] {
				q[h][i] = rng.Float32()*2 - 1
				out[h][i] = rng.Float32()*2 - 1
				wantOut[h][i] = out[h][i]
			}
		}

		got0, got1, got2, got3 := dotF32F16x4(
			&q[0][0], &q[1][0], &q[2][0], &q[3][0], &x[0], n)
		gotDots := [...]float32{got0, got1, got2, got3}
		for h, got := range gotDots {
			want := dotF32F16(q[h], x)
			if diff := math.Abs(float64(got - want)); diff > 1e-5 {
				t.Fatalf("n=%d dot %d = %g, want %g (diff %g)", n, h, got, want, diff)
			}
		}

		alpha := [...]float32{0.25, -0.5, 0.75, 1.25}
		for h := range wantOut {
			axpyF16(wantOut[h], alpha[h], x)
		}
		axpyF32F16x4(&out[0][0], &out[1][0], &out[2][0], &out[3][0],
			alpha[0], alpha[1], alpha[2], alpha[3], &x[0], n)
		for h := range out {
			for i, got := range out[h] {
				if got != wantOut[h][i] {
					t.Fatalf("n=%d axpy %d[%d] = %g, want %g", n, h, i, got, wantOut[h][i])
				}
			}
		}
	}
}

// queryHeadCounts covers the specialized 4-head SIMD path (onlineAttentionGroup4
// et al.) and the generic onlineAttentionGroupEither fallback used for every
// other GQA/MQA ratio — both must agree with per-head attention exactly, since
// the ForwardBodyInto/forwardBatchInto dispatch gate now enables grouping for
// any kvMul > 1, not just 4 (see groupedGQA in model.go/forward_batch.go).
var queryHeadCounts = []int{1, 2, 4, 6, 8}

func TestGroupedGQAAttentionMatchesSeparateHeads(t *testing.T) {
	const headDim = 32
	rng := rand.New(rand.NewSource(42))
	for _, queryHeads := range queryHeadCounts {
		const ctx = 67
		queries := make([]float32, queryHeads*headDim)
		keys := make([]float32, ctx*headDim)
		values := make([]float32, ctx*headDim)
		for i := range queries {
			queries[i] = rng.Float32()*2 - 1
		}
		for i := range keys {
			keys[i] = rng.Float32()*2 - 1
			values[i] = rng.Float32()*2 - 1
		}

		for _, softcap := range []float32{0, 2.5} {
			want := make([]float32, len(queries))
			got := make([]float32, len(queries))
			scale := float32(1 / math.Sqrt(headDim))
			for h := 0; h < queryHeads; h++ {
				off := h * headDim
				onlineAttention(queries[off:off+headDim], keys, values,
					headDim, headDim, headDim, headDim, 0, ctx-1, scale, softcap,
					want[off:off+headDim])
			}
			onlineAttentionGroup(queries, keys, values, queryHeads,
				headDim, headDim, headDim, headDim, 0, ctx-1, scale, softcap, got)
			for i := range got {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
					t.Fatalf("queryHeads=%d softcap=%g output[%d] = %g, want %g (diff %g)", queryHeads, softcap, i, got[i], want[i], diff)
				}
			}
		}
	}
}

func TestGroupedGQAAttentionF16MatchesSeparateHeads(t *testing.T) {
	const headDim = 32
	rng := rand.New(rand.NewSource(84))
	for _, queryHeads := range queryHeadCounts {
		const ctx = 67
		queries := make([]float32, queryHeads*headDim)
		keys := make([]uint16, ctx*headDim)
		values := make([]uint16, ctx*headDim)
		for i := range queries {
			queries[i] = rng.Float32()*2 - 1
		}
		for i := range keys {
			keys[i] = F32ToF16(rng.Float32()*2 - 1)
			values[i] = F32ToF16(rng.Float32()*2 - 1)
		}

		want := make([]float32, len(queries))
		got := make([]float32, len(queries))
		scale := float32(1 / math.Sqrt(headDim))
		for h := 0; h < queryHeads; h++ {
			off := h * headDim
			onlineAttentionF16(queries[off:off+headDim], keys, values,
				headDim, headDim, headDim, headDim, 0, ctx-1, scale, 2.5,
				want[off:off+headDim])
		}
		onlineAttentionGroupF16(queries, keys, values, queryHeads,
			headDim, headDim, headDim, headDim, 0, ctx-1, scale, 2.5, got)
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
				t.Fatalf("queryHeads=%d output[%d] = %g, want %g (diff %g)", queryHeads, i, got[i], want[i], diff)
			}
		}
	}
}

// TestGroupedGQAAttentionI8MatchesSeparateHeads is the Q8_0-KV-cache analogue
// of the F32/F16 tests above. onlineAttentionGroupI8 had no correctness test
// at all before the dispatch gate started allowing kvI8 caches into the
// grouped path (previously grouping only ever ran on a kvF32 cache).
func TestGroupedGQAAttentionI8MatchesSeparateHeads(t *testing.T) {
	const headDim = 32
	rng := rand.New(rand.NewSource(91))
	for _, queryHeads := range queryHeadCounts {
		const ctx = 67
		queries := make([]float32, queryHeads*headDim)
		keysF := randomVec(rng, ctx*headDim)
		valuesF := randomVec(rng, ctx*headDim)
		keys8 := make([]byte, 0, ctx*q8RowBytes(headDim))
		values8 := make([]byte, 0, ctx*q8RowBytes(headDim))
		for i := range queries {
			queries[i] = rng.Float32()*2 - 1
		}
		for t := 0; t < ctx; t++ {
			keys8 = append(keys8, QuantizeRowQ8_0(keysF[t*headDim:(t+1)*headDim], headDim)...)
			values8 = append(values8, QuantizeRowQ8_0(valuesF[t*headDim:(t+1)*headDim], headDim)...)
		}

		want := make([]float32, len(queries))
		got := make([]float32, len(queries))
		scale := float32(1 / math.Sqrt(headDim))
		for h := 0; h < queryHeads; h++ {
			off := h * headDim
			onlineAttentionI8WithSink(queries[off:off+headDim], keys8, values8,
				headDim, headDim, headDim, headDim, 0, ctx-1, scale, 2.5, 0, 0, false,
				want[off:off+headDim])
		}
		onlineAttentionGroupI8(queries, keys8, values8, queryHeads,
			headDim, headDim, headDim, headDim, 0, ctx-1, scale, 2.5, got)
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
				t.Fatalf("queryHeads=%d output[%d] = %g, want %g (diff %g)", queryHeads, i, got[i], want[i], diff)
			}
		}
	}
}

// TestAttendHeadGroupsRangeMatchesPerHeadDispatch exercises attendHeadGroupsRange
// itself — the exact function ForwardBodyInto's groupedGQA branch calls — across
// all three KV cache formats and a kvMul that is NOT 4, so it covers both
// relaxations to the groupedGQA gate (model.go/forward_batch.go): kvMul > 1
// instead of kvMul == 4, and no format restriction instead of kvF32-only.
func TestAttendHeadGroupsRangeMatchesPerHeadDispatch(t *testing.T) {
	const (
		nHeads   = 6
		nKVHeads = 2 // kvMul = 3, deliberately not 4
		kvMul    = nHeads / nKVHeads
		headDim  = 8
		ctx      = 11
	)
	rng := rand.New(rand.NewSource(103))
	config := Config{NHeads: nHeads, NKVHeads: nKVHeads, HeadDim: headDim, ValueDim: headDim}

	newCache := map[string]func() *KVCache{
		"f32": func() *KVCache { return NewKVCache(1, nKVHeads*headDim, nKVHeads*headDim, ctx) },
		"f16": func() *KVCache { return NewKVCacheF16(1, nKVHeads*headDim, nKVHeads*headDim, ctx) },
		"i8":  func() *KVCache { return NewKVCacheI8(1, nKVHeads*headDim, nKVHeads*headDim, ctx) },
	}
	for name, build := range newCache {
		t.Run(name, func(t *testing.T) {
			cache := build()
			for pos := 0; pos < ctx; pos++ {
				k := randomVec(rng, nKVHeads*headDim)
				v := randomVec(rng, nKVHeads*headDim)
				cache.storeKV(0, pos, k, v)
			}
			q := randomVec(rng, nHeads*headDim)
			scale := float32(1 / math.Sqrt(headDim))

			want := make([]float32, nHeads*headDim)
			for h := 0; h < nHeads; h++ {
				off := h * headDim
				cache.attendHeadWithSink(0, h/kvMul, q[off:off+headDim], headDim, headDim,
					0, ctx-1, scale, 0, 0, 0, false, want[off:off+headDim])
			}

			buf := &DecodeBuffer{Q: q, AttnOut: make([]float32, nHeads*headDim)}
			attendHeadGroupsRange(&config, cache, buf, 0, ctx-1, 0, scale, kvMul, 0, nKVHeads)
			for i, got := range buf.AttnOut {
				if diff := math.Abs(float64(got - want[i])); diff > 1e-6 {
					t.Fatalf("%s: output[%d] = %g, want %g (diff %g)", name, i, got, want[i], diff)
				}
			}
		})
	}
}

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
			0, ctx-1, scale, 0, 0, layer.AttnSinks[h], true, want[off:off+headDim])
	}

	buf := &DecodeBuffer{Q: q, AttnOut: make([]float32, len(want))}
	parallelAttendHeads(config, layer, cache, buf, 0, ctx-1, 0, scale, 1)
	for i, got := range buf.AttnOut {
		if math.Abs(float64(got-want[i])) > 1e-6 {
			t.Fatalf("attention output[%d] = %g, want %g", i, got, want[i])
		}
	}
}
