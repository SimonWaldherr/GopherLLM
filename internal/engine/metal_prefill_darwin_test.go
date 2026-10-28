//go:build darwin && cgo && metal

package engine

import (
	"context"
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func metalPrefillFixture(t testing.TB, arch string) (Config, ModelWeights, ModelWeights) {
	t.Helper()
	const dim, hidden, heads, kvheads = 256, 512, 4, 1
	rng := rand.New(rand.NewSource(14117))
	makeWeight := func(rows, cols int, typ GGMLType) Weight {
		raw := make([]byte, 0)
		for range rows {
			var row []byte
			switch typ {
			case GGMLTypeQ4_K:
				row = randomQ4KRow(rng, cols)
				for b := 0; b < len(row); b += 144 {
					row[b+1] = 0x08
					row[b+3] = 0x04
				}
			case GGMLTypeQ6_K:
				row = randomQ6KRow(rng, cols)
				for b := 0; b < len(row); b += 210 {
					row[b+209] = 0x04
				}
			case GGMLTypeQ8_0:
				row = randomQ8_0Row(rng, cols)
				for b := 0; b < len(row); b += 34 {
					row[b+1] = 0x18
				}
			}
			raw = append(raw, row...)
		}
		return Weight{Raw: raw, Rows: rows, Cols: cols, Type: typ}
	}
	ones := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = 1
		}
		return x
	}
	c := Config{Arch: arch, Dim: dim, HiddenDim: hidden, NLayers: 2, NHeads: heads, NKVHeads: kvheads, HeadDim: 128, ValueDim: 128, KVDim: 128, KVMul: 4, VocabSize: 16, RopeTheta: 10000, RopeDimensionCount: 128, RMSNormEps: 1e-5, EmbeddingScale: 1, ResidualScale: 1, LogitScale: 1, AttentionTemperatureScale: .1, AttentionTemperatureFloor: 8}
	emb := make([]float32, c.VocabSize*dim)
	for i := range emb {
		emb[i] = float32(rng.NormFloat64()) * .2
	}
	cpu := ModelWeights{TokenEmbd: Weight{F32: emb, Rows: c.VocabSize, Cols: dim}, OutputNorm: ones(dim), Output: makeWeight(c.VocabSize, dim, GGMLTypeQ8_0)}
	for i := 0; i < c.NLayers; i++ {
		l := LayerWeights{AttnNorm: ones(dim), FFNNorm: ones(dim), WQ: makeWeight(heads*128, dim, GGMLTypeQ4_K), WK: makeWeight(128, dim, GGMLTypeQ6_K), WV: makeWeight(128, dim, GGMLTypeQ8_0), WO: makeWeight(dim, heads*128, GGMLTypeQ4_K), W1: makeWeight(hidden, dim, GGMLTypeQ4_K), W3: makeWeight(hidden, dim, GGMLTypeQ4_K), W2: makeWeight(dim, hidden, GGMLTypeQ6_K)}
		if arch == "qwen3" {
			l.AttnQNorm = ones(128)
			l.AttnKNorm = ones(128)
		}
		cpu.Layers = append(cpu.Layers, l)
	}
	gpu := cpu
	gpu.Layers = append([]LayerWeights(nil), cpu.Layers...)
	for i := range gpu.Layers {
		l := &gpu.Layers[i]
		// The synthetic graph is deliberately below the loader's offload thresholds.
		l.W1.Metal = &MetalWeight{q4: metalbackend.PrepareQ4K(l.W1.Raw, l.W1.Rows, l.W1.Cols, false)}
		if l.W1.Metal.q4 == nil {
			t.Fatal(MetalError())
		}
		weight := l.W1.Metal
		t.Cleanup(func() { releaseMetalWeight(weight) })
	}
	gpu.Output.Metal = &MetalWeight{q8: metalbackend.PrepareQ8_0(gpu.Output.Raw, c.VocabSize, dim, false)}
	t.Cleanup(func() { releaseMetalWeight(gpu.Output.Metal) })
	return c, cpu, gpu
}

func metalPrefillBuffer(t testing.TB, c Config, maxlen int) (*KVCache, *DecodeBuffer) {
	t.Helper()
	k := NewKVCache(c.NLayers, c.KVDim, c.NKVHeads*c.ValueDim, maxlen)
	b := NewDecodeBuffer(c, c.HeadDim, c.NKVHeads, c.ValueDim)
	t.Cleanup(func() {
		if b.metalDense != nil {
			b.metalDense.close()
		}
	})
	return k, b
}
func metalPrefillTokens(n int) []uint32 {
	x := make([]uint32, n)
	for i := range x {
		x[i] = uint32(i % 16)
	}
	return x
}
func checkMetalPrefillKV(t *testing.T, got, want *KVCache, end int) {
	t.Helper()
	for i := range got.K {
		assertMetalFiniteClose(t, got.K[i][:end*got.PerPosKDim], want.K[i][:end*want.PerPosKDim])
		assertMetalFiniteClose(t, got.V[i][:end*got.PerPosVDim], want.V[i][:end*want.PerPosVDim])
	}
}

func TestMetalWholePrefillParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	for _, shared := range []string{"0", "1"} {
		for _, arch := range []string{"ministral", "qwen3", "llama"} {
			t.Run(shared+"/"+arch, func(t *testing.T) {
				t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
				c, cpu, gpu := metalPrefillFixture(t, arch)
				c.SlidingWindow = 7
				c.SWAPattern = []bool{true, false}
				ck, cb := metalPrefillBuffer(t, c, 96)
				gk, gb := metalPrefillBuffer(t, c, 96)
				for _, step := range []struct{ start, n int }{{0, 17}, {17, 33}, {9, 17}} {
					tokens := metalPrefillTokens(step.n)
					var want, got []float32
					ForwardBatchInto(c, cpu, ck, cb, tokens, step.start, true, &want)
					if err := forwardBatchIntoContext(context.Background(), c, gpu, gk, gb, tokens, step.start, true, &got, nil); err != nil {
						t.Fatal(err)
					}
					if gb.metalDense == nil || gb.metalDense.decoder.PrefillScratchBytes() == 0 {
						t.Fatal("whole GPU prefill did not execute")
					}
					if shared == "1" && gb.metalDense.sharedK[0] == nil {
						t.Fatal("shared KV inactive")
					}
					assertMetalFiniteClose(t, got, want)
					assertMetalFiniteClose(t, gb.XN, cb.XN)
					checkMetalPrefillKV(t, gk, ck, step.start+step.n)
				}
				// A CPU continuation after whole-prefill consumes every published F32 KV row.
				pos := 26
				var want, got []float32
				ForwardInto(c, cpu, ck, cb, 3, pos, &want)
				ForwardInto(c, cpu, gk, gb, 3, pos, &got)
				assertMetalFiniteClose(t, got, want)
				next, ok := tryMetalDenseGreedy(c, gpu, gk, gb, 3, pos, nil, 1)
				if !ok || next != argmaxFiniteToken(want) {
					t.Fatalf("greedy=%d/%v want %d", next, ok, argmaxFiniteToken(want))
				}
			})
		}
	}
}

func TestMetalWholePrefillRestoredPrefix(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	for _, shared := range []string{"0", "1"} {
		t.Run("shared_"+shared, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
			c, cpu, gpu := metalPrefillFixture(t, "ministral")
			ck, cb := metalPrefillBuffer(t, c, 96)
			gk, gb := metalPrefillBuffer(t, c, 96)
			const prefix = 19
			// Same pointers and same prefix length, but alternating restored branch contents.
			for _, branch := range []int{0, 1, 0} {
				for l := range ck.K {
					for i := 0; i < prefix*128; i++ {
						v := float32((i+branch*13+l*5)%23-11) * .03
						ck.K[l][i] = v
						gk.K[l][i] = v
						ck.V[l][i] = v * .7
						gk.V[l][i] = v * .7
					}
				}
				for _, step := range []struct {
					n     int
					whole string
				}{{17, "0"}, {17, "1"}, {7, "1"}} {
					n := step.n
					t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", step.whole)
					var want, got []float32
					tokens := metalPrefillTokens(n)
					ForwardBatchInto(c, cpu, ck, cb, tokens, prefix, true, &want)
					ForwardBatchInto(c, gpu, gk, gb, tokens, prefix, true, &got)
					assertMetalFiniteClose(t, got, want)
					checkMetalPrefillKV(t, gk, ck, prefix+n)
				}
			}
		})
	}
}

// A deterministic cancellation at PrefillBatch's post-submission check tests
// the transaction boundary without timing assumptions or a production hook.
type prefillCancelContext struct {
	context.Context
	calls int
}

func (c *prefillCancelContext) Err() error {
	c.calls++
	if c.calls >= 3 {
		return context.Canceled
	}
	return nil
}
func TestMetalWholePrefillCancellationAndBounds(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	for _, shared := range []string{"0", "1"} {
		t.Run(shared, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
			c, cpu, gpu := metalPrefillFixture(t, "ministral")
			k, b := metalPrefillBuffer(t, c, 64)
			ForwardBatchInto(c, gpu, k, b, metalPrefillTokens(17), 0, false, nil)
			if b.metalDense == nil || b.metalDense.decoder.PrefillScratchBytes() == 0 {
				t.Fatal("GPU prefill inactive")
			}
			beforeK, beforeV := make([][]float32, c.NLayers), make([][]float32, c.NLayers)
			for l := range k.K {
				beforeK[l] = append([]float32(nil), k.K[l]...)
				beforeV[l] = append([]float32(nil), k.V[l]...)
			}
			residual, output := make([]float32, 17*c.Dim), make([]float32, 17*c.Dim)
			for i := range residual {
				residual[i] = 123
				output[i] = -456
			}
			canceled := &prefillCancelContext{Context: context.Background()}
			ok, err := b.metalDense.decoder.PrefillBatch(canceled, b.batch.XFlat, b.batch.RopeSinFlat, b.batch.RopeCosFlat, b.batch.RopeTemperature, 17, 17, false, true, k.K, k.V, residual, output)
			if ok || err != context.Canceled {
				t.Fatalf("ok=%v err=%v", ok, err)
			}
			if !reflect.DeepEqual(k.K, beforeK) || !reflect.DeepEqual(k.V, beforeV) {
				t.Fatal("canceled GPU work published KV")
			}
			for i := range residual {
				if residual[i] != 123 || output[i] != -456 {
					t.Fatal("canceled GPU work published outputs")
				}
			}
			for _, bounds := range [][2]int{{-1, 17}, {48, 17}, {0, 1}, {0, 257}} {
				ok, err = b.metalDense.decoder.PrefillBatch(context.Background(), make([]float32, 257*c.Dim), make([]float32, 257*64), make([]float32, 257*64), make([]float32, 257), bounds[0], bounds[1], false, true, k.K, k.V, make([]float32, 257*c.Dim), make([]float32, 257*c.Dim))
				if ok || err != nil {
					t.Fatalf("invalid bounds %v accepted: %v", bounds, err)
				}
			}
			if shared == "1" {
				wrongKeys := append([][]float32(nil), k.K...)
				wrongKeys[0] = append([]float32(nil), k.K[0]...)
				ok, err = b.metalDense.decoder.PrefillBatch(context.Background(), b.batch.XFlat, b.batch.RopeSinFlat, b.batch.RopeCosFlat, b.batch.RopeTemperature, 17, 17, false, true, wrongKeys, k.V, residual, output)
				if ok || err != nil {
					t.Fatal("replaced shared KV accepted")
				}
			}
			if !reflect.DeepEqual(k.K, beforeK) || !reflect.DeepEqual(k.V, beforeV) {
				t.Fatal("declined GPU work published KV")
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if err := forwardBatchIntoContext(ctx, c, gpu, k, b, metalPrefillTokens(17), 0, true, nil, nil); err != context.Canceled {
				t.Fatal(err)
			}
			ck, cb := metalPrefillBuffer(t, c, 64)
			for l := range ck.K {
				copy(ck.K[l], beforeK[l])
				copy(ck.V[l], beforeV[l])
			}
			var want, got []float32
			ForwardBatchInto(c, cpu, ck, cb, metalPrefillTokens(17), 17, true, &want)
			ForwardBatchInto(c, gpu, k, b, metalPrefillTokens(17), 17, true, &got)
			assertMetalFiniteClose(t, got, want)
			checkMetalPrefillKV(t, k, ck, 34)
		})
	}
}

func TestMetalWholePrefillTiledAndScratchLimit(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	c, cpu, gpu := metalPrefillFixture(t, "ministral")
	c.SlidingWindow = 7
	ck, cb := metalPrefillBuffer(t, c, 16384)
	gk, gb := metalPrefillBuffer(t, c, 16384)
	// Long absolute position forces the hypothetical full-context partial slab
	// beyond 64 MiB; SWA keeps this regression's actual arithmetic small.
	const start, batch = 16128, 256
	var want, got []float32
	tokens := metalPrefillTokens(batch)
	ForwardBatchInto(c, cpu, ck, cb, tokens, start, true, &want)
	ForwardBatchInto(c, gpu, gk, gb, tokens, start, true, &got)
	if gb.metalDense == nil {
		t.Fatal("GPU path inactive")
	}
	bytes := gb.metalDense.decoder.PrefillScratchBytes()
	if bytes < 64<<20 || bytes > 256<<20 {
		t.Fatalf("unexpected scratch %d", bytes)
	}
	assertMetalFiniteClose(t, got, want)
	checkMetalPrefillKV(t, gk, ck, start+batch)
	old := metalBatchAttentionTiled
	metalBatchAttentionTiled = false
	t.Cleanup(func() { metalBatchAttentionTiled = old })
	ForwardBatchInto(c, gpu, gk, gb, tokens[:17], 17, true, &got)
	ForwardBatchInto(c, cpu, ck, cb, tokens[:17], 17, true, &want)
	assertMetalFiniteClose(t, got, want)
}

func BenchmarkMetalWholePrefill(b *testing.B) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	c, _, gpu := metalPrefillFixture(b, "ministral")
	for _, n := range []int{32, 128} {
		for _, whole := range []string{"0", "1"} {
			b.Run(fmt.Sprintf("P%d/whole_%s", n, whole), func(b *testing.B) {
				b.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", whole)
				k, buf := metalPrefillBuffer(b, c, 256)
				tokens := metalPrefillTokens(n)
				ForwardBatchInto(c, gpu, k, buf, tokens, 0, false, nil)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					ForwardBatchInto(c, gpu, k, buf, tokens, 0, false, nil)
				}
			})
		}
	}
}

func TestMetalWholePrefillPooling(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	c, cpu, gpu := metalPrefillFixture(t, "ministral")
	ck, cb := metalPrefillBuffer(t, c, 64)
	gk, gb := metalPrefillBuffer(t, c, 64)
	want, got := make([]float32, c.Dim), make([]float32, c.Dim)
	tokens := metalPrefillTokens(33)
	forwardBatchPoolInto(c, cpu, ck, cb, tokens, 0, want)
	forwardBatchPoolInto(c, gpu, gk, gb, tokens, 0, got)
	if gb.metalDense == nil || gb.metalDense.decoder.PrefillScratchBytes() == 0 {
		t.Fatal("whole GPU path inactive")
	}
	assertMetalFiniteClose(t, got, want)
	assertMetalFiniteClose(t, gb.batch.XFlat, cb.batch.XFlat)
	assertMetalFiniteClose(t, gb.batch.XNFlat, cb.batch.XNFlat)
}
