//go:build darwin && cgo && metal

package gopherllm

import (
	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"
)

func TestMetalDenseDecodeParity(t *testing.T) {
	for _, option := range []string{"0", "1"} {
		t.Run("options_"+option, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_DENSE_VECTOR", option)
			t.Setenv("GOPHERLLM_METAL_DENSE_CONCURRENT", option)
			t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", option)
			testMetalDenseDecodeParity(t)
		})
	}
}
func testMetalDenseDecodeParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	old := metalDenseDecodeEnabled
	metalDenseDecodeEnabled = true
	t.Cleanup(func() { metalDenseDecodeEnabled = old })
	for _, arch := range []string{"ministral", "qwen3"} {
		t.Run(arch, func(t *testing.T) {
			const dim, hidden, heads, kvheads, maxlen = 256, 8192, 4, 1, 161
			rng := rand.New(rand.NewSource(982))
			makeWeight := func(rows, cols int, kind GGMLType) Weight {
				var raw []byte
				for range rows {
					var row []byte
					switch kind {
					case GGMLTypeQ4_K:
						row = randomQ4KRow(rng, cols)
					case GGMLTypeQ6_K:
						row = randomQ6KRow(rng, cols)
					default:
						row = randomQ8_0Row(rng, cols)
					}
					// Keep weights in the model-like range to avoid an ill-conditioned synthetic network.
					if kind == GGMLTypeQ4_K {
						for b := 0; b < len(row); b += 144 {
							row[b+1] = 0x08
							row[b+3] = 0x04
						}
					}
					if kind == GGMLTypeQ6_K {
						for b := 0; b < len(row); b += 210 {
							row[b+209] = 0x04
						}
					}
					if kind == GGMLTypeQ8_0 {
						for b := 0; b < len(row); b += 34 {
							row[b+1] = 0x18
						}
					}
					raw = append(raw, row...)
				}
				return Weight{Raw: raw, Rows: rows, Cols: cols, Type: kind}
			}
			typ := GGMLTypeQ4_K
			if arch == "qwen3" {
				typ = GGMLTypeQ8_0
			}
			ones := func(n int) []float32 {
				x := make([]float32, n)
				for i := range x {
					x[i] = 1
				}
				return x
			}
			c := Config{Arch: arch, Dim: dim, HiddenDim: hidden, NLayers: 2, NHeads: heads, NKVHeads: kvheads, HeadDim: 128, ValueDim: 128, KVDim: 128, KVMul: heads / kvheads, VocabSize: 4, RopeTheta: 10000, RopeDimensionCount: 128, RMSNormEps: 1e-5, EmbeddingScale: 1, ResidualScale: 1, LogitScale: 1}
			c.AttentionTemperatureScale = .1
			c.AttentionTemperatureFloor = 8
			emb := make([]float32, 4*dim)
			for i := range emb {
				emb[i] = float32(rng.NormFloat64()) * .2
			}
			w := ModelWeights{TokenEmbd: Weight{F32: emb, Rows: 4, Cols: dim}, OutputNorm: ones(dim)}
			for range 2 {
				downTyp := typ
				if arch == "ministral" && len(w.Layers) == 0 {
					downTyp = GGMLTypeQ6_K
				}
				l := LayerWeights{AttnNorm: ones(dim), FFNNorm: ones(dim), WQ: makeWeight(heads*128, dim, typ), WK: makeWeight(128, dim, typ), WV: makeWeight(128, dim, typ), WO: makeWeight(dim, heads*128, typ), W1: makeWeight(hidden, dim, typ), W3: makeWeight(hidden, dim, typ), W2: makeWeight(dim, hidden, downTyp)}
				if arch == "qwen3" {
					l.AttnQNorm = ones(128)
					l.AttnKNorm = ones(128)
				}
				w.Layers = append(w.Layers, l)
			}
			// Match the loader: absent biases can be allocated zero vectors.
			w.OutputNormBias = make([]float32, dim)
			for i := range w.Layers {
				w.Layers[i].BQ = make([]float32, heads*128)
				w.Layers[i].BO = make([]float32, dim)
			}
			gpu := w
			gpu.Layers = append([]LayerWeights(nil), w.Layers...)
			for i := range gpu.Layers {
				l := &gpu.Layers[i]
				l.W1.Metal = prepareMetalWeight(l.W1.Raw, l.W1.Type, l.W1.Rows, l.W1.Cols, false)
				if l.W1.Metal == nil {
					t.Fatal(MetalError())
				}
				defer releaseMetalWeight(l.W1.Metal)
			}
			gpu.Output = makeWeight(4, dim, typ)
			w.Output = gpu.Output
			// The public preparation thresholds exclude this deliberately tiny vocabulary.
			switch typ {
			case GGMLTypeQ4_K:
				gpu.Output.Metal = &MetalWeight{q4: metalbackend.PrepareQ4K(gpu.Output.Raw, 4, dim, false)}
			case GGMLTypeQ8_0:
				gpu.Output.Metal = &MetalWeight{q8: metalbackend.PrepareQ8_0(gpu.Output.Raw, 4, dim, false)}
			}
			defer releaseMetalWeight(gpu.Output.Metal)
			cpuCache := NewKVCache(2, 128, 128, maxlen)
			gpuCache := NewKVCache(2, 128, 128, maxlen)
			cpuBuf := NewDecodeBuffer(c, 128, 1, 128)
			gpuBuf := NewDecodeBuffer(c, 128, 1, 128)
			defer func() {
				if gpuBuf.metalDense != nil {
					gpuBuf.metalDense.close()
				}
			}()
			check := func(pos int) {
				ForwardBodyInto(c, w, cpuCache, cpuBuf, uint32(pos%4), pos)
				if !tryMetalDenseDecode(c, gpu, gpuCache, gpuBuf, uint32(pos%4), pos) {
					t.Fatal("dense GPU path failed:", MetalError())
				}
				if os.Getenv("GOPHERLLM_METAL_DENSE_SHARED_KV") == "1" && metalKVShareable(gpuCache.K[0]) && gpuBuf.metalDense.sharedK[0] == nil {
					t.Fatal("shared KV was not activated")
				}
				assertMetalFiniteClose(t, gpuBuf.XN, cpuBuf.XN)
				var got, want []float32
				ProjectLogitsInto(c, w, cpuBuf, &want)
				if !tryMetalDenseDecodeOutput(c, gpu, gpuCache, gpuBuf, uint32(pos%4), pos, &got) {
					t.Fatal("fused output failed")
				}
				assertMetalFiniteClose(t, got, want)
				recent := []uint32{1, 1, 3, 99999}
				for _, penalty := range []float32{1, 1.1, .7} {
					reference := append([]float32(nil), want...)
					applyRepeatPenalty(reference, recent, penalty)
					next, ok := tryMetalDenseGreedy(c, gpu, gpuCache, gpuBuf, uint32(pos%4), pos, recent, penalty)
					if !ok || next != argmaxFiniteToken(reference) {
						t.Fatalf("greedy penalty %g: got %d/%v want %d", penalty, next, ok, argmaxFiniteToken(reference))
					}
				}
				for l := range w.Layers {
					assertMetalFiniteClose(t, gpuCache.K[l][pos*128:(pos+1)*128], cpuCache.K[l][pos*128:(pos+1)*128])
					assertMetalFiniteClose(t, gpuCache.V[l][pos*128:(pos+1)*128], cpuCache.V[l][pos*128:(pos+1)*128])
				}
			}
			// Sequential decode, multi-part attention, cache edits and rewinds.
			for pos := 0; pos < 35; pos++ {
				check(pos)
			}
			for l := range w.Layers {
				for i := 35 * 128; i < 160*128; i++ {
					v := float32(math.Sin(float64(i))) * .1
					cpuCache.K[l][i] = v
					gpuCache.K[l][i] = v
					cpuCache.V[l][i] = v
					gpuCache.V[l][i] = v
				}
			}
			check(160)
			cpuCache.K[0][0] += .25
			gpuCache.K[0][0] += .25
			runtime.GC()
			check(12)
			// Replacing a public KV slice must invalidate a borrowed device buffer.
			gpuCache.K[0] = append([]float32(nil), gpuCache.K[0]...)
			check(13)
			c.SlidingWindow = 8
			check(35)
			gpu.Layers[0].BQ[0] = 0.1
			if metalDenseEligible(c, gpu, gpuCache) {
				t.Fatal("nonzero bias must fall back")
			}
			if tryMetalDenseDecode(c, gpu, gpuCache, gpuBuf, 0, 0) || gpuBuf.metalDense != nil {
				t.Fatal("unsupported configuration must release its device workspace")
			}
			gpu.Layers[0].BQ[0] = 0
			c.SWAPattern = []bool{true, false}
			if metalDenseEligible(c, gpu, gpuCache) {
				t.Fatal("mixed sliding-window patterns must fall back")
			}
			c.SWAPattern = nil
			if metalDenseEligible(c, gpu, NewKVCacheF16(2, 128, 128, maxlen)) {
				t.Fatal("F16 cache must fall back")
			}
		})
	}
}
