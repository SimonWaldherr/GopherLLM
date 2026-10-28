//go:build darwin && cgo && metal

package engine

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

// Keep an optional complete state export for comparing the same fixture across
// decoder implementations on one GPU, including every residual, hidden and KV
// float rather than only the final sampled token.
func TestMetalDenseResidualNormState(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	t.Setenv("GOPHERLLM_METAL_WHOLE_PREFILL", "1")
	var encoded bytes.Buffer
	appendFloats := func(values []float32) {
		for _, value := range values {
			if err := binary.Write(&encoded, binary.LittleEndian, math.Float32bits(value)); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, arch := range []string{"ministral", "qwen3", "llama"} {
		for _, dim := range []int{256, 3072} {
			t.Run(fmt.Sprintf("%s/dim_%d", arch, dim), func(t *testing.T) {
				c, cpu, gpu := metalResidualNormFixture(t, arch, dim)
				ck, cb := metalPrefillBuffer(t, c, 96)
				gk, gb := metalPrefillBuffer(t, c, 96)
				record := func(end int) {
					assertMetalFiniteClose(t, gb.X, cb.X)
					assertMetalFiniteClose(t, gb.XN, cb.XN)
					checkMetalPrefillKV(t, gk, ck, end)
					appendFloats(gb.X)
					appendFloats(gb.XN)
					for layer := range gk.K {
						appendFloats(gk.K[layer][:end*gk.PerPosKDim])
						appendFloats(gk.V[layer][:end*gk.PerPosVDim])
					}
				}
				for _, start := range []int{0, 17} {
					tokens := metalPrefillTokens(17)
					ForwardBatchInto(c, cpu, ck, cb, tokens, start, false, nil)
					ForwardBatchInto(c, gpu, gk, gb, tokens, start, false, nil)
					if gb.metalDense == nil || gb.metalDense.decoder.PrefillScratchBytes() == 0 {
						t.Fatal("whole Metal prefill inactive")
					}
					appendFloats(gb.batch.XFlat)
					record(start + len(tokens))
				}
				for _, pos := range []int{34, 9, 35} {
					ForwardBodyInto(c, cpu, ck, cb, uint32(pos%16), pos)
					if !tryMetalDenseDecode(c, gpu, gk, gb, uint32(pos%16), pos) {
						t.Fatal("dense decode inactive:", MetalError())
					}
					record(pos + 1)
				}
				// Verification writes a longer branch, then a rejected suffix is
				// overwritten by the serial continuation at the accepted boundary.
				tokens := []uint32{3, 5, 7, 11}
				if _, err := forwardBatchStatesIntoContext(context.Background(), c, cpu, ck, cb, tokens, 36); err != nil {
					t.Fatal(err)
				}
				var predictions []uint32
				states, ok, err := tryMetalDenseVerify(context.Background(), c, gpu, gk, gb, tokens, 36, make([][]uint32, len(tokens)), 1, &predictions)
				if !ok || err != nil || len(states) != len(tokens) {
					t.Fatalf("verification=%v, states=%d, err=%v", ok, len(states), err)
				}
				for i := range states {
					assertMetalFiniteClose(t, states[i], cb.batch.XN[i])
					appendFloats(states[i])
				}
				checkMetalPrefillKV(t, gk, ck, 40)
				for _, pos := range []int{38, 39, 40} {
					ForwardBodyInto(c, cpu, ck, cb, uint32((pos+3)%16), pos)
					if !tryMetalDenseDecode(c, gpu, gk, gb, uint32((pos+3)%16), pos) {
						t.Fatal("replay continuation inactive:", MetalError())
					}
					record(pos + 1)
				}
			})
		}
	}
	if t.Failed() {
		return
	}
	t.Logf("residual/hidden/KV bytes=%d sha256=%x", encoded.Len(), sha256.Sum256(encoded.Bytes()))
	if path := os.Getenv("GOPHERLLM_TEST_DENSE_STATE"); path != "" {
		if err := os.WriteFile(path, encoded.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

func metalResidualNormFixture(t testing.TB, arch string, dim int) (Config, ModelWeights, ModelWeights) {
	t.Helper()
	const hidden, heads, kvheads = 768, 4, 1
	rng := rand.New(rand.NewSource(91821))
	makeWeight := func(rows, cols int, typ GGMLType) Weight {
		var raw []byte
		for range rows {
			var row []byte
			switch typ {
			case GGMLTypeQ4_K:
				row = randomQ4KRow(rng, cols)
				for b := 0; b < len(row); b += 144 {
					row[b+1], row[b+3] = 0x04, 0x00
				}
			case GGMLTypeQ6_K:
				row = randomQ6KRow(rng, cols)
				for b := 0; b < len(row); b += 210 {
					row[b+209] = 0x00
				}
			case GGMLTypeQ8_0:
				row = randomQ8_0Row(rng, cols)
				for b := 0; b < len(row); b += 34 {
					row[b+1] = 0x10
				}
			}
			raw = append(raw, row...)
		}
		return Weight{Raw: raw, Rows: rows, Cols: cols, Type: typ}
	}
	ones := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = .9 + float32(i%13)*.01
		}
		return x
	}
	c := Config{Arch: arch, Dim: dim, HiddenDim: hidden, NLayers: 2, NHeads: heads, NKVHeads: kvheads, HeadDim: 128, ValueDim: 128, KVDim: 128, KVMul: 4, VocabSize: 16, RopeTheta: 10000, RopeDimensionCount: 128, RMSNormEps: 1e-5, EmbeddingScale: 1, ResidualScale: 1, LogitScale: 1, SlidingWindow: 7, SWAPattern: []bool{true, false}}
	emb := make([]float32, c.VocabSize*dim)
	for i := range emb {
		emb[i] = float32(rng.NormFloat64()) * .2
	}
	cpu := ModelWeights{TokenEmbd: Weight{F32: emb, Rows: c.VocabSize, Cols: dim}, OutputNorm: ones(dim), Output: makeWeight(c.VocabSize, dim, GGMLTypeQ8_0)}
	for range c.NLayers {
		l := LayerWeights{AttnNorm: ones(dim), FFNNorm: ones(dim), WQ: makeWeight(heads*128, dim, GGMLTypeQ4_K), WK: makeWeight(128, dim, GGMLTypeQ6_K), WV: makeWeight(128, dim, GGMLTypeQ8_0), WO: makeWeight(dim, heads*128, GGMLTypeQ4_K), W1: makeWeight(hidden, dim, GGMLTypeQ4_K), W3: makeWeight(hidden, dim, GGMLTypeQ4_K), W2: makeWeight(dim, hidden, GGMLTypeQ6_K)}
		if arch == "qwen3" {
			l.AttnQNorm, l.AttnKNorm = ones(128), ones(128)
		}
		cpu.Layers = append(cpu.Layers, l)
	}
	gpu := cpu
	gpu.Layers = append([]LayerWeights(nil), cpu.Layers...)
	for i := range gpu.Layers {
		l := &gpu.Layers[i]
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
