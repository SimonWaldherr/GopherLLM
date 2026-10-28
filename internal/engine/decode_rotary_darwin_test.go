//go:build darwin && cgo && metal

package engine

import (
	"fmt"
	"testing"
)

// Q/K normalization selects different input and output buffers for rotary
// embedding. Exercise both independent selections, both rotary layouts, and
// cache rewinds across sliding-window and temperature boundaries.
func TestMetalDenseRotaryCacheRanges(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	old := metalDenseDecodeEnabled
	metalDenseDecodeEnabled = true
	t.Cleanup(func() { metalDenseDecodeEnabled = old })
	for _, shared := range []string{"0", "1"} {
		for _, arch := range []string{"ministral", "qwen3"} {
			for mask := 0; mask < 4; mask++ {
				t.Run(fmt.Sprintf("shared_%s/%s/norm_%d", shared, arch, mask), func(t *testing.T) {
					t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", shared)
					c, cpu, gpu := metalPrefillFixture(t, arch)
					c.SlidingWindow, c.SWAPattern = 7, []bool{true, false}
					for i := range cpu.Layers {
						var qnorm, knorm []float32
						if mask&1 != 0 {
							qnorm = make([]float32, 128)
							for j := range qnorm {
								qnorm[j] = .9 + float32(j%17)*.01
							}
						}
						if mask&2 != 0 {
							knorm = make([]float32, 128)
							for j := range knorm {
								knorm[j] = .8 + float32(j%19)*.01
							}
						}
						cpu.Layers[i].AttnQNorm, gpu.Layers[i].AttnQNorm = qnorm, qnorm
						cpu.Layers[i].AttnKNorm, gpu.Layers[i].AttnKNorm = knorm, knorm
					}
					ck, cb := metalPrefillBuffer(t, c, 96)
					gk, gb := metalPrefillBuffer(t, c, 96)
					for _, pos := range []int{0, 1, 2, 7, 8, 31, 32, 33, 9, 63, 95} {
						token := uint32(pos % c.VocabSize)
						ForwardBodyInto(c, cpu, ck, cb, token, pos)
						if !tryMetalDenseDecode(c, gpu, gk, gb, token, pos) {
							t.Fatal("dense decode inactive:", MetalError())
						}
						assertMetalFiniteClose(t, gb.X, cb.X)
						assertMetalFiniteClose(t, gb.XN, cb.XN)
						checkMetalPrefillKV(t, gk, ck, pos+1)
					}
				})
			}
		}
	}
}

func TestMetalDenseDecodeRejectsResizedSharedCache(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	t.Setenv("GOPHERLLM_METAL_DENSE_SHARED_KV", "1")
	forceExactMetalReference(t)
	old := metalDenseDecodeEnabled
	metalDenseDecodeEnabled = true
	t.Cleanup(func() { metalDenseDecodeEnabled = old })
	for _, key := range []bool{true, false} {
		for _, shape := range []string{"short_row", "empty_row", "fewer_layers"} {
			t.Run(fmt.Sprintf("key_%t/%s", key, shape), func(t *testing.T) {
				c, cpu, gpu := metalPrefillFixture(t, "ministral")
				cache, buf := metalPrefillBuffer(t, c, 16)
				refCache, refBuf := metalPrefillBuffer(t, c, 16)
				ForwardBodyInto(c, cpu, refCache, refBuf, 1, 0)
				if !tryMetalDenseDecode(c, gpu, cache, buf, 1, 0) || buf.metalDense.sharedK[0] == nil {
					t.Fatal("shared decode inactive:", MetalError())
				}
				bound := buf.metalDense
				rows := &cache.V
				if key {
					rows = &cache.K
				}
				layers, row := *rows, (*rows)[0]
				switch shape {
				case "short_row":
					(*rows)[0] = row[:len(row)-1]
				case "empty_row":
					(*rows)[0] = row[:0]
				case "fewer_layers":
					*rows = layers[:len(layers)-1]
				}
				if tryMetalDenseDecode(c, gpu, cache, buf, 2, 1) || buf.metalDense != nil || bound.decoder != nil {
					t.Fatal("decode reused a resized shared cache binding")
				}
				*rows, layers[0] = layers, row
				ForwardBodyInto(c, cpu, refCache, refBuf, 2, 1)
				if !tryMetalDenseDecode(c, gpu, cache, buf, 2, 1) {
					t.Fatal("repaired cache did not rebind:", MetalError())
				}
				assertMetalFiniteClose(t, buf.XN, refBuf.XN)
				checkMetalPrefillKV(t, cache, refCache, 2)
			})
		}
	}
}
