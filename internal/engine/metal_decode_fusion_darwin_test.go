//go:build darwin && cgo && metal

package engine

import (
	"fmt"
	"math"
	"testing"
)

func TestMetalDenseFusedAttentionParity(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	for _, arch := range []string{"ministral", "qwen3", "llama"} {
		for _, dim := range []int{256, 3072} {
			t.Run(fmt.Sprintf("%s/%d", arch, dim), func(t *testing.T) {
				cfg, _, w := metalResidualNormFixture(t, arch, dim)
				k, b := metalPrefillBuffer(t, cfg, 600)
				for _, concurrent := range []string{"0", "1"} {
					t.Setenv("GOPHERLLM_METAL_DENSE_CONCURRENT", concurrent)
					for _, pos := range []int{0, 1, 31, 32, 63, 64, 95, 96, 127, 128, 129, 159, 160, 191, 192, 223, 224, 255, 256, 257, 287, 288, 319, 320, 351, 352, 383, 384, 415, 416, 447, 448, 479, 480, 511, 512, 513} {
						var want string
						for _, mode := range []string{"0", "1"} {
							t.Setenv("GOPHERLLM_METAL_DENSE_FUSED_ATTENTION", mode)
							if !tryMetalDenseDecode(cfg, w, k, b, uint32(pos%15+1), pos) {
								t.Fatal("dense decode unavailable", MetalError())
							}
							values := [][]float32{b.X, b.XN}
							for layer := range k.K {
								values = append(values, k.K[layer], k.V[layer])
							}
							got := ministralDecodeFloatDigest(t, values...)
							if mode == "0" {
								want = got
							} else if want != got {
								t.Fatal("residual, normalization or KV bits changed")
							}
						}
					}
				}
			})
		}
	}
}

func TestMetalQwenInplaceActivation(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	t.Setenv("GOPHERLLM_METAL_FFN_INPLACE", "1")
	const dim, hidden = 5120, 17408
	gate := qwen35InputWeight(t, GGMLTypeQ4_K, hidden, dim)
	up := qwen35InputWeight(t, GGMLTypeQ4_K, hidden, dim)
	// Different gate/up rows expose accidental overwrites or buffer mixups.
	rowBytes, _ := up.Type.DataSize(dim)
	first := append([]byte(nil), up.Raw[:rowBytes]...)
	copy(up.Raw, up.Raw[rowBytes:])
	copy(up.Raw[len(up.Raw)-rowBytes:], first)
	for _, downType := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		t.Run(downType.String(), func(t *testing.T) {
			down := qwen35InputWeight(t, downType, dim, hidden)
			for _, w := range []*Weight{&gate, &up, &down} {
				w.Metal = prepareMetalWeight(w.Raw, w.Type, w.Rows, w.Cols, false)
				if w.Metal == nil {
					t.Fatal(MetalError())
				}
				handle := w.Metal
				t.Cleanup(func() { releaseMetalWeight(handle) })
			}
			input := randomExpertInput(32*dim, 51191)
			for _, vector := range []string{"0", "1"} {
				t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", vector)
				for _, n := range []int{1, 2, 3, 4, 8, 17, 32} {
					var want, got []float32
					for _, mode := range []string{"0", "1"} {
						t.Setenv("GOPHERLLM_METAL_QWEN_INPLACE_SILU", mode)
						if !matvecMetalSwiGLUDecodeBatchInto(gate.Metal, up.Metal, down.Metal, input[:n*dim], n, &got) {
							t.Fatal("FFN declined")
						}
						if mode == "0" {
							want = append([]float32(nil), got...)
							continue
						}
						for i, v := range want {
							if !finite32(v) || math.Float32bits(v) != math.Float32bits(got[i]) {
								t.Fatalf("P%d output%d: %g vs %g", n, i, v, got[i])
							}
						}
					}
				}
			}
		})
	}
}
