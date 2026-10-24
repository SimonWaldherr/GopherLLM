//go:build darwin && cgo && metal

package gopherllm

import (
	"fmt"
	"os"
	"testing"
	"time"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func sharedFFNWeight(t testing.TB, kind GGMLType, rows, cols int) *metalbackend.Weight {
	t.Helper()
	w := qwen35InputWeight(t, kind, rows, cols)
	var p *metalbackend.Weight
	switch kind {
	case GGMLTypeQ4_K:
		p = metalbackend.PrepareQ4K(w.Raw, rows, cols, false)
	case GGMLTypeQ6_K:
		p = metalbackend.PrepareQ6K(w.Raw, rows, cols, false)
	case GGMLTypeQ8_0:
		p = metalbackend.PrepareQ8_0(w.Raw, rows, cols, false)
	}
	if p == nil {
		t.Fatal(MetalError())
	}
	t.Cleanup(func() { metalbackend.Release(p) })
	return p
}

func TestMetalSharedFFNInplace(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ8_0} {
		t.Run(typ.String(), func(t *testing.T) {
			g := sharedFFNWeight(t, typ, 768, 256)
			u := sharedFFNWeight(t, typ, 768, 256)
			downType := GGMLTypeQ6_K
			if typ == GGMLTypeQ8_0 {
				downType = typ
			}
			d := sharedFFNWeight(t, downType, 257, 768)
			for _, decode := range []bool{false, true} {
				for _, n := range []int{1, 3, 17, 32, 33} {
					x := randomExpertInput(n*256, 72)
					var reference []float32
					out := make([]float32, n*257)
					for _, mode := range []string{"0", "1"} {
						t.Setenv("GOPHERLLM_METAL_FFN_INPLACE", mode)
						t.Setenv("GOPHERLLM_METAL_QWEN_INPLACE_SILU", "1")
						var ok bool
						if typ == GGMLTypeQ4_K {
							if decode {
								ok = metalbackend.MatvecQ4K2SwiGLUDecodeBatch(g, u, d, x, out, n, false)
							} else {
								ok = metalbackend.MatvecQ4K2SwiGLUQ6KBatch(g, u, d, x, out, n)
							}
						} else {
							if decode {
								ok = metalbackend.MatvecQ8_0SwiGLUDecodeBatch(g, u, d, x, out, n)
							} else {
								ok = metalbackend.MatvecQ8_0SwiGLUBatch(g, u, d, x, out, n)
							}
						}
						if !ok {
							t.Fatalf("FFN declined: %s", MetalError())
						}
						if mode == "0" {
							reference = append([]float32(nil), out...)
						} else {
							requireMoEPrefillExact(t, out, reference)
						}
					}
				}
			}
		})
	}
}

// Run in a fresh process per mode; shared buffers retain their high-water mark.
func TestMetalFFNWorkspace(t *testing.T) {
	kind := os.Getenv("GOPHERLLM_TEST_FFN_KIND")
	if kind == "" {
		t.Skip("set GOPHERLLM_TEST_FFN_KIND=ministral|qwen|gemma")
	}
	if !MetalAvailable() {
		t.Fatal(MetalError())
	}
	dim, hidden, typ := 3072, 9216, GGMLTypeQ4_K
	switch kind {
	case "ministral":
	case "qwen":
		dim, hidden, typ = 2560, 9216, GGMLTypeQ8_0
	case "gemma":
		dim, hidden = 1536, 6144
	default:
		t.Fatal("unknown FFN kind")
	}
	downType := GGMLTypeQ6_K
	if typ == GGMLTypeQ8_0 {
		downType = typ
	}
	g, u, d := sharedFFNWeight(t, typ, hidden, dim), sharedFFNWeight(t, typ, hidden, dim), sharedFFNWeight(t, downType, dim, hidden)
	x, out := randomExpertInput(32*dim, 42), make([]float32, 32*dim)
	run := func(n int) {
		var ok bool
		if kind == "gemma" {
			ok = metalbackend.GeluFFN(g, u, d, [3]uint32{4, 4, 6}, x[:n*dim], out[:n*dim], n)
		} else if kind == "qwen" {
			ok = metalbackend.MatvecQ8_0SwiGLUDecodeBatch(g, u, d, x[:n*dim], out[:n*dim], n)
		} else {
			ok = metalbackend.MatvecQ4K2SwiGLUQ6KBatch(g, u, d, x[:n*dim], out[:n*dim], n)
		}
		if !ok {
			t.Fatal(MetalError())
		}
	}
	run(32)
	fmt.Printf("FFN_WORKSPACE kind=%s inplace=%s bytes=%d digest=%s\n", kind, os.Getenv("GOPHERLLM_METAL_FFN_INPLACE"), metalbackend.BatchScratchBytes(), ministralDecodeFloatDigest(t, out))
	if os.Getenv("GOPHERLLM_TEST_FFN_PAIRS") == "1" {
		for _, n := range []int{1, 32} {
			var expected string
			for pair := -2; pair < 32; pair++ {
				var elapsed [2]time.Duration
				for _, mode := range []int{0, 1, 1, 0} {
					t.Setenv("GOPHERLLM_METAL_FFN_INPLACE", fmt.Sprint(mode))
					start := time.Now()
					run(n)
					elapsed[mode] += time.Since(start) / 2
					digest := ministralDecodeFloatDigest(t, out[:n*dim])
					if expected != "" && expected != digest {
						t.Fatal("FFN output changed")
					}
					expected = digest
				}
				if pair >= 0 {
					fmt.Printf("FFN_PAIR kind=%s batch=%d pair=%d baseline_ns=%d inplace_ns=%d\n", kind, n, pair, elapsed[0].Nanoseconds(), elapsed[1].Nanoseconds())
				}
			}
		}
	}
}
