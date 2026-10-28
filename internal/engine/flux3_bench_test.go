package engine

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// These small graph benchmarks test integration and allocation changes, not
// throughput of the pretrained checkpoint. Setup and loading are untimed.
func BenchmarkFlux3Stages(b *testing.B) {
	m, _, _, _ := flux3Fixture(b)
	ctx := context.Background()
	text, err := m.encoder.encode(ctx, "A blue cube.")
	if err != nil {
		b.Fatal(err)
	}
	b.Run("encoder", func(b *testing.B) {
		prompt := strings.Repeat("A blue cube. ", 12)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if _, err := m.encoder.encode(ctx, prompt); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("dit_step", func(b *testing.B) {
		state, err := m.transformer.newState(text, 16, 16)
		if err != nil {
			b.Fatal(err)
		}
		latents := benchFloatSlice(16 * 16 * 128)
		if _, err = state.predict(ctx, latents, 0.9, 0.7); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if _, err := state.predict(ctx, latents, 0.9, 0.7); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("vae_256", func(b *testing.B) {
		latents := benchFloatSlice(16 * 16 * 128)
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			if _, err := m.vae.decode(ctx, latents, 16, 16); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkImageUpsampleConv(b *testing.B) {
	for _, shape := range [][3]int{{64, 32, 32}, {256, 128, 128}} {
		for _, fused := range []bool{false, true} {
			b.Run(fmt.Sprintf("channels=%d/grid=%dx%d/fused=%t", shape[0], shape[1], shape[2], fused), func(b *testing.B) {
				c, h, w := shape[0], shape[1], shape[2]
				conv := qi21VAEConv{weight: benchFloatSlice(c * c * 9), bias: benchFloatSlice(c), in: c, out: c, kernel: 3}
				x := diffusionTensor{benchFloatSlice(c * h * w), c, h, w}
				s := &qi21VAEWorkspace{ctx: context.Background()}
				run := conv.upsample
				if !fused {
					run = func(s *qi21VAEWorkspace, x diffusionTensor) diffusionTensor {
						return conv.forward(s, imageUpsampleReference(x))
					}
				}
				run(s, x)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					run(s, x)
					if s.err != nil {
						b.Fatal(s.err)
					}
				}
			})
		}
	}
}

func BenchmarkFlux3TextAttention(b *testing.B) {
	for _, real := range []int{160, 512} {
		for _, batched := range []bool{false, true} {
			b.Run(fmt.Sprintf("tokens=%d/batched=%t", real, batched), func(b *testing.B) {
				const heads, kvHeads, dim = 32, 8, 128
				n := real + 1
				q, k, v := benchFloatSlice(n*heads*dim), benchFloatSlice(real*kvHeads*dim), benchFloatSlice(real*kvHeads*dim)
				out := make([]float32, len(q))
				var scores []float32
				ctx := context.Background()
				if batched && !imageGroupedAttentionBLAS(ctx, q, k, v, out, n, real, heads, kvHeads, dim, 0, real, &scores) {
					b.Skip("Accelerate unavailable")
				}
				run := func() {
					if batched {
						imageGroupedAttention(ctx, q, k, v, out, n, real, heads, kvHeads, dim, 0, real, &scores)
					} else {
						imageGroupedAttentionOnline(ctx, q, k, v, out, n, real, heads, kvHeads, dim, 0, real)
					}
				}
				run()
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					run()
				}
			})
		}
	}
}
