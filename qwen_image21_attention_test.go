package gopherllm

import (
	"context"
	"fmt"
	"slices"
	"testing"
)

func qi21AttentionTestState(prefix, height, width, heads, dim int) *qi21TransformerState {
	total, stride := prefix+height*width, heads*dim
	return &qi21TransformerState{
		model:  &qi21Transformer{dim: stride, cfg: qi21TransformerConfig{Heads: heads, HeadDim: dim}},
		prefix: prefix, height: height, width: width,
		q: benchFloatSlice(total * stride), k: benchFloatSlice(total * stride), v: benchFloatSlice(total * stride),
		attn: make([]float32, total*stride),
	}
}

func TestQwenImage21AttentionMaskAndCache(t *testing.T) {
	for _, shape := range [][5]int{{3, 2, 2, 2, 8}, {5, 4, 4, 3, 16}, {17, 17, 17, 3, 16}, {32, 16, 16, 11, 128}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			s := qi21AttentionTestState(shape[0], shape[1], shape[2], shape[3], shape[4])
			n, stride := s.prefix+s.height*s.width, s.model.dim
			ctx := context.Background()
			s.attentionOnline(ctx, n, 0)
			want := slices.Clone(s.attn)
			guarded := make([]float32, len(want)+2)
			guarded[0], guarded[len(guarded)-1] = 123, -456
			s.attn = guarded[1 : len(guarded)-1]
			s.attention(ctx, n, 0)
			qi21AssertNear(t, s.attn, want, 5e-6)
			if guarded[0] != 123 || guarded[len(guarded)-1] != -456 || cap(s.scores) > 1<<20 {
				t.Fatal("attention exceeded its output or score bounds")
			}
			full := slices.Clone(s.attn)
			queries := s.q
			s.q, s.attn = s.q[s.prefix*stride:], s.attn[s.prefix*stride:]
			s.attention(ctx, n-s.prefix, s.prefix)
			qi21AssertNear(t, s.attn, want[s.prefix*stride:], 5e-6)
			// Prefix queries cannot observe any image keys or values.
			s.q, s.attn = queries, guarded[1:len(guarded)-1]
			for i := s.prefix * stride; i < len(s.k); i++ {
				s.k[i] += 100
				s.v[i] -= 100
			}
			s.attention(ctx, n, 0)
			qi21AssertNear(t, s.attn[:s.prefix*stride], full[:s.prefix*stride], 5e-6)
			ctx, cancel := context.WithCancel(ctx)
			cancel()
			clear(s.attn)
			s.attention(ctx, n, 0)
			for _, v := range s.attn {
				if v != 0 {
					t.Fatal("canceled attention wrote output")
				}
			}
		})
	}
}

func TestQwenImage21AttentionScoreBound(t *testing.T) {
	s := qi21AttentionTestState(32, 64, 64, 2, 8)
	const queries = 300
	offset := s.prefix + s.height*s.width - queries
	s.q, s.attn = s.q[offset*s.model.dim:], s.attn[offset*s.model.dim:]
	s.attentionOnline(context.Background(), queries, offset)
	want := slices.Clone(s.attn)
	clear(s.attn)
	s.attention(context.Background(), queries, offset)
	qi21AssertNear(t, s.attn, want, 5e-6)
	if cap(s.scores) > 1<<20 {
		t.Fatal("large key set exceeded the four-MiB score budget")
	}
}

func BenchmarkQwenImage21Attention(b *testing.B) {
	for _, cached := range []bool{false, true} {
		for _, batched := range []bool{false, true} {
			b.Run(fmt.Sprintf("cached=%t/batched=%t", cached, batched), func(b *testing.B) {
				s := qi21AttentionTestState(32, 16, 16, 32, 128)
				n, offset := 288, 0
				if cached {
					n, offset = 256, 32
					s.q, s.attn = s.q[32*s.model.dim:], s.attn[32*s.model.dim:]
				}
				ctx := context.Background()
				run := s.attentionOnline
				if batched {
					if !s.attentionBLAS(ctx, n, offset) {
						b.Skip("Accelerate unavailable")
					}
					run = s.attention
				}
				run(ctx, n, offset)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run(ctx, n, offset)
				}
			})
		}
	}
}
