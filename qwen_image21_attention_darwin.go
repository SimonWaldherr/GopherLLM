//go:build darwin && cgo

package gopherllm

import (
	"context"
	"math"

	"github.com/SimonWaldherr/GopherLLM/internal/voxtralblas"
)

func (s *qi21TransformerState) attentionBLAS(ctx context.Context, n, offset int) bool {
	return imageAttentionBLAS(ctx, s.q, s.k, s.v, s.attn, n, s.prefix+s.height*s.width, s.model.cfg.Heads, s.model.cfg.HeadDim, offset, s.prefix, &s.scores)
}

func imageAttentionBLAS(ctx context.Context, q, k, v, out []float32, n, total, heads, dim, offset, prefix int, scores *[]float32) bool {
	return imageGroupedAttentionBLAS(ctx, q, k, v, out, n, total, heads, heads, dim, offset, prefix, scores)
}

func imageGroupedAttentionBLAS(ctx context.Context, q, k, v, out []float32, n, total, heads, kvHeads, dim, offset, prefix int, scores *[]float32) bool {
	if n < 16 {
		return false
	}
	if ctx.Err() != nil {
		return true
	}
	width := heads * dim
	// Bound the score tile at four MiB even for large latent grids. The
	// state's scratch allowance already includes this temporary workspace.
	workers := voxtralblas.AttentionWorkers(min(n, 256), total, heads, numThreads())
	rows := max(1, min(n, 256, (1<<20)/total/workers))
	ensureLenNoClear(scores, workers*rows*total)
	scale := float32(1 / math.Sqrt(float64(dim)))
	for first := 0; first < n; first += rows {
		if ctx.Err() != nil {
			return true
		}
		count := min(rows, n-first)
		start := first * width
		voxtralblas.GroupedPrefixAttention(count, total, dim, heads, kvHeads, offset+first, prefix, scale,
			q[start:], k, v, (*scores)[:workers*count*total], out[start:], workers)
	}
	return true
}
