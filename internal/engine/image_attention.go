package engine

import (
	"context"
	"math"
)

// Token-major attention shared by the image transformers and VAE. Prefix
// queries are causal; with prefix=0 all queries attend to all supplied keys.
// The caller validates dimensions. Scores use a bounded tile, never N squared.
func imageAttention(ctx context.Context, q, k, v, out []float32, n, keys, heads, dim, offset, prefix int, scores *[]float32) {
	imageGroupedAttention(ctx, q, k, v, out, n, keys, heads, heads, dim, offset, prefix, scores)
}

// Grouped attention reads each K/V head directly. Repeating shared heads into
// a query-width buffer would multiply the encoder's memory use by heads/kvHeads.
func imageGroupedAttention(ctx context.Context, q, k, v, out []float32, n, keys, heads, kvHeads, dim, offset, prefix int, scores *[]float32) {
	if imageGroupedAttentionBLAS(ctx, q, k, v, out, n, keys, heads, kvHeads, dim, offset, prefix, scores) {
		return
	}
	imageGroupedAttentionOnline(ctx, q, k, v, out, n, keys, heads, kvHeads, dim, offset, prefix)
}

func imageAttentionOnline(ctx context.Context, q, k, v, out []float32, n, keys, heads, dim, offset, prefix int) {
	imageGroupedAttentionOnline(ctx, q, k, v, out, n, keys, heads, heads, dim, offset, prefix)
}

func imageGroupedAttentionOnline(ctx context.Context, q, k, v, out []float32, n, keys, heads, kvHeads, dim, offset, prefix int) {
	width, kvWidth, group := heads*dim, kvHeads*dim, heads/kvHeads
	scale := float32(1 / math.Sqrt(float64(dim)))
	parallelRows(n*heads, func(begin, end int) {
		for row := begin; row < end; row++ {
			if ctx.Err() != nil {
				return
			}
			token, head := row/heads, row%heads
			endKey := keys
			if token+offset < prefix {
				endKey = token + offset + 1
			}
			start := token*width + head*dim
			result := out[start : start+dim]
			clear(result)
			kv := (head / group) * dim
			onlineAttention(q[start:start+dim], k[kv:], v[kv:], kvWidth, kvWidth, dim, dim, 0, endKey-1, scale, 0, result)
		}
	})
}
