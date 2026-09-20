//go:build !darwin || !cgo

package gopherllm

import "context"

func prepareVoxtralStreamWeights(ctx context.Context, w *VoxtralRealtimeWeights) error {
	return ctx.Err()
}
func voxtralMatvecBatch(w Weight, xs, outs [][]float32) { matvecBatch(w, xs, outs) }

func voxtralAttentionBatch(q [][]float32, k, v []float32, out [][]float32, heads, dim, past, window, stride int, scale float32) bool {
	return false
}
