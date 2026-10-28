//go:build !darwin || !cgo

package engine

import "context"

func (s *qi21TransformerState) attentionBLAS(context.Context, int, int) bool { return false }

func imageAttentionBLAS(context.Context, []float32, []float32, []float32, []float32, int, int, int, int, int, int, *[]float32) bool {
	return false
}

func imageGroupedAttentionBLAS(context.Context, []float32, []float32, []float32, []float32, int, int, int, int, int, int, int, *[]float32) bool {
	return false
}
