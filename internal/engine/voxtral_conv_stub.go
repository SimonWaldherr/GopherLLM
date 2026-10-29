//go:build !darwin || !cgo

package engine

import "context"

// Portable builds retain the SIMD dot path and its bounded worker patches.
func voxtralCausalConvBatch(context.Context, []float32, int, VoxtralRealtimeEncoderConv, int, int, []float32) (bool, error) {
	return false, nil
}
