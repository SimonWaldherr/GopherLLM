//go:build !darwin || !cgo

package gopherllm

import "context"

func layaAttentionAccelerated(ctx context.Context, n, heads, dim, window int, scale float32, work *layaWorkspace) (bool, error) {
	return false, nil
}

func layaProjectAccelerated(w Weight, x [][]float32, out *layaBuffer, stride int) bool { return false }
