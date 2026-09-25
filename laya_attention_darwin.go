//go:build darwin && cgo

package gopherllm

import (
	"context"
	"unsafe"

	"github.com/SimonWaldherr/GopherLLM/internal/layablas"
)

func layaAttentionAccelerated(ctx context.Context, n, heads, dim, window int, scale float32, work *layaWorkspace) (bool, error) {
	return true, layablas.Attention(ctx, n, heads, dim, window, scale, work.qkv.data, work.attention.data, &work.scores)
}

// Only the layer workspace path uses this helper. Verify the strided layout
// before exposing a full backing slice to Accelerate; other layouts fall back.
func layaProjectAccelerated(w Weight, x [][]float32, out *layaBuffer, stride int) bool {
	n := len(x)
	if n < 2 || w.Rows <= 0 || w.Cols <= 0 || len(w.F32) != w.Rows*w.Cols || stride < w.Cols {
		return false
	}
	span := (n-1)*stride + w.Cols
	if len(x[0]) != w.Cols || cap(x[0]) < span {
		return false
	}
	base := uintptr(unsafe.Pointer(&x[0][0]))
	for i, r := range x {
		if len(r) != w.Cols || uintptr(unsafe.Pointer(&r[0])) != base+uintptr(i*stride)*4 {
			return false
		}
	}
	layablas.Project(n, w.Rows, w.Cols, stride, w.F32, x[0][:span], out.data)
	return true
}
