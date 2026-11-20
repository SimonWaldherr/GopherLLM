//go:build darwin && cgo && metal

package engine

import (
	"os"
	"runtime"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func prepareMetalBF16Head(w Weight) *MetalWeight {
	if runtime.GOARCH != "arm64" || os.Getenv("GOPHERLLM_METAL_BF16_HEAD") == "0" {
		return nil
	}
	handle := metalbackend.PrepareBorrowedBF16Head(w.Raw, w.Rows, w.Cols)
	if handle == nil {
		return nil
	}
	return &MetalWeight{bf16: handle, typ: GGMLTypeBF16, rows: w.Rows, cols: w.Cols}
}

func matvecMetalBF16HeadInto(w *MetalWeight, x []float32, out *[]float32) bool {
	if w == nil || w.bf16 == nil || len(x) != w.cols {
		return false
	}
	ensureLenNoClear(out, w.rows)
	return metalbackend.MatvecBF16Head(w.bf16, x, *out)
}

func argmaxMetalBF16Head(w *MetalWeight, x []float32, recent []uint32, penalty float32) (uint32, bool) {
	if w == nil || w.bf16 == nil {
		return 0, false
	}
	return metalbackend.ArgmaxBF16Head(w.bf16, x, recent, penalty)
}
