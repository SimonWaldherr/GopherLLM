package gopherllm

import (
	"context"
	"fmt"
	"testing"
)

// Representative ModernBERT geometry; synthetic weights isolate scheduling
// from checkpoint loading. This is one encoder layer, not a classification.
func BenchmarkLayaLayer(b *testing.B) {
	setPerformanceThreads(b, 8)
	const dim, hidden = 768, 1152
	linear := func(rows, cols, seed int) layaLinear {
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32((i*17+seed)%127-63) * .0001
		}
		return layaLinear{weight: Weight{F32: w, Rows: rows, Cols: cols}}
	}
	norm := layaNorm{weight: make([]float32, dim), epsilon: 1e-5}
	for i := range norm.weight {
		norm.weight[i] = 1
	}
	layer := layaLayer{norm1: norm, norm2: norm, qkv: linear(3*dim, dim, 1),
		output: linear(dim, dim, 2), up: linear(2*hidden, dim, 3), down: linear(dim, hidden, 4),
		theta: 10000, window: 64, heads: 12, gated: true}
	for _, n := range []int{128, 512} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			var work layaWorkspace
			x := work.input.resize(n, dim, false)
			for i := range work.input.data {
				work.input.data[i] = float32(i%43-21) * .02
			}
			if err := layer.forward(context.Background(), x, &work); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err := layer.forward(context.Background(), x, &work); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
