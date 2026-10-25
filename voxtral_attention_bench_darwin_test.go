//go:build darwin && cgo

package gopherllm

import (
	"fmt"
	"testing"
)

func BenchmarkVoxtralAttention(b *testing.B) {
	setPerformanceThreads(b, 8)
	const heads, dim, window = 32, 64, 750
	for _, n := range []int{1, 4, 16, 372} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			past := window
			if n == 372 {
				past = 0
			}
			stride := past + n
			q, out := audioMatrixRows(n, heads*dim), audioMatrixRows(n, heads*dim)
			k, v := make([]float32, heads*stride*dim), make([]float32, heads*stride*dim)
			for i := range k {
				k[i], v[i] = float32(i%47-23)*.01, float32(i%53-26)*.02
			}
			for i, row := range q {
				for j := range row {
					row[j] = float32((i+j)%41-20) * .03
				}
			}
			blasAttentionBatch(q, k, v, out, heads, dim, past, window, stride, .125)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				blasAttentionBatch(q, k, v, out, heads, dim, past, window, stride, .125)
			}
		})
	}
}
