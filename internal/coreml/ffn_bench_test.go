package coreml

import (
	"fmt"
	"testing"
)

func BenchmarkFFNValidation(b *testing.B) {
	// A copying backend isolates the Go validation/locking cost from Core ML.
	for _, dimension := range []int{3072, 5120} {
		b.Run(fmt.Sprint(dimension), func(b *testing.B) {
			input, output := make([]float32, 32*dimension), make([]float32, 32*dimension)
			for i := range input {
				input[i] = float32(i%31-15) * .125
			}
			m := &FFN{info: FFNInfo{Tokens: 32, Dimension: dimension}, backend: &testFFNBackend{}}
			b.ReportAllocs()
			b.SetBytes(int64(len(input) * 4))
			b.ResetTimer()
			for range b.N {
				if err := m.Predict(input, output); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
