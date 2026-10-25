package gopherllm

import (
	"context"
	"fmt"
	"os"
	"testing"
)

// Use the real packed dimensions without requiring a model download for
// projection benchmarks. Keep setup and quantized-weight allocation untimed.
func BenchmarkQwenImage21Projection(b *testing.B) {
	for _, shape := range [][2]int{{4096, 4096}, {12288, 4096}, {4096, 12288}} {
		for _, tokens := range []int{256, 288} {
			b.Run(fmt.Sprintf("%dx%d/tokens=%d", shape[0], shape[1], tokens), func(b *testing.B) {
				rows, cols := shape[0], shape[1]
				raw := make([]byte, rows*cols/32*34)
				for block := 0; block < len(raw); block += 34 {
					raw[block], raw[block+1] = 0, 0x20 // F16 scale 1/128.
					for j := 2; j < 34; j++ {
						raw[block+j] = byte((block/34+j)%255 - 127)
					}
				}
				weight := Weight{Raw: raw, Type: GGMLTypeQ8_0, Rows: rows, Cols: cols}
				input, output := make([]float32, tokens*cols), make([]float32, tokens*rows)
				for i := range input {
					input[i] = float32(i%31-15) / 16
				}
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := qi21Project(ctx, weight, input, output); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// Model loading is outside timing. This opt-in benchmark also permits a CPU
// profile of the complete graph; ordinary tests do not download weights.
func BenchmarkQwenImage21Generate(b *testing.B) {
	path := os.Getenv("GOPHERLLM_QWEN_IMAGE_MODEL")
	if path == "" {
		b.Skip("set GOPHERLLM_QWEN_IMAGE_MODEL to the local Turbo directory")
	}
	model, err := OpenQwenImage21(context.Background(), path)
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	opts := QwenImage21Options{Prompt: "A red ceramic teapot on a pale blue table, studio photograph.", Width: 256, Height: 256, Steps: 8, Seed: 42}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Generate(context.Background(), opts); err != nil {
			b.Fatal(err)
		}
	}
}
