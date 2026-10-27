package gopherllm

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func TestDenseGemm(t *testing.T) {
	setPerformanceThreads(t, 3)
	var scratch denseGemmScratch
	for _, shape := range [][3]int{{1, 1, 1}, {3, 7, 5}, {64, 128, 19}, {32, 513, 17}} {
		m, n, k := shape[0], shape[1], shape[2]
		a, b := ttsConvTestData(m*k), ttsConvTestData(k*n)
		out := make([]float32, m*n)
		for repeat := 0; repeat < 2; repeat++ {
			for i := range out {
				out[i] = float32(math.NaN()) // Every result must be overwritten.
			}
			if err := scratch.multiply(context.Background(), m, n, k, a, b, out); err != nil {
				t.Fatal(err)
			}
			for i, got := range out {
				var want float64
				for p := range k {
					want += float64(a[i/n*k+p]) * float64(b[p*n+i%n])
				}
				if !finite32(got) || math.Abs(float64(got)-want) > 1e-6+1e-5*math.Abs(want) {
					t.Fatalf("shape %v, output[%d]=%g, want %g", shape, i, got, want)
				}
			}
			if task := scratch.task; task != nil && (task.ctx != nil || task.a != nil || task.b != nil || task.out != nil) {
				t.Fatal("scratch retains matrix or context after the worker barrier")
			}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out := []float32{19}
	if err := scratch.multiply(ctx, 1, 1, 1, []float32{2}, []float32{3}, out); !errors.Is(err, context.Canceled) || out[0] != 19 {
		t.Fatalf("canceled multiplication: error=%v, output=%v", err, out)
	}
}

func TestCopyPaddedWindow(t *testing.T) {
	for repeat, copyWindow := range []func([]float32, []float32, int){copyPaddedWindow, copyPaddedRepeat2} {
		repeat++
		for _, srcLen := range []int{0, 1, 7, 128} {
			src := ttsConvTestData(srcLen)
			original := slices.Clone(src)
			for _, width := range []int{0, 1, 3, 127, 129} {
				storage := make([]float32, width+2)
				for start := -width - 2; start <= repeat*srcLen+2; start++ {
					for i := range storage {
						storage[i] = -123
					}
					dst := storage[1 : width+1]
					copyWindow(dst, src, start)
					for i, got := range dst {
						var want float32
						if at := start + i; at >= 0 && at < repeat*len(src) {
							want = src[at/repeat]
						}
						if got != want {
							t.Fatalf("repeat=%d, source=%d, width=%d, start=%d: output[%d]=%g, want %g", repeat, srcLen, width, start, i, got, want)
						}
					}
					if storage[0] != -123 || storage[width+1] != -123 || !slices.Equal(src, original) {
						t.Fatal("modified input or output boundary")
					}
				}
			}
		}
	}
}

// Compare patch gathering with the previous elementwise bounds checks. The
// source and destination layouts match the 128-frame TTS convolution tiles.
func BenchmarkConvolutionPatchCopy(b *testing.B) {
	for _, implementation := range []struct {
		name string
		copy func([]float32, []float32, int)
	}{
		{"elementwise", func(dst, src []float32, start int) {
			for i := range dst {
				at := start + i
				if at >= 0 && at < len(src) {
					dst[i] = src[at]
				} else {
					dst[i] = 0
				}
			}
		}},
		{"window", copyPaddedWindow},
	} {
		b.Run(implementation.name, func(b *testing.B) {
			const channels, kernel, frames = 192, 7, 128
			src := ttsConvTestData(channels * frames)
			dst := make([]float32, channels*kernel*frames)
			b.SetBytes(int64(len(dst) * 4))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				for ch := range channels {
					for k := range kernel {
						row := ch*kernel + k
						implementation.copy(dst[row*frames:(row+1)*frames], src[ch*frames:(ch+1)*frames], k-kernel/2)
					}
				}
			}
		})
	}
}
