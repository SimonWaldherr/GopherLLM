package gopherllm

import (
	"math"
	"math/rand"
	"runtime"
	"strconv"
	"testing"
)

// Keep the original variable-width loop independent of the specialized kernel.
//
//go:noinline
func qwen35ConvReference(kernel, history, values []float32, width int) {
	for ch := range values {
		k := kernel[ch*width : (ch+1)*width]
		h := history[ch*(width-1) : (ch+1)*(width-1)]
		input := values[ch]
		v := float32(0)
		for i := range h {
			v += k[i] * h[i]
		}
		v += k[len(h)] * input
		copy(h, h[1:])
		h[len(h)-1] = input
		values[ch] = v
	}
}

func TestQwen35Conv4Parity(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("ARM64 convolution kernel")
	}
	for _, channels := range []int{4, 5, 7, 8, 9, 15, 16, 31, 10240} {
		t.Run(strconv.Itoa(channels), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(channels)))
			kernel := make([]float32, channels*4+2)
			history := make([]float32, channels*3+2)
			values := make([]float32, channels+2)
			for _, dst := range [][]float32{kernel, history} {
				for i := range dst {
					dst[i] = float32(rng.NormFloat64())
				}
			}
			kernel[0], kernel[len(kernel)-1] = 987, -654
			history[0], history[len(history)-1] = 321, -432
			values[0], values[len(values)-1] = 456, -543
			wantHistory := append([]float32(nil), history...)
			for step := range 32 {
				for i := range channels {
					values[i+1] = float32(rng.NormFloat64())
				}
				want := append([]float32(nil), values...)
				qwen35ConvReference(kernel[1:len(kernel)-1], wantHistory[1:len(history)-1], want[1:len(want)-1], 4)
				if !qwen35Conv4(kernel[1:len(kernel)-1], history[1:len(history)-1], values[1:len(values)-1]) {
					t.Fatal("valid convolution declined")
				}
				for i := range values {
					if math.Float32bits(values[i]) != math.Float32bits(want[i]) {
						t.Fatalf("step %d output %d: got %08x, want %08x", step, i, math.Float32bits(values[i]), math.Float32bits(want[i]))
					}
				}
				for i := range history {
					if math.Float32bits(history[i]) != math.Float32bits(wantHistory[i]) {
						t.Fatalf("step %d history %d changed incorrectly", step, i)
					}
				}
			}
		})
	}
}

func TestQwen35Conv4SpecialValues(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("ARM64 convolution kernel")
	}
	special := []float32{0, math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		math.MaxFloat32, -math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc12345)}
	for offset := range special {
		kernel, history, values := make([]float32, 20), make([]float32, 15), make([]float32, 5)
		for _, dst := range [][]float32{kernel, history, values} {
			for i := range dst {
				dst[i] = special[(i+offset)%len(special)]
			}
		}
		wantHistory, want := append([]float32(nil), history...), append([]float32(nil), values...)
		qwen35ConvReference(kernel, wantHistory, want, 4)
		if !qwen35Conv4(kernel, history, values) {
			t.Fatal("valid convolution declined")
		}
		for i := range values {
			if math.Float32bits(values[i]) != math.Float32bits(want[i]) {
				t.Fatalf("offset %d output %d: got %08x, want %08x", offset, i, math.Float32bits(values[i]), math.Float32bits(want[i]))
			}
		}
		for i := range history {
			if math.Float32bits(history[i]) != math.Float32bits(wantHistory[i]) {
				t.Fatalf("offset %d history %d changed incorrectly", offset, i)
			}
		}
	}
}

func TestQwen35Conv4DeclinesWithoutMutation(t *testing.T) {
	for _, sizes := range [][3]int{{0, 0, 0}, {12, 9, 3}, {15, 12, 4}, {16, 11, 4}} {
		kernel, history, values := make([]float32, sizes[0]), make([]float32, sizes[1]), make([]float32, sizes[2])
		for _, dst := range [][]float32{kernel, history, values} {
			for i := range dst {
				dst[i] = float32(i + 1)
			}
		}
		if qwen35Conv4(kernel, history, values) {
			t.Fatalf("invalid dimensions accepted: %v", sizes)
		}
		for _, dst := range [][]float32{kernel, history, values} {
			for i, value := range dst {
				if value != float32(i+1) {
					t.Fatalf("declined convolution mutated storage: %v", sizes)
				}
			}
		}
	}
	if runtime.GOARCH != "arm64" && qwen35Conv4(make([]float32, 16), make([]float32, 12), make([]float32, 4)) {
		t.Fatal("portable path must decline")
	}
}

func BenchmarkQwen35Conv4(b *testing.B) {
	const channels = 10240
	for _, specialized := range []bool{false, true} {
		name := "scalar"
		if specialized {
			name = "neon"
		}
		b.Run(name, func(b *testing.B) {
			kernel, history, values := make([]float32, channels*4), make([]float32, channels*3), make([]float32, channels)
			input := make([]float32, channels)
			for i := range kernel {
				kernel[i] = float32(i%13-6) / 64
			}
			for i := range input {
				input[i] = float32(i%11-5) / 64
			}
			if specialized && !qwen35Conv4(kernel, history, values) {
				b.Skip("ARM64 convolution kernel")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				copy(values, input)
				if specialized {
					qwen35Conv4(kernel, history, values)
				} else {
					qwen35ConvReference(kernel, history, values, 4)
				}
			}
		})
	}
}
