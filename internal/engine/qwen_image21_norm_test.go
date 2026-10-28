package engine

import (
	"fmt"
	"math"
	"slices"
	"testing"
	"time"
)

func qi21VAENormReference(x diffusionTensor, gamma []float32, activate bool) {
	n := x.h * x.w
	scale := float32(math.Sqrt(float64(x.c)))
	for p := range n {
		sum := float32(0)
		for ch := range x.c {
			value := x.data[ch*n+p]
			sum += value * value
		}
		denom := max(float32(math.Sqrt(float64(sum))), float32(1e-12))
		for ch := range x.c {
			value := x.data[ch*n+p] / denom * scale * gamma[ch]
			if activate {
				value = siluExact(value)
			}
			x.data[ch*n+p] = value
		}
	}
}

func TestQwenImage21VAENormMatchesReference(t *testing.T) {
	for _, shape := range [][3]int{{3, 2, 3}, {32, 1, 255}, {64, 1, 256}, {256, 17, 19}} {
		for _, activate := range []bool{false, true} {
			t.Run(fmt.Sprintf("%dx%dx%d/activate=%t", shape[0], shape[1], shape[2], activate), func(t *testing.T) {
				c, h, w := shape[0], shape[1], shape[2]
				x := diffusionTensor{benchFloatSlice(c * h * w), c, h, w}
				// Cover zero and subnormal norms as well as ordinary latent values.
				for ch := range c {
					x.data[ch*h*w] = 0
					x.data[ch*h*w+1] = math.SmallestNonzeroFloat32
				}
				gamma := benchFloatSlice(c)
				want := diffusionTensor{slices.Clone(x.data), c, h, w}
				qi21VAENormReference(want, gamma, activate)
				qi21VAENorm(x, gamma, activate)
				for i, value := range x.data {
					if math.Float32bits(value) != math.Float32bits(want.data[i]) {
						t.Fatalf("element %d: %.9g, want %.9g", i, value, want.data[i])
					}
				}
			})
		}
	}
}

func TestQwenImage21VAENormSpecialValues(t *testing.T) {
	for _, input := range [][2]float32{
		{0, 0}, {float32(math.Copysign(0, -1)), 0},
		{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32},
		{math.MaxFloat32, math.MaxFloat32},
		{float32(math.Inf(1)), 1}, {float32(math.Inf(-1)), 1},
		{float32(math.NaN()), 1},
	} {
		for _, activate := range []bool{false, true} {
			x := diffusionTensor{slices.Clone(input[:]), 2, 1, 1}
			want := diffusionTensor{slices.Clone(input[:]), 2, 1, 1}
			gamma := []float32{1, 1}
			qi21VAENormReference(want, gamma, activate)
			qi21VAENorm(x, gamma, activate)
			for i, value := range x.data {
				if math.IsNaN(float64(value)) && math.IsNaN(float64(want.data[i])) {
					continue
				}
				if math.Float32bits(value) != math.Float32bits(want.data[i]) {
					t.Fatalf("input %v, activate=%t: element %d is %g, want %g", input, activate, i, value, want.data[i])
				}
			}
		}
	}
}

// Alternate reference and tiled calls on identical copied inputs, reducing
// order bias on a busy host. The default ns/op covers the pair plus timing;
// custom metrics report the median of each variant, including its source copy.
func BenchmarkQwenImage21VAENorm(b *testing.B) {
	for _, shape := range [][3]int{{64, 32, 32}, {256, 128, 128}} {
		for _, activate := range []bool{false, true} {
			b.Run(fmt.Sprintf("channels=%d/grid=%dx%d/activate=%t", shape[0], shape[1], shape[2], activate), func(b *testing.B) {
				c, h, w := shape[0], shape[1], shape[2]
				source := benchFloatSlice(c * h * w)
				x := diffusionTensor{make([]float32, len(source)), c, h, w}
				gamma := benchFloatSlice(c)
				reference, tiled := make([]int64, b.N), make([]int64, b.N)
				measure := func(run func(diffusionTensor, []float32, bool)) int64 {
					start := time.Now()
					copy(x.data, source)
					run(x, gamma, activate)
					return time.Since(start).Nanoseconds()
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := range b.N {
					if i%2 == 0 {
						reference[i] = measure(qi21VAENormReference)
						tiled[i] = measure(qi21VAENorm)
					} else {
						tiled[i] = measure(qi21VAENorm)
						reference[i] = measure(qi21VAENormReference)
					}
				}
				b.StopTimer()
				slices.Sort(reference)
				slices.Sort(tiled)
				b.ReportMetric(float64(reference[b.N/2]), "reference-ns/op")
				b.ReportMetric(float64(tiled[b.N/2]), "tiled-ns/op")
				b.Logf("reference ns/op: %v; tiled ns/op: %v", reference, tiled)
			})
		}
	}
}
