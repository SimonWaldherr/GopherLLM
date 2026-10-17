//go:build arm64

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// Match the existing SIMD accumulation order, including the scalar pairwise
// reduction and separately rounded scalar tail. The finite random inputs below
// are dyadic fractions with small numerators: their products and the addition
// of a float32 accumulator are exact in float64 before rounding to float32.
func arm64DotTreeReference(a, b []float32, width int) float32 {
	n := min(len(a), len(b))
	var acc [4][4]float32
	i := 0
	for ; i+width <= n; i += width {
		for j := 0; j < width; j++ {
			p := &acc[j/4][j%4]
			*p = float32(math.FMA(float64(a[i+j]), float64(b[i+j]), float64(*p)))
		}
	}
	for group := 1; group < width/4; group++ {
		for lane := range acc[0] {
			acc[0][lane] = float32(float64(acc[0][lane]) + float64(acc[group][lane]))
		}
	}
	p0 := float32(float64(acc[0][0]) + float64(acc[0][1]))
	p1 := float32(float64(acc[0][2]) + float64(acc[0][3]))
	sum := float32(float64(p0) + float64(p1))
	for ; i < n; i++ {
		product := float32(float64(a[i]) * float64(b[i]))
		sum = float32(float64(sum) + float64(product))
	}
	return sum
}

func arm64RequireDotBits(t *testing.T, label string, got, want float32) {
	t.Helper()
	if math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("%s = %g (%08x), want %g (%08x)", label, got, math.Float32bits(got), want, math.Float32bits(want))
	}
}

func TestARM64FloatDotReductionBits(t *testing.T) {
	z0, z1, z2, z3 := dotF32x4(nil, nil, nil, nil, nil, 0)
	for _, got := range [...]float32{z0, z1, z2, z3} {
		arm64RequireDotBits(t, "empty grouped", got, 0)
	}
	z0, z1, z2, z3 = dotF32F16x4NEON(nil, nil, nil, nil, nil, 0)
	for _, got := range [...]float32{z0, z1, z2, z3} {
		arm64RequireDotBits(t, "empty f16 grouped", got, 0)
	}
	rng := rand.New(rand.NewSource(139))
	lengths := make([]int, 66)
	for i := range lengths {
		lengths[i] = i
	}
	lengths = append(lengths, 95, 96, 97, 127, 128, 129, 4096)
	for _, n := range lengths {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			// Misaligned views with nonzero neighbors also exercise head and tail
			// boundaries, including empty views.
			xBacking := make([]float32, n+3)
			x := xBacking[1 : n+1]
			xBacking[0], xBacking[n+1] = 1e30, -1e30
			for i := range x {
				x[i] = float32(rng.Intn(1<<16)-(1<<15)) / 4096
			}
			var q [4][]float32
			for h := range q {
				backing := make([]float32, n+4)
				q[h] = backing[2 : n+2]
				for i := range q[h] {
					q[h][i] = float32(rng.Intn(1<<16)-(1<<15)) / 4096
				}
				want := arm64DotTreeReference(q[h], x, 16)
				arm64RequireDotBits(t, "single", DotF32(q[h], x), want)
				// Both slice-length minima must retain the same result.
				arm64RequireDotBits(t, "shorter q", DotF32(q[h], xBacking[1:]), want)
				arm64RequireDotBits(t, "shorter x", DotF32(backing[2:], x), want)
			}
			g0, g1, g2, g3 := dotF32x4(unsafe.SliceData(q[0]), unsafe.SliceData(q[1]), unsafe.SliceData(q[2]), unsafe.SliceData(q[3]), unsafe.SliceData(x), n)
			for h, got := range [...]float32{g0, g1, g2, g3} {
				arm64RequireDotBits(t, fmt.Sprintf("grouped head %d", h), got, arm64DotTreeReference(q[h], x, 16))
			}

			x16 := make([]uint16, n)
			xExpanded := make([]float32, n)
			for i := range x16 {
				x16[i] = F32ToF16(x[i])
				xExpanded[i] = F16ToF32(x16[i])
			}
			n8 := n &^ 7
			f0, f1, f2, f3 := dotF32F16x4(unsafe.SliceData(q[0]), unsafe.SliceData(q[1]), unsafe.SliceData(q[2]), unsafe.SliceData(q[3]), unsafe.SliceData(x16), n)
			for h, got := range [...]float32{f0, f1, f2, f3} {
				prefix := arm64DotTreeReference(q[h][:n8], xExpanded[:n8], 8)
				arm64RequireDotBits(t, "f16 single SIMD", dotF32F16NEON(q[h][:n8], x16[:n8]), prefix)
				// The existing Go tails use fused multiply-add on arm64. The
				// grouped wrapper adds to the SIMD prefix; the single-head
				// wrapper sums its scalar tail independently and then adds it.
				grouped := prefix
				var tail float32
				for i := n8; i < n; i++ {
					grouped = float32(math.FMA(float64(q[h][i]), float64(xExpanded[i]), float64(grouped)))
					tail = float32(math.FMA(float64(q[h][i]), float64(xExpanded[i]), float64(tail)))
				}
				arm64RequireDotBits(t, fmt.Sprintf("f16 grouped head %d", h), got, grouped)
				arm64RequireDotBits(t, "f16 single wrapper", dotF32F16(q[h], x16), float32(float64(prefix)+float64(tail)))
			}
		})
	}
}

func TestARM64FloatDotPairwiseCancellation(t *testing.T) {
	var q [4][17]float32
	var x [17]float32
	var x16 [17]uint16
	for i := range x {
		x[i], x16[i] = 1, F32ToF16(1)
	}
	for h := range q {
		q[h][0], q[h][1], q[h][2], q[h][3] = 1e20, 3, -1e20, 4
		q[h][16] = float32(h + 1)
		arm64RequireDotBits(t, "single pairwise cancellation", DotF32(q[h][:], x[:]), float32(h+1))
	}
	g0, g1, g2, g3 := dotF32x4(&q[0][0], &q[1][0], &q[2][0], &q[3][0], &x[0], 17)
	f0, f1, f2, f3 := dotF32F16x4(&q[0][0], &q[1][0], &q[2][0], &q[3][0], &x16[0], 17)
	for h, got := range [...]float32{g0, g1, g2, g3} {
		arm64RequireDotBits(t, "grouped pairwise cancellation", got, float32(h+1))
	}
	for h, got := range [...]float32{f0, f1, f2, f3} {
		arm64RequireDotBits(t, "f16 grouped pairwise cancellation", got, float32(h+1))
	}
}

var arm64FloatStackSink float32

// Each call owns fresh arrays. A missing noescape declaration makes these
// arrays escape even though the assembly retains none of their pointers.
//
//go:noinline
func arm64FloatStackInputs(operation int) float32 {
	var a, b, out [17]float32
	for i := range a {
		a[i], b[i] = float32(i+1)/8, float32(i%7-3)/4
	}
	switch operation {
	case 0:
		return DotF32(a[:], b[:])
	case 1:
		AxpyF32(a[:], .25, b[:])
	case 2:
		ScaleF32(a[:], .25)
	case 3:
		ScaleAddF32(a[:], .25, b[:])
	case 4:
		mulScaleF32(a[:], b[:], .25, out[:])
		return out[0] + out[16]
	}
	return a[0] + a[16]
}

func TestARM64FloatStackInputsDoNotEscape(t *testing.T) {
	for operation, name := range []string{"dot", "axpy", "scale", "scale_add", "mul_scale"} {
		t.Run(name, func(t *testing.T) {
			if allocs := testing.AllocsPerRun(20, func() { arm64FloatStackSink = arm64FloatStackInputs(operation) }); allocs != 0 {
				t.Fatalf("local float arrays allocated %g times; want 0", allocs)
			}
		})
	}
}

func BenchmarkARM64FloatStackInputs(b *testing.B) {
	for operation, name := range []string{"dot", "axpy", "scale", "scale_add", "mul_scale"} {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				arm64FloatStackSink = arm64FloatStackInputs(operation)
			}
		})
	}
}

func BenchmarkARM64FloatDotReduction(b *testing.B) {
	for _, n := range []int{128, 4096} {
		for _, operation := range []string{"f32", "f16", "f32x4", "f16x4"} {
			b.Run(fmt.Sprintf("%s/n=%d", operation, n), func(b *testing.B) {
				q := [4][]float32{}
				for h := range q {
					q[h] = benchFloatSlice(n)
				}
				x := benchFloatSlice(n)
				x16 := make([]uint16, n)
				for i := range x16 {
					x16[i] = F32ToF16(x[i])
				}
				b.ReportAllocs()
				for b.Loop() {
					switch operation {
					case "f32":
						arm64FloatStackSink = DotF32(q[0], x)
					case "f16":
						arm64FloatStackSink = dotF32F16NEON(q[0], x16)
					case "f32x4":
						s0, s1, s2, s3 := dotF32x4(&q[0][0], &q[1][0], &q[2][0], &q[3][0], &x[0], n)
						arm64FloatStackSink = s0 + s1 + s2 + s3
					case "f16x4":
						s0, s1, s2, s3 := dotF32F16x4NEON(&q[0][0], &q[1][0], &q[2][0], &q[3][0], &x16[0], n)
						arm64FloatStackSink = s0 + s1 + s2 + s3
					}
				}
			})
		}
	}
}
