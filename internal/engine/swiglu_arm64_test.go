//go:build arm64

package engine

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func requireSwiGLUBits(t *testing.T, gate, up, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.IsNaN(float64(want[i])) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("i=%d gate=%g up=%g: got %g, want NaN", i, gate[i], up[i], got[i])
			}
		} else if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("i=%d gate=%g up=%g: got %g (%08x), want %g (%08x)", i, gate[i], up[i], got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestARM64SwiGLUBits(t *testing.T) {
	rng := rand.New(rand.NewSource(17026))
	var gate []float32
	// Cover the ordinary activation range, every exponent-reduction boundary
	// and its neighboring floats, saturation, subnormals and non-finite inputs.
	for i := -90000; i <= 90000; i++ {
		gate = append(gate, float32(i)/1000)
	}
	for k := -25; k <= 127; k++ {
		x := -float32((float64(k) + .5) * math.Ln2)
		gate = append(gate, x, math.Nextafter32(x, float32(math.Inf(-1))), math.Nextafter32(x, float32(math.Inf(1))))
	}
	for _, x := range []float32{-87.33654, 17, 0, math.Float32frombits(0x80000000), math.Float32frombits(1), math.Float32frombits(0x80000001), math.MaxFloat32, -math.MaxFloat32} {
		gate = append(gate, x, math.Nextafter32(x, float32(math.Inf(-1))), math.Nextafter32(x, float32(math.Inf(1))))
	}
	gate = append(gate, float32(math.Inf(-1)), float32(math.Inf(1)), math.Float32frombits(0x7fc12345), math.Float32frombits(0xff812345))
	for range 20000 {
		gate = append(gate, math.Float32frombits(rng.Uint32()))
	}
	up := make([]float32, len(gate))
	specialUp := [...]uint32{0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345}
	for i := range up {
		up[i] = (rng.Float32() - .5) * 16
		if i%257 == 0 {
			up[i] = math.Float32frombits(specialUp[i%len(specialUp)])
		}
	}
	got, want := make([]float32, len(gate)), make([]float32, len(gate))
	siluMulF32Scalar(gate, up, want, 0, len(gate))
	siluMulF32(gate, up, got)
	requireSwiGLUBits(t, gate, up, got, want)
}

func TestARM64SwiGLULengthsAndAliases(t *testing.T) {
	lengths := make([]int, 66)
	for i := range lengths {
		lengths[i] = i
	}
	lengths = append(lengths, 127, 128, 129, 9216)
	for _, n := range lengths {
		for short := range 3 {
			t.Run(fmt.Sprintf("n=%d/short=%d", n, short), func(t *testing.T) {
				gate, up, out := make([]float32, n+7), make([]float32, n+7), make([]float32, n+7)
				for i := range gate {
					gate[i], up[i], out[i] = float32(i%53-26)*.375, float32(i%31-15)*.25, 1234
				}
				// Offset views ensure every vector load/store can be unaligned.
				g, u, o := gate[1:n+4], up[2:n+5], out[3:n+6]
				switch short {
				case 0:
					g = g[:n]
				case 1:
					u = u[:n]
				case 2:
					o = o[:n]
				}
				want := make([]float32, n)
				siluMulF32Scalar(g, u, want, 0, n)
				siluMulF32(g, u, o)
				requireSwiGLUBits(t, g, u, o[:n], want)
				for i, x := range out {
					if (i < 3 || i >= n+3) && x != 1234 {
						t.Fatalf("output guard %d overwritten: %g", i, x)
					}
				}
				for i := range gate {
					if gate[i] != float32(i%53-26)*.375 || up[i] != float32(i%31-15)*.25 {
						t.Fatalf("input %d overwritten", i)
					}
				}

				// FFN callers may reuse either complete input as the output.
				aliasGate, aliasUp := append([]float32(nil), g...), append([]float32(nil), u...)
				siluMulF32(aliasGate, u, aliasGate[:n])
				siluMulF32(g, aliasUp, aliasUp[:n])
				requireSwiGLUBits(t, g, u, aliasGate[:n], want)
				requireSwiGLUBits(t, g, u, aliasUp[:n], want)
			})
		}
	}
	siluMulF32(nil, nil, nil)
}

var swigluARM64Sink float32

//go:noinline
func swigluARM64StackInputs() float32 {
	var gate, up, out [33]float32
	for i := range gate {
		gate[i], up[i] = float32(i%19-9)*.25, float32(i%7-3)*.5
	}
	siluMulF32(gate[:], up[:], out[:])
	return out[0] + out[32]
}

func TestARM64SwiGLUStackInputsDoNotEscape(t *testing.T) {
	if n := testing.AllocsPerRun(20, func() { swigluARM64Sink = swigluARM64StackInputs() }); n != 0 {
		t.Fatalf("local SwiGLU arrays allocated %g times; want 0", n)
	}
}

func BenchmarkSwiGLUARM64(b *testing.B) {
	for _, n := range []int{4, 16, 128, 3072, 9216, 16384} {
		for _, scalar := range []bool{true, false} {
			mode := "neon"
			if scalar {
				mode = "scalar"
			}
			b.Run(fmt.Sprintf("n=%d/%s", n, mode), func(b *testing.B) {
				gate, up, out := make([]float32, n), make([]float32, n), make([]float32, n)
				for i := range gate {
					gate[i], up[i] = float32(i%211)/13-8, float32(i%97)/11-4
				}
				b.ReportAllocs()
				for b.Loop() {
					if scalar {
						siluMulF32Scalar(gate, up, out, 0, n)
					} else {
						siluMulF32(gate, up, out)
					}
				}
				swigluARM64Sink = out[n-1]
			})
		}
	}
}
