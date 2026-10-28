//go:build amd64

package engine

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// Explicit product conversions require float32 rounding before addition,
// including GOAMD64=v3 builds whose scalar MUL+ADD may otherwise fuse to FMA.
func amd64DotFourLaneReference(a, b []float32) float32 {
	n := min(len(a), len(b))
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += float32(a[i] * b[i])
		s1 += float32(a[i+1] * b[i+1])
		s2 += float32(a[i+2] * b[i+2])
		s3 += float32(a[i+3] * b[i+3])
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += float32(a[i] * b[i])
	}
	return sum
}

func TestAMD64SSE2DotMatchesScalarBits(t *testing.T) {
	rng := rand.New(rand.NewSource(264))
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 4096} {
		for _, lengths := range [][2]int{{n, n}, {n, n + 3}, {n + 3, n}, {n, 0}, {0, n}} {
			for pattern := range 4 {
				a, b := make([]float32, lengths[0]+5), make([]float32, lengths[1]+5)
				for i := range a {
					a[i] = (rng.Float32() - .5) * 16
				}
				for i := range b {
					b[i] = (rng.Float32() - .5) * 16
				}
				if pattern > 0 {
					// Cancellation, signed zero and subnormal products exercise the
					// lane order rather than only well-conditioned positive sums.
					bits := [3][]uint32{
						{0x5f000000, 0x3f800000, 0xdf000000, 0x3f000000, 0xbf000000},
						{0, 0x80000000, 0x80000000, 0, 0x3f800000},
						{1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000},
					}[pattern-1]
					for i := range a {
						a[i] = math.Float32frombits(bits[i%len(bits)])
					}
					for i := range b {
						b[i] = float32(i%7)*.25 - .5
					}
				}
				aBefore, bBefore := append([]float32(nil), a...), append([]float32(nil), b...)
				// Nonzero offsets exercise unaligned loads and neighboring guards.
				x, y := a[1:1+lengths[0]], b[2:2+lengths[1]]
				want := amd64DotFourLaneReference(x, y)
				got := dotF32SSE2(x, y)
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("lengths=%v pattern=%d bits=%08x, want %08x", lengths, pattern, math.Float32bits(got), math.Float32bits(want))
				}
				if self, wantSelf := dotF32SSE2(x, x), amd64DotFourLaneReference(x, x); math.Float32bits(self) != math.Float32bits(wantSelf) {
					t.Fatalf("aliased lengths=%v pattern=%d bits=%08x, want %08x", lengths, pattern, math.Float32bits(self), math.Float32bits(wantSelf))
				}
				for i := range a {
					if math.Float32bits(a[i]) != math.Float32bits(aBefore[i]) {
						t.Fatal("dot changed its first input or guard")
					}
				}
				for i := range b {
					if math.Float32bits(b[i]) != math.Float32bits(bBefore[i]) {
						t.Fatal("dot changed its second input or guard")
					}
				}
			}
		}
	}
}

func TestAMD64SSE2DotDisableSetting(t *testing.T) {
	want := os.Getenv("GOPHERLLM_DISABLE_SIMD") == ""
	if useSSE2Dot != want {
		t.Fatalf("SSE2 enabled=%t, want %t", useSSE2Dot, want)
	}
	if !want && (hasAVX2 || hasF16C) {
		t.Fatal("SIMD disable setting left an amd64 feature enabled")
	}
}

var amd64DotStackSink float32

//go:noinline
func amd64DotStackInputs() float32 {
	var a, b [128]float32
	for i := range a {
		a[i] = float32(i%9)*.125 - .5
		b[i] = float32(i%7)*.25 - .75
	}
	return DotF32(a[:], b[:])
}

func TestAMD64DotStackInputsDoNotEscape(t *testing.T) {
	if allocs := testing.AllocsPerRun(20, func() { amd64DotStackSink = amd64DotStackInputs() }); allocs != 0 {
		t.Fatalf("stack dot inputs allocated %g times; want 0", allocs)
	}
}

func BenchmarkAMD64DotStackInputs(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		amd64DotStackSink = amd64DotStackInputs()
	}
}

// Native amd64 hosts may also use this benchmark. Timings collected on an ARM
// host via Rosetta are emulation measurements, not Intel/AMD speedup evidence.
func BenchmarkAMD64SSE2Dot(b *testing.B) {
	for _, n := range []int{16, 128, 129, 4096} {
		for _, scalar := range []bool{true, false} {
			name := "sse2"
			if scalar {
				name = "scalar"
			}
			b.Run(fmt.Sprintf("n=%d/%s", n, name), func(b *testing.B) {
				a, c := make([]float32, n), make([]float32, n)
				for i := range a {
					a[i], c[i] = float32(i%7)*.125, float32(i%11)*.25-.5
				}
				b.ReportAllocs()
				for b.Loop() {
					if scalar {
						amd64DotStackSink = dotF32Scalar(a, c)
					} else {
						amd64DotStackSink = dotF32SSE2(a, c)
					}
				}
			})
		}
	}
}
