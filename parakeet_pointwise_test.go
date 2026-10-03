package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// Original per-pixel channel reduction, before the contiguous GEMM path.
func referenceParakeetPointwise(x []float32, inChannels, h, w int, conv ParakeetSubsamplingConv) []float32 {
	plane := h * w
	out := make([]float32, conv.OutChannels*plane)
	for oc := range conv.OutChannels {
		for i := range plane {
			var sum float32
			for ic := range inChannels {
				sum += x[ic*plane+i] * conv.Weight[oc*inChannels+ic]
			}
			out[oc*plane+i] = sum + float32At(conv.Bias, oc)
		}
	}
	return out
}

func TestParakeetPointwiseMatchesPixelReference(t *testing.T) {
	rng := rand.New(rand.NewSource(5201))
	for _, shape := range [][4]int{{3, 5, 7, 9}, {257, 33, 3, 17}, {1, 1, 1, 1}, {17, 19, 13, 3}} {
		in, out, h, w := shape[0], shape[1], shape[2], shape[3]
		for _, bias := range []bool{false, true} {
			t.Run(fmt.Sprintf("%v/bias=%v", shape, bias), func(t *testing.T) {
				x := randomVec(rng, in*h*w)
				before := append([]float32(nil), x...)
				conv := ParakeetSubsamplingConv{OutChannels: out, InPerGroup: in, KH: 1, KW: 1, StrideH: 1, StrideW: 1, Weight: randomVec(rng, out*in)}
				if bias {
					conv.Bias = randomVec(rng, out)
				}
				want := referenceParakeetPointwise(x, in, h, w, conv)
				got, oh, ow := parakeetConv2D(x, in, h, w, conv)
				if oh != h || ow != w || len(got) != len(want) {
					t.Fatal("pointwise output shape changed")
				}
				for i, v := range want {
					// SGEMM may accumulate with a different float32 reduction;
					// channel order remains exact in the portable path.
					if d := math.Abs(float64(got[i] - v)); d > 2e-5*(1+math.Abs(float64(v))) {
						t.Fatalf("[%d]: got %g, want %g", i, got[i], v)
					}
				}
				if !reflect.DeepEqual(before, x) {
					t.Fatal("pointwise convolution modified its input")
				}
			})
		}
	}
}

func BenchmarkParakeetPointwise(b *testing.B) {
	const channels = 256
	rng := rand.New(rand.NewSource(5201))
	conv := ParakeetSubsamplingConv{OutChannels: channels, InPerGroup: channels, KH: 1, KW: 1, StrideH: 1, StrideW: 1, Weight: randomVec(rng, channels*channels), Bias: randomVec(rng, channels)}
	for _, shape := range [][2]int{{88, 32}, {44, 16}} {
		h, w := shape[0], shape[1]
		x := randomVec(rng, channels*h*w)
		b.Run(fmt.Sprintf("%dx%d", h, w), func(b *testing.B) {
			for _, mode := range []string{"pixelReference", "contiguous"} {
				b.Run(mode, func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if mode == "pixelReference" {
							_ = referenceParakeetPointwise(x, channels, h, w, conv)
						} else {
							_, _, _ = parakeetConv2D(x, channels, h, w, conv)
						}
					}
				})
			}
		})
	}
}
