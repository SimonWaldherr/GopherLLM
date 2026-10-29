package engine

import (
	"context"
	"fmt"
	"math"
	"os"
	"sync/atomic"
	"testing"
	"time"
)

// referenceVoxtralCausalConv preserves the frame-at-a-time implementation,
// including its SIMD dot reduction, padding and tanh GELU.
func referenceVoxtralCausalConv(ctx context.Context, x []float32, in, length int, conv VoxtralRealtimeEncoderConv) ([]float32, int, error) {
	padLeft, padRight, outLen := causalConv1dOutputLen(length, conv.Kernel, conv.Stride)
	paddedLen := length + padLeft + padRight
	padded := make([]float32, in*paddedLen)
	for c := range in {
		copy(padded[c*paddedLen+padLeft:c*paddedLen+padLeft+length], x[c*length:(c+1)*length])
	}
	out := make([]float32, conv.Out*outLen)
	workers := 1
	if outLen*conv.Out*in*conv.Kernel >= 1<<18 {
		workers = max(1, min(outLen, numThreads()))
	}
	width := in * conv.Kernel
	patches := make([]float32, workers*width)
	parallelChunks(workers, func(start, end int) {
		for worker := start; worker < end; worker++ {
			patch := patches[worker*width : (worker+1)*width]
			for t := worker; t < outLen; t += workers {
				if ctx.Err() != nil {
					return
				}
				base := t * conv.Stride
				for c := range in {
					copy(patch[c*conv.Kernel:(c+1)*conv.Kernel], padded[c*paddedLen+base:c*paddedLen+base+conv.Kernel])
				}
				for o := range conv.Out {
					v := DotF32(conv.Weight[o*width:(o+1)*width], patch)
					if len(conv.Bias) > o {
						v += conv.Bias[o]
					}
					out[o*outLen+t] = geluTanh(v)
				}
			}
		}
	})
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return out, outLen, nil
}

func voxtralConvFixture(in, out, length, stride int) ([]float32, VoxtralRealtimeEncoderConv) {
	fill := func(n, seed int) []float32 {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(math.Sin(float64(i*13+seed)*.071)) * .04
		}
		return data
	}
	return fill(in*length, 1), VoxtralRealtimeEncoderConv{In: in, Out: out, Kernel: 3, Stride: stride, Weight: fill(out*in*3, 2), Bias: fill(out, 3)}
}

func TestVoxtralCausalConvBatchMatchesReference(t *testing.T) {
	for _, stride := range []int{1, 2} {
		for _, length := range []int{1, 7, 31, 32, 33, 64, 65, 131} {
			t.Run(fmt.Sprintf("stride=%d/frames=%d", stride, length), func(t *testing.T) {
				x, w := voxtralConvFixture(128, 128, length, stride)
				original := append([]float32(nil), x...)
				want, wn, err := referenceVoxtralCausalConv(context.Background(), x, w.In, length, w)
				if err != nil {
					t.Fatal(err)
				}
				got, gn, err := applyCausalConv1dContext(context.Background(), x, w.In, length, w)
				if err != nil {
					t.Fatal(err)
				}
				if gn != wn || len(got) != len(want) {
					t.Fatalf("shape %d/%d want %d/%d", gn, len(got), wn, len(want))
				}
				for i, v := range want {
					if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-v)) > 2e-6*(1+math.Abs(float64(v))) {
						t.Fatalf("[%d]: %g want %g", i, got[i], v)
					}
				}
				for i := range x {
					if math.Float32bits(x[i]) != math.Float32bits(original[i]) {
						t.Fatal("input changed")
					}
				}
			})
		}
	}
}

func TestVoxtralCausalConvCancelled(t *testing.T) {
	x, w := voxtralConvFixture(128, 128, 131, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out, n, err := applyCausalConv1dContext(ctx, x, w.In, 131, w)
	if err != context.Canceled || out != nil || n != 0 {
		t.Fatalf("got len=%d n=%d err=%v", len(out), n, err)
	}
}

func BenchmarkVoxtralCausalConv(b *testing.B) {
	for _, shape := range []struct {
		name                    string
		in, out, length, stride int
	}{
		{"stem0_100ms", 128, 1280, 10, 1}, {"stem1_100ms", 1280, 1280, 10, 2},
		{"stem0_1s", 128, 1280, 100, 1}, {"stem1_1s", 1280, 1280, 100, 2},
		{"stem0_7s", 128, 1280, 700, 1}, {"stem1_7s", 1280, 1280, 700, 2},
	} {
		x, w := voxtralConvFixture(shape.in, shape.out, shape.length, shape.stride)
		for _, ref := range []bool{true, false} {
			mode := "batched"
			if ref {
				mode = "reference"
			}
			b.Run(shape.name+"/"+mode, func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					var err error
					if ref {
						_, _, err = referenceVoxtralCausalConv(context.Background(), x, w.In, shape.length, w)
					} else {
						_, _, err = applyCausalConv1dContext(context.Background(), x, w.In, shape.length, w)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// Alternating same-process pairs reduce drift from unrelated host workloads.
// The full function (allocation, packing, projection and GELU) is timed.
func BenchmarkVoxtralCausalConvPaired(b *testing.B) {
	for _, shape := range []struct {
		name                    string
		in, out, length, stride int
	}{
		{"stem0_1s", 128, 1280, 100, 1}, {"stem1_1s", 1280, 1280, 100, 2},
		{"stem0_7s", 128, 1280, 700, 1}, {"stem1_7s", 1280, 1280, 700, 2},
	} {
		x, w := voxtralConvFixture(shape.in, shape.out, shape.length, shape.stride)
		b.Run(shape.name, func(b *testing.B) {
			ctx := context.Background()
			run := func(ref bool) time.Duration {
				start := time.Now()
				var err error
				if ref {
					_, _, err = referenceVoxtralCausalConv(ctx, x, w.In, shape.length, w)
				} else {
					_, _, err = applyCausalConv1dContext(ctx, x, w.In, shape.length, w)
				}
				if err != nil {
					b.Fatal(err)
				}
				return time.Since(start)
			}
			run(true)
			run(false)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var before, after time.Duration
				if i%2 == 0 {
					before = run(true)
					after = run(false)
				} else {
					after = run(false)
					before = run(true)
				}
				b.Logf("pair=%d reference_ns=%d batched_ns=%d", i, before, after)
			}
		})
	}
}

func TestVoxtralCausalConvBias(t *testing.T) {
	x, w := voxtralConvFixture(128, 128, 67, 1)
	for _, n := range []int{0, 17, 128} {
		t.Run(fmt.Sprintf("bias=%d", n), func(t *testing.T) {
			local := w
			local.Bias = w.Bias[:n]
			want, _, err := referenceVoxtralCausalConv(context.Background(), x, local.In, 67, local)
			if err != nil {
				t.Fatal(err)
			}
			got, _, err := applyCausalConv1dContext(context.Background(), x, local.In, 67, local)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range want {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-v)) > 2e-6*(1+math.Abs(float64(v))) {
					t.Fatalf("[%d]: %g want %g", i, got[i], v)
				}
			}
		})
	}
}

type voxtralConvCancelContext struct {
	context.Context
	checks atomic.Int32
}

func (c *voxtralConvCancelContext) Err() error {
	if c.checks.Add(1) >= 4 {
		return context.Canceled
	}
	return nil
}
func TestVoxtralCausalConvCancellationBetweenTiles(t *testing.T) {
	x, w := voxtralConvFixture(128, 128, 131, 1)
	ctx := &voxtralConvCancelContext{Context: context.Background()}
	out, n, err := applyCausalConv1dContext(ctx, x, w.In, 131, w)
	if err != context.Canceled || out != nil || n != 0 {
		t.Fatalf("got len=%d n=%d err=%v", len(out), n, err)
	}
}

// Read just the two convolution tensors and their biases, without expanding
// the 32-layer encoder or loading the decoder into memory.
func loadVoxtralCausalConvs(tb testing.TB) [2]VoxtralRealtimeEncoderConv {
	tb.Helper()
	path := os.Getenv("GOPHERLLM_VOXTRAL_MODEL")
	if path == "" {
		tb.Skip("set GOPHERLLM_VOXTRAL_MODEL to a local Voxtral Realtime GGUF")
	}
	mapping, err := OpenMmap(path)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { _ = mapping.Close() })
	data := mapping.Bytes()
	gguf, err := ParseGGUF(data)
	if err != nil {
		tb.Fatal(err)
	}
	tensors, inferred := indexTensors(gguf), inferTensorSizes(data, gguf)
	names := detectVoxtralTensorNames(tensors)
	var convs [2]VoxtralRealtimeEncoderConv
	for i := range convs {
		weightName, biasName := names.conv(i)
		info, exists := tensors[weightName]
		if !exists || len(info.Dims) != 3 || info.Dims[0] != 3 {
			tb.Fatalf("invalid convolution tensor %s", weightName)
		}
		weight, err := loadF32Vec(data, gguf.DataOffset, weightName, tensors, inferred)
		if err != nil {
			tb.Fatal(err)
		}
		bias, err := loadF32Vec(data, gguf.DataOffset, biasName, tensors, inferred)
		if err != nil {
			tb.Fatal(err)
		}
		convs[i] = VoxtralRealtimeEncoderConv{Weight: weight, Bias: bias, Kernel: int(info.Dims[0]), In: int(info.Dims[1]), Out: int(info.Dims[2]), Stride: i + 1}
	}
	return convs
}

func TestVoxtralCausalConvCheckpoint(t *testing.T) {
	convs := loadVoxtralCausalConvs(t)
	ctx := context.Background()
	for i, w := range convs {
		for _, n := range []int{67, 129} {
			x, _ := voxtralConvFixture(w.In, 1, n, w.Stride)
			want, wn, err := referenceVoxtralCausalConv(ctx, x, w.In, n, w)
			if err != nil {
				t.Fatal(err)
			}
			got, gn, err := applyCausalConv1dContext(ctx, x, w.In, n, w)
			if err != nil {
				t.Fatal(err)
			}
			if gn != wn || len(got) != len(want) {
				t.Fatal("output shape changed")
			}
			var maxAbs float64
			for j, v := range want {
				delta := math.Abs(float64(got[j] - v))
				maxAbs = max(maxAbs, delta)
				if math.IsNaN(float64(got[j])) || delta > 2e-5*(1+math.Abs(float64(v))) {
					t.Fatalf("conv%d frames=%d [%d]: %g want %g", i, n, j, got[j], v)
				}
			}
			t.Logf("conv%d frames=%d max_abs_error=%g", i, n, maxAbs)
		}
	}
}

func TestVoxtralCausalConvSmallChannelsKeepDotOrder(t *testing.T) {
	const frames = 97
	for _, shape := range []struct{ in, out int }{{33, 64}, {127, 128}, {128, 127}} {
		for _, stride := range []int{1, 2, 3} {
			t.Run(fmt.Sprintf("in=%d/out=%d/stride=%d", shape.in, shape.out, stride), func(t *testing.T) {
				x, w := voxtralConvFixture(shape.in, shape.out, frames, stride)
				want, wn, err := referenceVoxtralCausalConv(context.Background(), x, w.In, frames, w)
				if err != nil {
					t.Fatal(err)
				}
				got, gn, err := applyCausalConv1dContext(context.Background(), x, w.In, frames, w)
				if err != nil {
					t.Fatal(err)
				}
				if gn != wn || len(got) != len(want) {
					t.Fatal("output shape changed")
				}
				for i, v := range want {
					if math.Float32bits(got[i]) != math.Float32bits(v) {
						t.Fatalf("[%d]: got %g want %g", i, got[i], v)
					}
				}
			})
		}
	}
}
