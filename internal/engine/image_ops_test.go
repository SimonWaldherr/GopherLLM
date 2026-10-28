package engine

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestImageGroupedAttention(t *testing.T) {
	for _, shape := range [][7]int{
		{5, 7, 4, 2, 8, 0, 3},
		{5, 7, 4, 2, 8, 2, 3},
		{33, 32, 8, 2, 16, 0, 32},     // one representative padding query
		{300, 513, 8, 1, 16, 19, 350}, // multiple query tiles
		{17, 16, 32, 8, 128, 0, 16},   // production head layout
	} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			n, keys, heads, kvHeads, dim, offset, prefix := shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]
			q, k, v := benchFloatSlice(n*heads*dim), benchFloatSlice(keys*kvHeads*dim), benchFloatSlice(keys*kvHeads*dim)
			originalQ, originalK, originalV := slices.Clone(q), slices.Clone(k), slices.Clone(v)
			want := make([]float32, len(q))
			// Independent float64 softmax reference, with explicit GQA indexing.
			scale := 1 / math.Sqrt(float64(dim))
			for token := range n {
				end := keys
				if token+offset < prefix {
					end = token + offset + 1
				}
				for head := range heads {
					prob := make([]float64, end)
					largest := math.Inf(-1)
					for key := range end {
						for d := range dim {
							prob[key] += float64(q[(token*heads+head)*dim+d]) * float64(k[(key*kvHeads+head/(heads/kvHeads))*dim+d])
						}
						prob[key] *= scale
						largest = max(largest, prob[key])
					}
					sum := 0.0
					for i := range prob {
						prob[i] = math.Exp(prob[i] - largest)
						sum += prob[i]
					}
					for d := range dim {
						value := 0.0
						for key, p := range prob {
							value += p / sum * float64(v[(key*kvHeads+head/(heads/kvHeads))*dim+d])
						}
						want[(token*heads+head)*dim+d] = float32(value)
					}
				}
			}
			for _, batched := range []bool{false, true} {
				storage := make([]float32, len(q)+2)
				storage[0], storage[len(storage)-1] = 123, -456
				out := storage[1 : len(storage)-1]
				var scores []float32
				run := func(ctx context.Context) {
					if batched {
						imageGroupedAttention(ctx, q, k, v, out, n, keys, heads, kvHeads, dim, offset, prefix, &scores)
					} else {
						imageGroupedAttentionOnline(ctx, q, k, v, out, n, keys, heads, kvHeads, dim, offset, prefix)
					}
				}
				run(context.Background())
				qi21AssertNear(t, out, want, 5e-6)
				if storage[0] != 123 || storage[len(storage)-1] != -456 || cap(scores) > 1<<20 {
					t.Fatal("attention exceeded its output or score bounds")
				}
				clear(out)
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				run(ctx)
				if !slices.Equal(out, make([]float32, len(out))) {
					t.Fatal("canceled attention wrote output")
				}
			}
			if !slices.Equal(q, originalQ) || !slices.Equal(k, originalK) || !slices.Equal(v, originalV) {
				t.Fatal("attention changed its inputs")
			}
		})
	}
}

func TestImageUpsampleConv(t *testing.T) {
	for _, shape := range [][5]int{{1, 1, 1, 1, 1}, {3, 5, 2, 7, 1}, {3, 5, 1, 1, 3}, {5, 3, 7, 19, 3}, {2, 3, 17, 129, 3}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			in, out, h, w, kernel := shape[0], shape[1], shape[2], shape[3], shape[4]
			conv := qi21VAEConv{benchFloatSlice(out * in * kernel * kernel), benchFloatSlice(out), in, out, kernel}
			x := diffusionTensor{benchFloatSlice(in * h * w), in, h, w}
			original := slices.Clone(x.data)
			s := &qi21VAEWorkspace{ctx: context.Background()}
			want := conv.forward(s, imageUpsampleReference(x))
			// Reuse dirty patch/product buffers to exercise every boundary.
			got := conv.upsample(s, x)
			if s.err != nil {
				t.Fatal(s.err)
			}
			if got.h != 2*h || got.w != 2*w || got.c != out {
				t.Fatal("upsampling changed the output shape")
			}
			qi21AssertNear(t, got.data, want.data, 1e-5)
			if !slices.Equal(x.data, original) {
				t.Fatal("upsampling changed the input")
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			s.ctx = ctx
			conv.upsample(s, x)
			if s.err != context.Canceled {
				t.Fatalf("cancellation error = %v", s.err)
			}
		})
	}
}

func imageUpsampleReference(x diffusionTensor) diffusionTensor {
	y := diffusionTensor{make([]float32, len(x.data)*4), x.c, x.h * 2, x.w * 2}
	for ch := range y.c {
		for yy := range y.h {
			for xx := range y.w {
				y.data[(ch*y.h+yy)*y.w+xx] = x.data[(ch*x.h+yy/2)*x.w+xx/2]
			}
		}
	}
	return y
}

func TestImageProjectionConcurrent(t *testing.T) {
	for id := range 4 {
		t.Run(fmt.Sprint(id), func(t *testing.T) {
			t.Parallel()
			const tokens, rows, cols = 273, 33, 64
			w := Weight{Rows: rows, Cols: cols, Type: GGMLTypeF32, F32: make([]float32, rows*cols)}
			for r := range rows {
				w.F32[r*cols+(r+id)%cols] = 1
			}
			input, output := benchFloatSlice(tokens*cols), make([]float32, tokens*rows)
			for repeat := range 3 {
				for i := range input {
					input[i] += float32(id+repeat) / 16
				}
				if err := qi21Project(context.Background(), w, input, output); err != nil {
					t.Fatal(err)
				}
				for token := range tokens {
					for r := range rows {
						if output[token*rows+r] != input[token*cols+(r+id)%cols] {
							t.Fatalf("concurrent projection corrupted row %d, column %d", token, r)
						}
					}
				}
			}
		})
	}
}

func TestImageConvDestination(t *testing.T) {
	for _, kernel := range []int{1, 3} {
		t.Run(fmt.Sprint(kernel), func(t *testing.T) {
			const in, out, h, w = 3, 5, 7, 19
			conv := qi21VAEConv{benchFloatSlice(out * in * kernel * kernel), benchFloatSlice(out), in, out, kernel}
			x := diffusionTensor{benchFloatSlice(in * h * w), in, h, w}
			original := slices.Clone(x.data)
			s := &qi21VAEWorkspace{ctx: context.Background()}
			want := conv.forward(s, x)
			storage := make([]float32, out*h*w+2)
			storage[0], storage[len(storage)-1] = 123, -456
			for i := 1; i < len(storage)-1; i++ {
				storage[i] = float32(math.NaN())
			}
			// A short view with enough capacity must grow without discarding
			// its backing buffer and must overwrite every old value.
			got := conv.forwardInto(s, x, storage[1:2:len(storage)-1])
			if s.err != nil || &got.data[0] != &storage[1] {
				t.Fatalf("destination was not reused: %v", s.err)
			}
			qi21AssertNear(t, got.data, want.data, 1e-6)
			if storage[0] != 123 || storage[len(storage)-1] != -456 || !slices.Equal(x.data, original) {
				t.Fatal("convolution changed input or destination boundaries")
			}
			// Insufficient capacity still produces the complete output.
			got = conv.forwardInto(s, x, make([]float32, 1))
			qi21AssertNear(t, got.data, want.data, 1e-6)
			beforeCancel := slices.Clone(storage)
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			s.ctx = ctx
			conv.forwardInto(s, x, storage[1:len(storage)-1])
			if s.err != context.Canceled || !slices.Equal(storage, beforeCancel) {
				t.Fatal("canceled convolution changed the destination")
			}
		})
	}
}
