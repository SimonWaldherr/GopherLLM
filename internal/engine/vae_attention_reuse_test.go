package engine

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
	"time"
)

// Keep the previous layout/allocation paths as independent numerical references.
func qwenVAEAttentionBeforeReuse(v *qwenImage21VAE, s *qi21VAEWorkspace, x diffusionTensor) diffusionTensor {
	n := x.h * x.w
	normalized := diffusionTensor{slices.Clone(x.data), x.c, x.h, x.w}
	qi21VAENorm(normalized, v.attentionNorm, false)
	qkv := v.qkv.forward(s, normalized)
	q, k, values := qkv.data[:x.c*n], qkv.data[x.c*n:2*x.c*n], qkv.data[2*x.c*n:]
	queries := make([]float32, n*x.c)
	for ch := range x.c {
		for p := range n {
			queries[p*x.c+ch] = q[ch*n+p]
		}
	}
	scores := make([]float32, n*n)
	s.gemm(n, n, x.c, queries, k, scores)
	scale := float32(1 / math.Sqrt(float64(x.c)))
	for p := range n {
		row := scores[p*n : (p+1)*n]
		maximum := float32(-math.MaxFloat32)
		for i := range row {
			row[i] *= scale
			maximum = max(maximum, row[i])
		}
		softmaxF64InPlace(row, maximum)
	}
	transposed := make([]float32, n*n)
	for i := range n {
		for j := range n {
			transposed[j*n+i] = scores[i*n+j]
		}
	}
	mixed := diffusionTensor{make([]float32, len(x.data)), x.c, x.h, x.w}
	s.gemm(x.c, n, n, values, transposed, mixed.data)
	y := v.projection.forward(s, mixed)
	for i, value := range x.data {
		y.data[i] += value
	}
	return y
}

func fluxVAEAttentionBeforeReuse(v *flux3VAE, s *flux3VAEWorkspace, x diffusionTensor) diffusionTensor {
	n := x.h * x.w
	y := diffusionTensor{make([]float32, len(x.data)), x.c, x.h, x.w}
	v.attentionNorm.apply(&s.norm, y.data, x.data, n, false)
	flat := func(c qi21VAEConv) []float32 {
		z := c.forward(&s.conv, y)
		out := make([]float32, len(z.data))
		for p := range n {
			for ch := range x.c {
				out[p*x.c+ch] = z.data[ch*n+p]
			}
		}
		return out
	}
	q, k, values := flat(v.q), flat(v.k), flat(v.v)
	mixed := make([]float32, len(x.data))
	imageAttention(s.conv.ctx, q, k, values, mixed, n, n, 1, x.c, 0, 0, &s.scores)
	for p := range n {
		for ch := range x.c {
			y.data[ch*n+p] = mixed[p*x.c+ch]
		}
	}
	y = v.projection.forward(&s.conv, y)
	addInPlace(y.data, x.data)
	return y
}

func vaeAttentionTestModels(channels int) (*qwenImage21VAE, *flux3VAE) {
	conv := func(out, seed int) qi21VAEConv {
		w := make([]float32, out*channels)
		for i := range w {
			w[i] = float32((i+seed)%17-8) / float32(channels)
		}
		bias := make([]float32, out)
		for i := range bias {
			bias[i] = float32((i+seed)%7-3) / 100
		}
		return qi21VAEConv{w, bias, channels, out, 1}
	}
	return &qwenImage21VAE{attentionNorm: onesF32(channels), qkv: conv(3*channels, 1), projection: conv(channels, 2)},
		&flux3VAE{attentionNorm: diffusionNorm{onesF32(channels), make([]float32, channels)},
			q: conv(channels, 1), k: conv(channels, 2), v: conv(channels, 3), projection: conv(channels, 4)}
}

func TestTransposeVAEIntoRectangularTails(t *testing.T) {
	for _, shape := range [][2]int{{0, 0}, {1, 17}, {17, 1}, {31, 33}, {33, 31}, {65, 67}} {
		rows, cols := shape[0], shape[1]
		src, dst := make([]float32, rows*cols), make([]float32, rows*cols)
		bits := []uint32{0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 1, 0x3f800000}
		for i := range src {
			src[i] = math.Float32frombits(bits[i%len(bits)])
		}
		transposeVAEInto(dst, src, rows, cols)
		for row := range rows {
			for col := range cols {
				if math.Float32bits(dst[col*rows+row]) != math.Float32bits(src[row*cols+col]) {
					t.Fatalf("%dx%d: value at %d,%d changed", rows, cols, row, col)
				}
			}
		}
	}
}

func TestVAEAttentionReuseMatchesPrevious(t *testing.T) {
	for _, shape := range [][3]int{{32, 2, 3}, {64, 3, 5}, {64, 4, 4}, {64, 16, 16}, {32, 17, 17}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			c, h, w := shape[0], shape[1], shape[2]
			x := diffusionTensor{benchFloatSlice(c * h * w), c, h, w}
			original := slices.Clone(x.data)
			qwen, flux := vaeAttentionTestModels(c)
			for _, family := range []string{"qwen", "flux"} {
				var got, want diffusionTensor
				if family == "qwen" {
					old, current := &qi21VAEWorkspace{ctx: context.Background()}, &qi21VAEWorkspace{ctx: context.Background()}
					want, got = qwenVAEAttentionBeforeReuse(qwen, old, x), qwen.attention(current, x)
					if old.err != nil || current.err != nil {
						t.Fatalf("attention errors: %v, %v", old.err, current.err)
					}
				} else {
					old, current := &flux3VAEWorkspace{conv: qi21VAEWorkspace{ctx: context.Background()}}, &flux3VAEWorkspace{conv: qi21VAEWorkspace{ctx: context.Background()}}
					want, got = fluxVAEAttentionBeforeReuse(flux, old, x), flux.attention(current, x)
					if old.conv.err != nil || current.conv.err != nil {
						t.Fatalf("attention errors: %v, %v", old.conv.err, current.conv.err)
					}
				}
				for i, value := range got.data {
					if math.Float32bits(value) != math.Float32bits(want.data[i]) {
						t.Fatalf("%s element %d: %.9g, want %.9g", family, i, value, want.data[i])
					}
				}
				if !slices.Equal(x.data, original) {
					t.Fatalf("%s attention modified its input", family)
				}
			}
		})
	}
}

func TestVAEAttentionReuseCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	x := diffusionTensor{benchFloatSlice(32 * 16), 32, 4, 4}
	qwen, flux := vaeAttentionTestModels(32)
	qws := &qi21VAEWorkspace{ctx: ctx}
	fws := &flux3VAEWorkspace{conv: qi21VAEWorkspace{ctx: ctx}}
	for family, got := range map[string]diffusionTensor{"qwen": qwen.attention(qws, x), "flux": flux.attention(fws, x)} {
		if len(got.data) != len(x.data) || &got.data[0] != &x.data[0] {
			t.Fatalf("%s did not preserve input on cancellation", family)
		}
	}
	if qws.err != context.Canceled || fws.conv.err != context.Canceled {
		t.Fatalf("cancelled contexts lost: %v, %v", qws.err, fws.conv.err)
	}
}

var vaeAttentionResult diffusionTensor

func BenchmarkVAEAttentionReuse(b *testing.B) {
	for _, tc := range []struct {
		name    string
		c, h, w int
	}{{"qwen", 768, 16, 16}, {"flux", 512, 32, 32}} {
		qwen, flux := vaeAttentionTestModels(tc.c)
		x := diffusionTensor{benchFloatSlice(tc.c * tc.h * tc.w), tc.c, tc.h, tc.w}
		for _, reference := range []bool{true, false} {
			b.Run(fmt.Sprintf("%s/reference=%t", tc.name, reference), func(b *testing.B) {
				b.ReportAllocs()
				for range b.N {
					if tc.name == "qwen" {
						s := &qi21VAEWorkspace{ctx: context.Background()}
						if reference {
							vaeAttentionResult = qwenVAEAttentionBeforeReuse(qwen, s, x)
						} else {
							vaeAttentionResult = qwen.attention(s, x)
						}
					} else {
						s := &flux3VAEWorkspace{conv: qi21VAEWorkspace{ctx: context.Background()}}
						if reference {
							vaeAttentionResult = fluxVAEAttentionBeforeReuse(flux, s, x)
						} else {
							vaeAttentionResult = flux.attention(s, x)
						}
					}
				}
			})
		}
	}
}

// Alternate the call order so neighboring samples share background load and
// neither implementation consistently gets the warmer caches.
func BenchmarkVAEAttentionPaired(b *testing.B) {
	for _, tc := range []struct {
		name    string
		c, h, w int
	}{{"qwen", 768, 16, 16}, {"flux", 512, 32, 32}} {
		b.Run(tc.name, func(b *testing.B) {
			qwen, flux := vaeAttentionTestModels(tc.c)
			x := diffusionTensor{benchFloatSlice(tc.c * tc.h * tc.w), tc.c, tc.h, tc.w}
			reference, reused := make([]int64, b.N), make([]int64, b.N)
			run := func(old bool) int64 {
				start := time.Now()
				if tc.name == "qwen" {
					s := &qi21VAEWorkspace{ctx: context.Background()}
					if old {
						vaeAttentionResult = qwenVAEAttentionBeforeReuse(qwen, s, x)
					} else {
						vaeAttentionResult = qwen.attention(s, x)
					}
				} else {
					s := &flux3VAEWorkspace{conv: qi21VAEWorkspace{ctx: context.Background()}}
					if old {
						vaeAttentionResult = fluxVAEAttentionBeforeReuse(flux, s, x)
					} else {
						vaeAttentionResult = flux.attention(s, x)
					}
				}
				return time.Since(start).Nanoseconds()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				if i&1 == 0 {
					reference[i], reused[i] = run(true), run(false)
				} else {
					reused[i], reference[i] = run(false), run(true)
				}
			}
			b.StopTimer()
			slices.Sort(reference)
			slices.Sort(reused)
			b.ReportMetric(float64(reference[b.N/2]), "reference-ns/op")
			b.ReportMetric(float64(reused[b.N/2]), "reuse-ns/op")
			b.ReportMetric(float64(reference[b.N/2])/float64(reused[b.N/2]), "speedup")
			b.Logf("reference_sorted_samples_ns=%v reuse_sorted_samples_ns=%v", reference, reused)
		})
	}
}
